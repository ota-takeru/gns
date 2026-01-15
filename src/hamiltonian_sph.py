"""Central Conservative + Folded Dissipation (+ optional wall particles / Wall-MLP)

This matches the "GNS-style" idea:
  - Default: no wall particles.
  - Optional: wall particles (kinematic) can be added to the dataset and used
    for reflection via pair interactions.
  - (Legacy) Each particle can also get wall-distance features and a small wall
    network outputs per-particle wall acceleration a_wall.
  - Pair interactions remain rotation-safe (dist-only center forces + folded radial dissipation).
  - Rollout may use boundary reflection as a safety guardrail.
  - Training uses instantaneous acceleration a(x_t, v_t) and target Δv (= a*dt^2)
    computed in the SAME noisy world as the model input.

Key choices (stable baseline):
  1) Undirected unique pairs (don't drop interactions due to edge direction / truncation).
  2) Smooth cutoff w(d) (cosine) for pair forces and dissipation.
  3) Conservative term learns center-force magnitude phi(d) directly (dist-only).
  4) Dissipation is folded radial damping with bounded alpha(d, |s|).
  5) Wall term (optional MLP) is axis-aligned (because a box boundary breaks rotation symmetry anyway):
       For each axis k:
         dL = x_k - lower_k
         dU = upper_k - x_k
         v_toward_lower = relu(-v_k)
         v_toward_upper = relu(+v_k)
         magL = f_wall(dL, v_toward_lower) >= 0
         magU = f_wall(dU, v_toward_upper) >= 0
         a_wall_k = +magL - magU
     This guarantees "push inward" direction without wall particles.

Notes:
  - This is intended for axis-aligned box boundaries.
  - If you want arbitrary walls, use SDF distance + normal instead (still no wall particles).

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import radius_graph

BOUNDARY_MODE_WALLS = "walls"
BOUNDARY_MODE_PERIODIC = "periodic"

import graph_network
from learned_simulator import BaseSimulator


# ------------------------------------------------------------
# Configs
@dataclass
class SPHConfig:
    """Neighborhood settings only (for radius graph)."""

    smoothing_length: float | None = None
    max_num_neighbors: int = 128


@dataclass
class ConservativeConfig:
    """Center-force magnitude phi(d) network settings.

    phi(d) has units of acceleration (since we add phi(d)*r_hat to acceleration).
    We bound output with tanh for stability:
      phi = phi_max * tanh(raw)
    """

    mlp_layers: int = 2
    mlp_hidden_dim: int = 128
    dropout: float = 0.0
    phi_max_multiplier: float = 2.0  # phi_max = multiplier * (pos_scale / dt^2)
    use_density: bool = False  # optional density inputs (rho_avg, rho_diff)


@dataclass
class DissipationConfig:
    """Folded dissipation alpha(d, |s|) settings.

    alpha is dimensionless, bounded to [0, alpha_max]:
      alpha = alpha_max * sigmoid(raw)

    We also clip s in "scaled" space for robustness:
      s_scaled = s / vel_scale
      s_scaled_clipped = clamp(s_scaled, -s_clip_scaled, +s_clip_scaled)
      s_used = s_scaled_clipped * vel_scale
    """

    mlp_layers: int = 2
    mlp_hidden_dim: int = 128
    dropout: float = 0.0
    alpha_max: float = 1.0
    s_clip_scaled: float = 5.0


@dataclass
class TermConfig:
    """Ablation switches / weights for pairwise terms."""

    enable_central: bool = True
    enable_pressure: bool = False
    enable_contact: bool = False
    enable_damping: bool = True

    w_central: float = 1.0
    w_pressure: float = 1.0
    w_contact: float = 1.0
    w_damping: float = 1.0


@dataclass
class PressureConfig:
    """Density-based repulsive magnitude (pressure-like) term."""

    enabled: bool = False
    mlp_layers: int = 2
    mlp_hidden_dim: int = 128
    dropout: float = 0.0
    rep_max_multiplier: float = 2.0
    use_compression_gate: bool = True


@dataclass
class ContactConfig:
    """Approaching-only impulse term."""

    enabled: bool = False
    mlp_layers: int = 2
    mlp_hidden_dim: int = 128
    dropout: float = 0.0
    alpha_max: float = 10.0
    s_clip_scaled: float = 50.0


@dataclass
class WallConfig:
    """Wall-MLP settings (no wall particles).

    We predict nonnegative magnitudes for each axis and each side (lower/upper),
    then combine with correct direction:
      a_wall_k = +mag_lower_k - mag_upper_k

    Magnitude head:
      mag = near(d) * a_wall_max * sigmoid(raw)
    where near(d) = exp(-d / d0) (fixed shape, learn only the modulation).

    d0 is tied to smoothing_length:
      d0 = d0_multiplier * h
    """

    enabled: bool = True
    mlp_layers: int = 2
    mlp_hidden_dim: int = 64
    dropout: float = 0.0
    a_wall_max_multiplier: float = 1.0  # a_wall_max = multiplier * (pos_scale / dt^2)
    d0_multiplier: float = 0.5  # d0 = multiplier * h
    use_velocity_gate: bool = True  # gate by velocity to avoid distance-only pushing


@dataclass
class WallParticleConfig:
    """Wall particle settings (kinematic particles included in the dataset).

    These particles interact through the same pair forces, but are kept fixed.
    """

    enabled: bool = False
    particle_type_id: int = 3  # KINEMATIC_PARTICLE_ID
    freeze_walls: bool = True  # keep wall particles fixed in integration
    zero_wall_acceleration: bool = True  # zero out wall particle accelerations
    disable_wall_mlp: bool = True  # disable Wall-MLP when wall particles are used
    disable_rollout_reflection: bool = True  # avoid double reflection


@dataclass
class CutoffConfig:
    """Smooth cutoff weight w(d) with cosine ramp."""

    kind: Literal["cosine"] = "cosine"


@dataclass
class IntegratorConfig:
    dt: float = 1e-3
    type: Literal["leapfrog", "symplectic_euler"] = "leapfrog"


# ------------------------------------------------------------
# Networks
class PairConservativePhiNetDistOnly(nn.Module):
    """phi = f(dist) -> scalar (acceleration magnitude along r_hat)"""

    def __init__(
        self,
        cfg: ConservativeConfig,
        *,
        pos_scale: float,
        dt: float,
        typical_acc: float | None = None,
    ) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._dt = float(dt)
        self._drop = nn.Dropout(cfg.dropout)

        # acceleration scale ~ typ acc (fallback: pos/dt^2)
        a_typ = (
            float(typical_acc)
            if typical_acc is not None
            else (self._pos_scale / max(self._dt * self._dt, 1e-12))
        )
        self._phi_max = float(cfg.phi_max_multiplier) * a_typ

        self._mlp = graph_network.build_mlp(
            input_size=1,
            hidden_layer_sizes=[cfg.mlp_hidden_dim for _ in range(cfg.mlp_layers)],
            output_size=1,
            activation=nn.SiLU,
            output_activation=nn.Identity,
        )

    def forward(self, *, dist: torch.Tensor) -> torch.Tensor:
        x = (dist / self._pos_scale).unsqueeze(-1)  # (P,1)
        x = self._drop(x)
        raw = self._mlp(x).squeeze(-1)  # (P,)
        return self._phi_max * torch.tanh(raw)  # bounded accel magnitude


class PairConservativePhiNetWithDensity(nn.Module):
    """phi = f(dist, rho_avg, rho_diff) -> scalar (accel magnitude along r_hat)

    rho inputs are expected to be pre-scaled (e.g., divide by rho0).
    """

    def __init__(
        self,
        cfg: ConservativeConfig,
        *,
        pos_scale: float,
        dt: float,
        typical_acc: float | None = None,
    ) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._dt = float(dt)
        self._drop = nn.Dropout(cfg.dropout)

        a_typ = (
            float(typical_acc)
            if typical_acc is not None
            else (self._pos_scale / max(self._dt * self._dt, 1e-12))
        )
        self._phi_max = float(cfg.phi_max_multiplier) * a_typ

        self._mlp = graph_network.build_mlp(
            input_size=3,  # [d_scaled, rho_avg_scaled, rho_diff_scaled]
            hidden_layer_sizes=[cfg.mlp_hidden_dim for _ in range(cfg.mlp_layers)],
            output_size=1,
            activation=nn.SiLU,
            output_activation=nn.Identity,
        )

    def forward(
        self,
        *,
        dist: torch.Tensor,
        rho_avg_scaled: torch.Tensor,
        rho_diff_scaled: torch.Tensor,
    ) -> torch.Tensor:
        d_scaled = (dist / self._pos_scale).unsqueeze(-1)  # (P,1)
        rho_avg_scaled = rho_avg_scaled.unsqueeze(-1)
        rho_diff_scaled = rho_diff_scaled.unsqueeze(-1)
        x = torch.cat([d_scaled, rho_avg_scaled, rho_diff_scaled], dim=-1)  # (P,3)
        x = self._drop(x)
        raw = self._mlp(x).squeeze(-1)  # (P,)
        return self._phi_max * torch.tanh(raw)


class PairDissipationAlphaNet(nn.Module):
    """alpha = g(d, |s|) in [0, alpha_max]
    d = ||r||
    s = v_rel · r_hat
    """

    def __init__(self, cfg: DissipationConfig, *, pos_scale: float, vel_scale: float) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._vel_scale = float(vel_scale)
        self._alpha_max = float(cfg.alpha_max)
        self._drop = nn.Dropout(cfg.dropout)

        self._mlp = graph_network.build_mlp(
            input_size=2,  # [d, |s|] (both scaled)
            hidden_layer_sizes=[cfg.mlp_hidden_dim for _ in range(cfg.mlp_layers)],
            output_size=1,
            activation=nn.SiLU,
            output_activation=nn.Identity,
        )

    def forward(self, *, dist: torch.Tensor, s_abs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                (dist / self._pos_scale).unsqueeze(-1),
                (s_abs / self._vel_scale).unsqueeze(-1),
            ],
            dim=-1,
        )  # (P,2)
        x = self._drop(x)
        raw = self._mlp(x).squeeze(-1)
        return self._alpha_max * torch.sigmoid(raw)


class PairPressureRepulsionNet(nn.Module):
    """rep = f(d, rho_rel_avg, rho_rel_diff) -> nonnegative repulsive magnitude"""

    def __init__(
        self,
        cfg: PressureConfig,
        *,
        pos_scale: float,
        dt: float,
        typical_acc: float | None = None,
    ) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._dt = float(dt)
        self._drop = nn.Dropout(cfg.dropout)

        a_typ = (
            float(typical_acc)
            if typical_acc is not None
            else (self._pos_scale / max(self._dt * self._dt, 1e-12))
        )
        self._rep_max = float(cfg.rep_max_multiplier) * a_typ

        self._mlp = graph_network.build_mlp(
            input_size=3,  # [d_scaled, rho_rel_avg, rho_rel_diff]
            hidden_layer_sizes=[cfg.mlp_hidden_dim for _ in range(cfg.mlp_layers)],
            output_size=1,
            activation=nn.SiLU,
            output_activation=nn.Identity,
        )

    def forward(
        self,
        *,
        dist: torch.Tensor,
        rho_rel_avg: torch.Tensor,
        rho_rel_diff: torch.Tensor,
    ) -> torch.Tensor:
        d_scaled = (dist / (self._pos_scale + 1e-12)).unsqueeze(-1)
        x = torch.cat(
            [d_scaled, rho_rel_avg.unsqueeze(-1), rho_rel_diff.unsqueeze(-1)],
            dim=-1,
        )
        x = self._drop(x)
        raw = self._mlp(x).squeeze(-1)
        return self._rep_max * torch.sigmoid(raw)


class PairContactAlphaNet(nn.Module):
    """alpha_contact = g(d, s_neg) in [0, alpha_max]"""

    def __init__(self, cfg: ContactConfig, *, pos_scale: float, vel_scale: float) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._vel_scale = float(vel_scale)
        self._alpha_max = float(cfg.alpha_max)
        self._drop = nn.Dropout(cfg.dropout)

        self._mlp = graph_network.build_mlp(
            input_size=2,  # [d_scaled, s_neg_scaled]
            hidden_layer_sizes=[cfg.mlp_hidden_dim for _ in range(cfg.mlp_layers)],
            output_size=1,
            activation=nn.SiLU,
            output_activation=nn.Identity,
        )

    def forward(self, *, dist: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                (dist / (self._pos_scale + 1e-12)).unsqueeze(-1),
                (s_neg / (self._vel_scale + 1e-12)).unsqueeze(-1),
            ],
            dim=-1,
        )
        x = self._drop(x)
        raw = self._mlp(x).squeeze(-1)
        return self._alpha_max * torch.sigmoid(raw)


class WallMagnitudeNet(nn.Module):
    """mag = near(d) * a_wall_max * sigmoid(raw(d, v_toward))

    Inputs are scaled:
      d_scaled = d / pos_scale
      v_scaled = v_toward / vel_scale
    """

    def __init__(
        self,
        cfg: WallConfig,
        *,
        pos_scale: float,
        vel_scale: float,
        dt: float,
        typical_acc: float | None = None,
    ) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._vel_scale = float(vel_scale)
        self._dt = float(dt)
        self._drop = nn.Dropout(cfg.dropout)

        a_typ = (
            float(typical_acc)
            if typical_acc is not None
            else (self._pos_scale / max(self._dt * self._dt, 1e-12))
        )
        self._a_wall_max = float(cfg.a_wall_max_multiplier) * a_typ
        self._use_velocity_gate = bool(cfg.use_velocity_gate)

        self._mlp = graph_network.build_mlp(
            input_size=2,  # [d_scaled, v_toward_scaled]
            hidden_layer_sizes=[cfg.mlp_hidden_dim for _ in range(cfg.mlp_layers)],
            output_size=1,
            activation=nn.SiLU,
            output_activation=nn.Identity,
        )

    def forward(self, *, d: torch.Tensor, v_toward: torch.Tensor, d0: float) -> torch.Tensor:
        # d: (N,) >= 0 ; v_toward: (N,) >= 0
        d_scaled = d / (self._pos_scale + 1e-12)
        if self._use_velocity_gate:
            v_scaled = v_toward / (self._vel_scale + 1e-12)
        else:
            v_scaled = torch.zeros_like(v_toward)

        x = torch.stack([d_scaled, v_scaled], dim=-1)  # (N,2)
        x = self._drop(x)
        raw = self._mlp(x).squeeze(-1)  # (N,)
        # shift raw so that initial output is almost zero
        base = self._a_wall_max * torch.sigmoid(raw - 6.0)  # (N,)

        # Hard gate: if not moving toward the wall, zero the magnitude completely.
        if self._use_velocity_gate:
            gate = (v_toward > 0).to(base.dtype)
            base = base * gate

        # near(d): exponential decay with d0 tied to h
        d0_t = torch.as_tensor(d0, device=d.device, dtype=d.dtype)
        near = torch.exp(-d / (d0_t + 1e-12))
        return near * base


# ------------------------------------------------------------
# Simulator
class HamiltonianSPHVarAWithDissipation(BaseSimulator):
    def __init__(
        self,
        particle_dimensions: int,
        nnode_in: int,
        nedge_in: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        connectivity_radius: float,
        normalization_stats: dict,
        nparticle_types: int,
        particle_type_embedding_size: int,
        boundaries,
        boundary_clamp_limit: float = 1.0,
        boundary_mode: str = BOUNDARY_MODE_WALLS,
        device: torch.device | str = "cpu",
        *,
        sph: SPHConfig | None = None,
        conservative: ConservativeConfig | None = None,
        dissipation: DissipationConfig | None = None,
        wall: WallConfig | None = None,
        wall_particles: WallParticleConfig | None = None,
        cutoff: CutoffConfig | None = None,
        integrator: IntegratorConfig | None = None,
        particle_mass: float = 1.0,
        gravity: torch.Tensor | None = None,
        pos_feature_scale: float | None = None,
        vel_feature_scale: float | None = None,
        include_external_acceleration: bool = True,
        terms: TermConfig | None = None,
        pressure: PressureConfig | None = None,
        contact: ContactConfig | None = None,
        rollout_reflect_walls: bool = True,
    ) -> None:
        super().__init__()
        # interface placeholders (kept for compatibility)
        del (
            nnode_in,
            nedge_in,
            latent_dim,
            nmessage_passing_steps,
            nparticle_types,
            particle_type_embedding_size,
            boundary_clamp_limit,
        )
        del particle_mass  # not used in this baseline (all accel-based)

        self._dim = int(particle_dimensions)
        self._device = torch.device(device)
        self._connectivity_radius = float(connectivity_radius)
        self._normalization_stats = normalization_stats

        boundaries_arr = torch.as_tensor(boundaries, dtype=torch.float32)
        if boundaries_arr.ndim != 2 or boundaries_arr.shape != (self._dim, 2):
            raise ValueError(
                f"Expected boundaries shape ({self._dim},2); got {tuple(boundaries_arr.shape)}"
            )
        self._boundaries = boundaries_arr
        self._boundary_mode = self._normalize_boundary_mode(boundary_mode)
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            lengths = boundaries_arr[:, 1] - boundaries_arr[:, 0]
            if torch.any(lengths <= 0):
                raise ValueError("Periodic boundaries require bounds with positive length.")
            self._periodic_length = lengths
        else:
            self._periodic_length = None
        self._boundary_restitution = 0.5
        self._rollout_reflect_walls = bool(rollout_reflect_walls)

        def _coerce(obj, cls, fallback):
            if obj is None:
                return fallback
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, dict):
                return cls(**obj)
            raise TypeError(f"Expected {cls.__name__} or dict, got {type(obj)}")

        self._sph = _coerce(sph, SPHConfig, SPHConfig())
        self._int_cfg = _coerce(integrator, IntegratorConfig, IntegratorConfig())
        self._cut_cfg = _coerce(cutoff, CutoffConfig, CutoffConfig())
        self._dt = float(self._int_cfg.dt)

        # データ統計に合わせた代表加速度／速度スケールを推定
        acc_stats = normalization_stats.get("acceleration", {})
        acc_std = torch.as_tensor(acc_stats.get("std", 0.0), dtype=torch.float32)
        acc_std_max = float(acc_std.abs().max().item()) if acc_std.numel() > 0 else 0.0
        self._typical_acc = acc_std_max / max(self._dt * self._dt, 1e-12)

        if self._sph.smoothing_length is None:
            # set so that cutoff radius ~= 2h
            self._sph.smoothing_length = self._connectivity_radius * 0.5

        self._cons_cfg = _coerce(
            conservative,
            ConservativeConfig,
            ConservativeConfig(mlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim, dropout=0.0),
        )
        self._diss_cfg = _coerce(
            dissipation,
            DissipationConfig,
            DissipationConfig(mlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim, dropout=0.0),
        )
        self._wall_cfg = _coerce(wall, WallConfig, WallConfig())
        self._wall_particles_cfg = _coerce(wall_particles, WallParticleConfig, WallParticleConfig())
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            self._rollout_reflect_walls = False
            self._wall_cfg.enabled = False
        self._terms_cfg = _coerce(terms, TermConfig, TermConfig())
        self._pressure_cfg = _coerce(pressure, PressureConfig, PressureConfig())
        self._contact_cfg = _coerce(contact, ContactConfig, ContactConfig())
        self._use_density = bool(getattr(self._cons_cfg, "use_density", False))

        # Scales for stable feature scaling
        self._pos_scale = float(pos_feature_scale or (2.0 * float(self._sph.smoothing_length)))

        if vel_feature_scale is not None:
            self._vel_scale = float(vel_feature_scale)
        else:
            vel_stats = normalization_stats.get("velocity", {})
            vel_std = torch.as_tensor(vel_stats.get("std", 0.0), dtype=torch.float32)
            vel_std_max = float(vel_std.abs().max().item()) if vel_std.numel() > 0 else 0.0
            dt_safe = max(self._dt, 1e-8)
            if vel_std_max > 0:
                # dataset velocity stats are per-step displacements; convert to velocity scale
                self._vel_scale = float(vel_std_max / dt_safe)
            else:
                self._vel_scale = float(self._pos_scale / dt_safe)

        # Pair nets
        if self._use_density:
            self._phi_net = PairConservativePhiNetWithDensity(
                cfg=self._cons_cfg,
                pos_scale=self._pos_scale,
                dt=self._dt,
                typical_acc=self._typical_acc if self._typical_acc > 0 else None,
            ).to(self._device)
        else:
            self._phi_net = PairConservativePhiNetDistOnly(
                cfg=self._cons_cfg,
                pos_scale=self._pos_scale,
                dt=self._dt,
                typical_acc=self._typical_acc if self._typical_acc > 0 else None,
            ).to(self._device)

        self._alpha_net = PairDissipationAlphaNet(
            cfg=self._diss_cfg,
            pos_scale=self._pos_scale,
            vel_scale=self._vel_scale,
        ).to(self._device)

        self._pressure_net: PairPressureRepulsionNet | None = None
        if self._pressure_cfg.enabled:
            self._pressure_net = PairPressureRepulsionNet(
                cfg=self._pressure_cfg,
                pos_scale=self._pos_scale,
                dt=self._dt,
                typical_acc=self._typical_acc if self._typical_acc > 0 else None,
            ).to(self._device)

        self._contact_alpha_net: PairContactAlphaNet | None = None
        if self._contact_cfg.enabled:
            self._contact_alpha_net = PairContactAlphaNet(
                cfg=self._contact_cfg,
                pos_scale=self._pos_scale,
                vel_scale=self._vel_scale,
            ).to(self._device)

        # Wall net (no wall particles)
        if self._wall_particles_cfg.enabled and self._wall_particles_cfg.disable_wall_mlp:
            self._wall_cfg.enabled = False

        self._wall_mag_net: WallMagnitudeNet | None
        if self._wall_cfg.enabled:
            self._wall_mag_net = WallMagnitudeNet(
                cfg=self._wall_cfg,
                pos_scale=self._pos_scale,
                vel_scale=self._vel_scale,
                dt=self._dt,
                typical_acc=self._typical_acc if self._typical_acc > 0 else None,
            ).to(self._device)
        else:
            self._wall_mag_net = None

        # If wall particles are enabled, optionally disable rollout reflection.
        if self._wall_particles_cfg.enabled and self._wall_particles_cfg.disable_rollout_reflection:
            self._rollout_reflect_walls = False

        # External acceleration (gravity)
        if gravity is not None:
            g = torch.as_tensor(gravity, dtype=torch.float32, device=self._device)
            if g.numel() >= self._dim:
                g = g[: self._dim]
            elif g.numel() == 1:
                g = g.repeat(self._dim)
            else:
                raise ValueError("gravity の次元が粒子次元と合いません。")
            self._gravity = g
        else:
            self._gravity = None
        self._include_external_acceleration = bool(include_external_acceleration)
        # 直近バッチの統計を取るかどうか（ログ用）。デフォルトで有効。
        self._record_debug_stats = True
        self._last_debug_stats: dict[str, float | None] | None = None
        self._neighbor_debug_logged: bool = False
        self._neighbor_debug_info: dict[str, str | bool] | None = None
        # proxy density scale (EMA of mean rho)
        self.register_buffer("_rho_ema", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------
    # Graph construction
    def _compute_graph_connectivity(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        *,
        add_self_edges: bool = False,
    ) -> torch.Tensor:
        counts = nparticles_per_example.to(node_position.device, dtype=torch.long)
        batch_ids = torch.repeat_interleave(
            torch.arange(len(counts), device=node_position.device), counts
        )
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            backend = "dense_radius_graph_periodic"
            backend_note = None
            edge_index = self._dense_radius_graph_periodic(
                node_position, nparticles_per_example, radius, add_self_edges
            )
        else:
            backend = "radius_graph"
            backend_note = None
            try:
                edge_index = radius_graph(
                    node_position,
                    r=radius,
                    batch=batch_ids,
                    loop=add_self_edges,
                    max_num_neighbors=self._sph.max_num_neighbors,
                )
            except (ImportError, RuntimeError) as e:
                backend = "dense_radius_graph"
                backend_note = f"{type(e).__name__}: {e}"
                edge_index = self._dense_radius_graph(
                    node_position, nparticles_per_example, radius, add_self_edges
                )
        self._log_neighbor_backend(
            backend=backend,
            node_position=node_position,
            batch_device=batch_ids.device,
            note=backend_note,
        )
        return edge_index

    def _dense_radius_graph(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        add_self_edges: bool,
    ) -> torch.Tensor:
        device = node_position.device
        edges: list[torch.Tensor] = []
        start_idx = 0
        eps = 1e-8

        for n in nparticles_per_example.tolist():
            n = int(n)
            if n == 0:
                continue
            slice_pos = node_position[start_idx : start_idx + n]
            dist_mat = torch.cdist(slice_pos, slice_pos)
            mask = dist_mat <= (radius + eps)
            if not add_self_edges:
                idx = torch.arange(n, device=device)
                mask[idx, idx] = False
            r_idx, s_idx = torch.nonzero(mask, as_tuple=True)
            if r_idx.numel() > 0:
                # mimic radius_graph: edge_index[0]=receivers, edge_index[1]=senders
                receivers = r_idx + start_idx
                senders = s_idx + start_idx
                edges.append(torch.stack([receivers, senders], dim=0))
            start_idx += n

        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(edges, dim=1)

    def _dense_radius_graph_periodic(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        add_self_edges: bool,
    ) -> torch.Tensor:
        device = node_position.device
        edges: list[torch.Tensor] = []
        start_idx = 0
        eps = 1e-8

        for n in nparticles_per_example.tolist():
            n = int(n)
            if n == 0:
                continue
            slice_pos = node_position[start_idx : start_idx + n]
            delta = slice_pos[:, None, :] - slice_pos[None, :, :]
            delta = self._minimum_image_displacement(delta)
            dist_mat = torch.linalg.norm(delta, dim=-1)
            mask = dist_mat <= (radius + eps)
            if not add_self_edges:
                idx = torch.arange(n, device=device)
                mask[idx, idx] = False
            r_idx, s_idx = torch.nonzero(mask, as_tuple=True)
            if r_idx.numel() > 0:
                receivers = r_idx + start_idx
                senders = s_idx + start_idx
                edges.append(torch.stack([receivers, senders], dim=0))
            start_idx += n

        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(edges, dim=1)

    def _log_neighbor_backend(
        self,
        *,
        backend: str,
        node_position: torch.Tensor,
        batch_device: torch.device,
        note: str | None,
    ) -> None:
        """Log once whether neighbor search runs on GPU or CPU."""
        if self._neighbor_debug_logged:
            return
        uses_gpu = node_position.is_cuda
        info: dict[str, str | bool] = {
            "backend": backend,
            "node_device": str(node_position.device),
            "batch_device": str(batch_device),
            "uses_gpu": uses_gpu,
            "torch_cuda_available": torch.cuda.is_available(),
        }
        if note:
            info["note"] = str(note).replace("\n", " ")
        self._neighbor_debug_info = info
        parts = [f"{k}={v}" for k, v in info.items()]
        print("[neighbor][HamiltonianSPH]", " ".join(parts))
        self._neighbor_debug_logged = True

    def _normalize_boundary_mode(self, mode: str) -> str:
        value = str(mode).strip().lower()
        if value not in {BOUNDARY_MODE_WALLS, BOUNDARY_MODE_PERIODIC}:
            raise ValueError(
                f"Unsupported boundary_mode '{mode}'. Use '{BOUNDARY_MODE_WALLS}' or '{BOUNDARY_MODE_PERIODIC}'."
            )
        return value

    def _wrap_positions(
        self,
        x: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        wall_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        span = upper - lower
        x_wrap = lower + torch.remainder(x - lower, span)
        if wall_mask is None:
            return x_wrap
        mask = wall_mask[:, None].to(dtype=torch.bool, device=x_wrap.device)
        return torch.where(mask, x, x_wrap)

    def _minimum_image_displacement(self, delta: torch.Tensor) -> torch.Tensor:
        if self._boundary_mode != BOUNDARY_MODE_PERIODIC:
            return delta
        if self._periodic_length is None:
            raise RuntimeError("Periodic lengths are not available.")
        length = self._periodic_length.to(device=delta.device, dtype=delta.dtype)
        half = 0.5 * length
        return torch.remainder(delta + half, length) - half

    def _relative_displacement(self, pos_a: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
        delta = pos_a - pos_b
        return self._minimum_image_displacement(delta)

    def get_neighbor_debug_info(self) -> dict[str, str | bool] | None:
        """Return cached neighbor-search device info (first call only)."""
        return self._neighbor_debug_info

    def _build_edges(self, x: torch.Tensor, nparticles_per_example: torch.Tensor) -> torch.Tensor:
        rc = 2.0 * float(self._sph.smoothing_length)
        edge_index_rs = self._compute_graph_connectivity(
            x, nparticles_per_example, rc, add_self_edges=False
        )
        # Convert to (senders -> receivers)
        receivers = edge_index_rs[0]
        senders = edge_index_rs[1]
        return torch.stack([senders, receivers], dim=0)  # (2,E)

    def _build_pairs(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Directed edges -> unique undirected pairs.
        Prevents dropping interactions due to edge direction / neighbor truncation.
        """
        s = edge_index[0]
        r = edge_index[1]
        if s.numel() == 0:
            return None

        i = torch.minimum(s, r)
        j = torch.maximum(s, r)
        pairs = torch.stack([i, j], dim=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.numel() == 0:
            return None

        pairs = torch.unique(pairs, dim=0)
        return pairs[:, 0], pairs[:, 1]

    def _pair_geometry(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> (
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        """Compute pair-wise rel/dist/r_hat and cutoff weight."""
        pairs = self._build_pairs(edge_index)
        if pairs is None:
            return None
        i_idx, j_idx = pairs
        rel = self._relative_displacement(x[j_idx], x[i_idx])  # (P,dim)
        dist = torch.sqrt((rel * rel).sum(dim=-1) + 1e-12)  # (P,)
        r_hat = rel / (dist.unsqueeze(-1) + 1e-8)  # (P,dim)
        w = self._cutoff_weight(dist)
        return (i_idx, j_idx), rel, dist, r_hat, w

    def _proxy_density(
        self, *, pairs: tuple[torch.Tensor, torch.Tensor], w: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """ρ_i = Σ_j w(d_ij) using cutoff weights (proxy density)."""
        rho = torch.zeros(num_nodes, device=w.device, dtype=w.dtype)
        i_idx, j_idx = pairs
        rho = rho.index_add(0, i_idx, w)
        rho = rho.index_add(0, j_idx, w)
        return rho

    # ------------------------------------------------------------
    # Smooth cutoff for pair terms
    def _cutoff_weight(self, dist: torch.Tensor) -> torch.Tensor:
        rc = 2.0 * float(self._sph.smoothing_length)
        rc_t = torch.as_tensor(rc, device=dist.device, dtype=dist.dtype)
        x = dist / (rc_t + 1e-12)
        w = 0.5 * (torch.cos(math.pi * x) + 1.0)
        w = torch.where(dist < rc_t, w, torch.zeros_like(w))
        return w

    # ------------------------------------------------------------
    # Rollout boundary reflection (safety guardrail)
    def _apply_boundary_conditions(
        self, x: torch.Tensor, v: torch.Tensor, mask_exclude: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundaries = self._boundaries.to(device=x.device, dtype=x.dtype)
        lower = boundaries[:, 0]
        upper = boundaries[:, 1]
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            return self._wrap_positions(x, lower, upper, mask_exclude), v

        below = x < lower
        above = x > upper
        if not (below.any() or above.any()):
            return x, v

        x_ref = x.clone()
        v_ref = v.clone()

        x_ref = torch.where(below, lower + (lower - x_ref), x_ref)
        v_ref = torch.where(below, -v_ref * self._boundary_restitution, v_ref)

        x_ref = torch.where(above, upper - (x_ref - upper), x_ref)
        v_ref = torch.where(above, -v_ref * self._boundary_restitution, v_ref)

        x_ref = torch.min(torch.max(x_ref, lower), upper)

        if mask_exclude is not None:
            mask = mask_exclude[:, None].to(dtype=torch.bool, device=x_ref.device)
            x_ref = torch.where(mask, x, x_ref)
            v_ref = torch.where(mask, v, v_ref)

        return x_ref, v_ref

    def _wall_particle_mask(
        self, particle_types: torch.Tensor | None, num_nodes: int, device: torch.device
    ) -> torch.Tensor | None:
        if not self._wall_particles_cfg.enabled or particle_types is None:
            return None
        if particle_types.numel() != num_nodes:
            return None
        wall_id = int(self._wall_particles_cfg.particle_type_id)
        return particle_types.to(device=device) == wall_id

    def _clamp_positions(
        self,
        x: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        wall_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            return self._wrap_positions(x, lower, upper, wall_mask)
        x_clamped = torch.min(torch.max(x, lower), upper)
        if wall_mask is None:
            return x_clamped
        mask = wall_mask[:, None].to(dtype=torch.bool, device=x_clamped.device)
        return torch.where(mask, x, x_clamped)

    # ------------------------------------------------------------
    # Wall acceleration (no wall particles)
    def _wall_acc(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            return torch.zeros_like(x)
        if self._wall_mag_net is None:
            return torch.zeros_like(x)

        boundaries = self._boundaries.to(device=x.device, dtype=x.dtype)
        lower = boundaries[:, 0]
        upper = boundaries[:, 1]

        # For features, clamp x into the box (avoids negative distances due to noise)
        x_in = torch.min(torch.max(x, lower), upper)

        dL = x_in - lower  # (N,dim) >=0
        dU = upper - x_in  # (N,dim) >=0

        # d0 tied to h
        h = float(self._sph.smoothing_length)
        d0 = float(self._wall_cfg.d0_multiplier) * h

        a_wall = torch.zeros_like(x)
        for k in range(self._dim):
            v_k = v[:, k]
            if self._wall_cfg.use_velocity_gate:
                v_tow_lower = torch.relu(-v_k)  # approaching lower wall
                v_tow_upper = torch.relu(+v_k)  # approaching upper wall
            else:
                # distance-only gating: ignore velocity
                v_tow_lower = torch.zeros_like(v_k)
                v_tow_upper = torch.zeros_like(v_k)

            magL = self._wall_mag_net(d=dL[:, k], v_toward=v_tow_lower, d0=d0)  # (N,) >=0
            magU = self._wall_mag_net(d=dU[:, k], v_toward=v_tow_upper, d0=d0)  # (N,) >=0

            a_wall[:, k] = a_wall[:, k] + magL - magU

        return a_wall

    # ------------------------------------------------------------
    # Pair accelerations (instantaneous)
    def _conservative_acc(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        collect_debug: bool = False,
        pair_geom: tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        geom = pair_geom if pair_geom is not None else self._pair_geometry(x, edge_index)
        if geom is None:
            zero = torch.zeros_like(x)
            return (zero, {"phi": None, "dist": None, "rho": None}) if collect_debug else zero

        pairs, rel, dist, r_hat, w = geom
        i_idx, j_idx = pairs

        rho: torch.Tensor | None
        rho_scale: torch.Tensor | None
        if self._use_density:
            rho = self._proxy_density(pairs=pairs, w=w, num_nodes=x.shape[0])
            # update EMA during training (detach to avoid grads)
            rho_mean = rho.detach().mean()
            decay = 0.99
            if self.training:
                self._rho_ema.mul_(decay).add_((1.0 - decay) * rho_mean)
            rho_scale = torch.clamp(self._rho_ema.to(rho.device, rho.dtype), min=1e-6)
            rho_avg = 0.5 * (rho[i_idx] + rho[j_idx])
            rho_diff = torch.abs(rho[i_idx] - rho[j_idx])
            rho_avg_scaled = rho_avg / rho_scale
            rho_diff_scaled = rho_diff / rho_scale
            phi = self._phi_net(
                dist=dist, rho_avg_scaled=rho_avg_scaled, rho_diff_scaled=rho_diff_scaled
            )
        else:
            rho = None
            rho_scale = None
            phi = self._phi_net(dist=dist)  # accel magnitude
        delta = (w * phi).unsqueeze(-1) * r_hat  # (P,dim)

        a = torch.zeros_like(x)
        a = a.index_add(0, i_idx, +delta)
        a = a.index_add(0, j_idx, -delta)
        if not collect_debug:
            return a
        return a, {"phi": w * phi, "dist": dist, "rho": rho, "rho_scale": rho_scale}

    def _dissipative_acc(
        self,
        *,
        x: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        collect_debug: bool = False,
        pair_geom: tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None = None,
        vel_hist_mean: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        geom = pair_geom if pair_geom is not None else self._pair_geometry(x, edge_index)
        if geom is None:
            zero = torch.zeros_like(v)
            return (zero, {"alpha": None, "dist": None}) if collect_debug else zero

        pairs, rel, dist, r_hat, _ = geom
        i_idx, j_idx = pairs

        v_rel = v[j_idx] - v[i_idx]
        s = (v_rel * r_hat).sum(dim=-1)  # radial relative speed

        # scaled clipping for stability
        s_scaled = s / max(self._vel_scale, 1e-12)
        s_scaled = torch.clamp(
            s_scaled, -self._diss_cfg.s_clip_scaled, +self._diss_cfg.s_clip_scaled
        )
        s_used = s_scaled * self._vel_scale

        # velocity 履歴（5 ステップ）に基づきダンピング係数を緩やかにスケール
        # gate は [0.5, 1.0) に収まり、低速時の過剰減衰を抑える
        if vel_hist_mean is not None:
            hist = 0.5 * (vel_hist_mean[i_idx] + vel_hist_mean[j_idx])
            hist_scaled = torch.tanh(hist / (self._vel_scale + 1e-12))
            gate = 0.5 + 0.5 * hist_scaled  # 0.5〜1.0
        else:
            gate = 1.0

        alpha = self._alpha_net(dist=dist, s_abs=torch.abs(s_used)) * gate  # >=0
        w = self._cutoff_weight(dist)
        delta = (w * (alpha / max(self._dt, 1e-8)) * s_used).unsqueeze(-1) * r_hat

        a = torch.zeros_like(v)
        a = a.index_add(0, i_idx, +delta)
        a = a.index_add(0, j_idx, -delta)
        if not collect_debug:
            return a
        return a, {"alpha": alpha, "dist": dist, "damp_gate": gate}

    def _pressure_acc(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        collect_debug: bool = False,
        pair_geom: tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float | None]]:
        if self._pressure_net is None:
            zero = torch.zeros_like(x)
            return (
                (zero, {"rep": None, "rho_rel": None, "rho_scale": None, "comp_gate_mean": None})
                if collect_debug
                else zero
            )

        geom = pair_geom if pair_geom is not None else self._pair_geometry(x, edge_index)
        if geom is None:
            zero = torch.zeros_like(x)
            return (
                (zero, {"rep": None, "rho_rel": None, "rho_scale": None, "comp_gate_mean": None})
                if collect_debug
                else zero
            )

        pairs, _, dist, r_hat, w = geom
        i_idx, j_idx = pairs

        rho = self._proxy_density(pairs=pairs, w=w, num_nodes=x.shape[0])
        rho_mean = rho.detach().mean()
        decay = 0.99
        if self.training:
            self._rho_ema.mul_(decay).add_((1.0 - decay) * rho_mean)
        rho_scale = torch.clamp(self._rho_ema.to(rho.device, rho.dtype), min=1e-6)

        rho_rel = rho / rho_scale - 1.0
        rho_rel_avg = 0.5 * (rho_rel[i_idx] + rho_rel[j_idx])
        rho_rel_diff = torch.abs(rho_rel[i_idx] - rho_rel[j_idx])

        rep = self._pressure_net(dist=dist, rho_rel_avg=rho_rel_avg, rho_rel_diff=rho_rel_diff)

        comp_gate_mean: float | None = None
        if self._pressure_cfg.use_compression_gate:
            comp_gate = torch.relu(rho_rel_avg)
            rep = rep * comp_gate
            comp_gate_mean = (
                float(comp_gate.detach().mean().item()) if comp_gate.numel() > 0 else None
            )

        delta = (w * rep).unsqueeze(-1) * r_hat
        a = torch.zeros_like(x)
        a = a.index_add(0, i_idx, -delta)
        a = a.index_add(0, j_idx, +delta)

        if not collect_debug:
            return a
        return a, {
            "rep": w * rep,
            "rho_rel": rho_rel,
            "rho_scale": rho_scale,
            "comp_gate_mean": comp_gate_mean,
        }

    def _contact_acc(
        self,
        *,
        x: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        collect_debug: bool = False,
        pair_geom: tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float | None]]:
        if self._contact_alpha_net is None:
            zero = torch.zeros_like(v)
            return (
                (zero, {"alpha_contact": None, "s_neg": None, "active_rate": None})
                if collect_debug
                else zero
            )

        geom = pair_geom if pair_geom is not None else self._pair_geometry(x, edge_index)
        if geom is None:
            zero = torch.zeros_like(v)
            return (
                (zero, {"alpha_contact": None, "s_neg": None, "active_rate": None})
                if collect_debug
                else zero
            )

        pairs, _, dist, r_hat, w = geom
        i_idx, j_idx = pairs

        v_rel = v[j_idx] - v[i_idx]
        s = (v_rel * r_hat).sum(dim=-1)
        s_neg = torch.relu(-s)

        s_scaled = s_neg / max(self._vel_scale, 1e-12)
        s_scaled = torch.clamp(s_scaled, 0.0, self._contact_cfg.s_clip_scaled)
        s_used = s_scaled * self._vel_scale

        alpha = self._contact_alpha_net(dist=dist, s_neg=s_used)
        mag = w * (alpha / max(self._dt, 1e-8)) * s_used
        delta = mag.unsqueeze(-1) * r_hat
        a = torch.zeros_like(v)
        a = a.index_add(0, i_idx, -delta)
        a = a.index_add(0, j_idx, +delta)

        if not collect_debug:
            return a

        active_rate = (
            float((s_neg.detach() > 0).float().mean().item()) if s_neg.numel() > 0 else None
        )
        return a, {"alpha_contact": alpha, "s_neg": s_used, "active_rate": active_rate}

    # ------------------------------------------------------------
    # Dynamics
    def _dynamics(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        *,
        particle_types: torch.Tensor | None = None,
        collect_debug: bool | None = None,
        vel_hist_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = self._build_edges(x, nparticles_per_example)
        pair_geom = self._pair_geometry(x, edge_index)

        do_collect = self._record_debug_stats if collect_debug is None else collect_debug
        tcfg = self._terms_cfg

        if tcfg.enable_central and (tcfg.w_central != 0.0):
            cons_out = self._conservative_acc(
                x=x, edge_index=edge_index, collect_debug=do_collect, pair_geom=pair_geom
            )
            if do_collect:
                a_central, cons_dbg = cons_out  # type: ignore[assignment]
            else:
                a_central = cons_out  # type: ignore[assignment]
                cons_dbg = {"phi": None, "dist": None, "rho": None, "rho_scale": None}
            a_central_w = a_central * float(tcfg.w_central)
        else:
            a_central_w = torch.zeros_like(x)
            cons_dbg = {"phi": None, "dist": None, "rho": None, "rho_scale": None}

        press_dbg: dict[str, torch.Tensor | float | None] = {
            "rep": None,
            "rho_rel": None,
            "rho_scale": None,
            "comp_gate_mean": None,
        }
        if tcfg.enable_pressure and (tcfg.w_pressure != 0.0):
            pres_out = self._pressure_acc(
                x=x, edge_index=edge_index, collect_debug=do_collect, pair_geom=pair_geom
            )
            if do_collect:
                a_pressure, press_dbg = pres_out  # type: ignore[assignment]
            else:
                a_pressure = pres_out  # type: ignore[assignment]
            a_pressure_w = a_pressure * float(tcfg.w_pressure)
        else:
            a_pressure_w = torch.zeros_like(x)

        contact_dbg: dict[str, torch.Tensor | float | None] = {
            "alpha_contact": None,
            "s_neg": None,
            "active_rate": None,
        }
        if tcfg.enable_contact and (tcfg.w_contact != 0.0):
            con_out = self._contact_acc(
                x=x, v=v, edge_index=edge_index, collect_debug=do_collect, pair_geom=pair_geom
            )
            if do_collect:
                a_contact, contact_dbg = con_out  # type: ignore[assignment]
            else:
                a_contact = con_out  # type: ignore[assignment]
            a_contact_w = a_contact * float(tcfg.w_contact)
        else:
            a_contact_w = torch.zeros_like(v)

        if tcfg.enable_damping and (tcfg.w_damping != 0.0):
            diss_out = self._dissipative_acc(
                x=x,
                v=v,
                edge_index=edge_index,
                collect_debug=do_collect,
                pair_geom=pair_geom,
                vel_hist_mean=vel_hist_mean,
            )
            if do_collect:
                a_damp, diss_dbg = diss_out  # type: ignore[assignment]
            else:
                a_damp = diss_out  # type: ignore[assignment]
                diss_dbg = {"alpha": None, "dist": None}
            a_damp_w = a_damp * float(tcfg.w_damping)
        else:
            a_damp_w = torch.zeros_like(v)
            diss_dbg = {"alpha": None, "dist": None}

        a_wall = self._wall_acc(x, v)

        a_pair = a_central_w + a_pressure_w + a_contact_w + a_damp_w
        a = a_pair + a_wall

        if self._include_external_acceleration and self._gravity is not None:
            a = a + self._gravity.to(device=x.device, dtype=x.dtype)

        wall_mask = self._wall_particle_mask(particle_types, x.shape[0], x.device)
        if wall_mask is not None and self._wall_particles_cfg.zero_wall_acceleration:
            a = torch.where(wall_mask[:, None], torch.zeros_like(a), a)

        if do_collect:
            self._last_debug_stats = self._compute_debug_stats(
                a_pair=a_pair,
                a_wall=a_wall,
                a_central=a_central_w,
                a_pressure=a_pressure_w,
                a_contact=a_contact_w,
                a_damping=a_damp_w,
                cons_dbg=cons_dbg,
                press_dbg=press_dbg,
                contact_dbg=contact_dbg,
                diss_dbg=diss_dbg,
                rho=cons_dbg.get("rho"),
                x=x,
                pair_geom=pair_geom,
                nparticles_per_example=nparticles_per_example,
                a_total=a,
            )
        else:
            self._last_debug_stats = None

        return v, a  # xdot, vdot

    def _compute_debug_stats(
        self,
        *,
        a_pair: torch.Tensor,
        a_wall: torch.Tensor,
        a_central: torch.Tensor | None,
        a_pressure: torch.Tensor | None,
        a_contact: torch.Tensor | None,
        a_damping: torch.Tensor | None,
        cons_dbg: dict[str, torch.Tensor | None],
        press_dbg: dict[str, torch.Tensor | float | None],
        contact_dbg: dict[str, torch.Tensor | float | None],
        diss_dbg: dict[str, torch.Tensor | None],
        rho: torch.Tensor | None,
        x: torch.Tensor | None,
        pair_geom: tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None,
        nparticles_per_example: torch.Tensor,
        a_total: torch.Tensor,
    ) -> dict[str, float | None]:
        def _mean_and_max(t: torch.Tensor | None) -> tuple[float | None, float | None]:
            if t is None or t.numel() == 0:
                return None, None
            t_det = t.detach()
            return float(t_det.mean().item()), float(t_det.max().item())

        def _vec_mag_stats(vec: torch.Tensor | None) -> tuple[float | None, float | None]:
            if vec is None or vec.numel() == 0:
                return None, None
            mag = torch.linalg.norm(vec.detach(), dim=-1)
            return _mean_and_max(mag)

        stats: dict[str, float | None] = {}

        stats["a_pair_abs_mean"], stats["a_pair_abs_max"] = _vec_mag_stats(a_pair)
        stats["a_wall_abs_mean"], stats["a_wall_abs_max"] = _vec_mag_stats(a_wall)

        stats["a_central_abs_mean"], stats["a_central_abs_max"] = _vec_mag_stats(a_central)
        stats["a_pressure_abs_mean"], stats["a_pressure_abs_max"] = _vec_mag_stats(a_pressure)
        stats["a_contact_abs_mean"], stats["a_contact_abs_max"] = _vec_mag_stats(a_contact)
        stats["a_damping_abs_mean"], stats["a_damping_abs_max"] = _vec_mag_stats(a_damping)

        phi_vals = cons_dbg.get("phi") if cons_dbg is not None else None
        phi_abs = phi_vals.detach().abs() if phi_vals is not None else None
        stats["phi_mean"], stats["phi_max"] = _mean_and_max(phi_abs)

        rep_vals = press_dbg.get("rep")
        rep_abs = rep_vals.detach().abs() if isinstance(rep_vals, torch.Tensor) else None
        stats["rep_mean"], stats["rep_max"] = _mean_and_max(rep_abs)
        stats["pressure_comp_gate_mean"] = (
            float(press_dbg["comp_gate_mean"])
            if press_dbg.get("comp_gate_mean") is not None
            else None
        )

        alpha_c = contact_dbg.get("alpha_contact")
        alpha_c_t = alpha_c.detach() if isinstance(alpha_c, torch.Tensor) else None
        stats["alpha_contact_mean"], stats["alpha_contact_max"] = _mean_and_max(alpha_c_t)
        stats["contact_active_rate"] = (
            float(contact_dbg["active_rate"])
            if contact_dbg.get("active_rate") is not None
            else None
        )

        alpha_vals = diss_dbg.get("alpha") if diss_dbg is not None else None
        alpha_t = alpha_vals.detach() if alpha_vals is not None else None
        stats["alpha_mean"], stats["alpha_max"] = _mean_and_max(alpha_t)

        if pair_geom is not None:
            dist = pair_geom[2].detach()
            w = pair_geom[4].detach()
            stats["pair_count"] = float(dist.numel())
            stats["dist_mean"], stats["dist_max"] = _mean_and_max(dist)
            stats["w_mean"], stats["w_max"] = _mean_and_max(w)
            stats["w_min"] = float(w.min().item()) if w.numel() > 0 else None
        else:
            stats["pair_count"] = 0.0
            stats["dist_mean"] = None
            stats["dist_max"] = None
            stats["w_mean"] = None
            stats["w_max"] = None
            stats["w_min"] = None

        if rho is not None and rho.numel() > 0 and x is not None:
            rho_det = rho.detach()
            stats["rho_mean"], stats["rho_max"] = _mean_and_max(rho_det)
            stats["rho_std"] = float(rho_det.std().item())
            stats["rho_min"] = float(rho_det.min().item())

            if self._boundary_mode != BOUNDARY_MODE_PERIODIC:
                boundaries = self._boundaries.to(device=x.device, dtype=x.dtype)
                lower = boundaries[:, 0]
                upper = boundaries[:, 1]
                dist_lower = x - lower
                dist_upper = upper - x
                min_dist = torch.minimum(dist_lower, dist_upper).amin(dim=-1)
                threshold = float(self._sph.smoothing_length)
                near_mask = (min_dist < threshold).bool()
                away_mask = ~near_mask

                def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> float | None:
                    if not bool(mask.any()):
                        return None
                    return float(t[mask].mean().item())

                stats["rho_near_mean"] = _masked_mean(rho_det, near_mask)
                stats["rho_away_mean"] = _masked_mean(rho_det, away_mask)
            else:
                stats["rho_near_mean"] = None
                stats["rho_away_mean"] = None
        else:
            stats["rho_mean"] = None
            stats["rho_max"] = None
            stats["rho_std"] = None
            stats["rho_min"] = None
            stats["rho_near_mean"] = None
            stats["rho_away_mean"] = None

        if pair_geom is not None and x is not None and x.numel() > 0:
            i_idx, j_idx = pair_geom[0]
            num_nodes = x.shape[0]
            deg = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
            ones = torch.ones_like(i_idx, dtype=torch.float32, device=x.device)
            deg = deg.index_add(0, i_idx, ones)
            deg = deg.index_add(0, j_idx, ones)
            stats["neighbor_edges_mean"] = float(deg.mean().item()) if deg.numel() > 0 else None
            stats["neighbor_edges_max"] = float(deg.max().item()) if deg.numel() > 0 else None
        else:
            stats["neighbor_edges_mean"] = None
            stats["neighbor_edges_max"] = None

        if pair_geom is not None:
            dist = pair_geom[2].detach()
            stats["min_d_edges"] = float(dist.min().item()) if dist.numel() > 0 else None
        else:
            stats["min_d_edges"] = None

        if x is not None and nparticles_per_example is not None and x.numel() > 0:
            stats["min_d"] = self._compute_min_distance_all(
                x.detach(), nparticles_per_example.detach()
            )
        else:
            stats["min_d"] = None

        if a_total is not None and a_total.numel() > 0:
            a_mag = torch.linalg.norm(a_total.detach(), dim=-1)
            if a_mag.numel() > 0:
                a_mag_f = a_mag.float()
                stats["a_pred_abs_mean"] = float(a_mag_f.mean().item())
                stats["a_pred_abs_p95"] = float(torch.quantile(a_mag_f, 0.95).item())
            else:
                stats["a_pred_abs_mean"] = None
                stats["a_pred_abs_p95"] = None
        else:
            stats["a_pred_abs_mean"] = None
            stats["a_pred_abs_p95"] = None

        return stats

    def _compute_min_distance_all(
        self, x: torch.Tensor, nparticles_per_example: torch.Tensor
    ) -> float | None:
        with torch.no_grad():
            counts = nparticles_per_example.to(device=x.device, dtype=torch.long)
            mins: list[torch.Tensor] = []
            start = 0
            for n in counts.tolist():
                n_int = int(n)
                if n_int <= 1:
                    start += n_int
                    continue
                slice_pos = x[start : start + n_int]
                if slice_pos.numel() == 0:
                    start += n_int
                    continue
                dist = torch.cdist(slice_pos, slice_pos)
                if dist.numel() == 0:
                    start += n_int
                    continue
                dist.fill_diagonal_(torch.inf)
                mins.append(dist.min())
                start += n_int

            if not mins:
                return None
            return float(torch.stack(mins).min().item())

    def get_last_debug_stats(self) -> dict[str, float | None] | None:
        return self._last_debug_stats

    # ------------------------------------------------------------
    # Integrator (rollout only)
    def _integrate(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor | None = None,
        *,
        vel_hist_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self._dt
        wall_mask = self._wall_particle_mask(particle_types, x.shape[0], x.device)
        if wall_mask is not None and self._wall_particles_cfg.freeze_walls:
            v = torch.where(wall_mask[:, None], torch.zeros_like(v), v)

        if self._int_cfg.type == "symplectic_euler":
            _, vdot = self._dynamics(
                x,
                v,
                nparticles_per_example,
                particle_types=particle_types,
                vel_hist_mean=vel_hist_mean,
            )
            v_new = v + dt * vdot
            x_new = x + dt * v_new
            if self._boundary_mode == BOUNDARY_MODE_PERIODIC or self._rollout_reflect_walls:
                x_new, v_new = self._apply_boundary_conditions(x_new, v_new, mask_exclude=wall_mask)
            if wall_mask is not None and self._wall_particles_cfg.freeze_walls:
                x_new = torch.where(wall_mask[:, None], x, x_new)
                v_new = torch.where(wall_mask[:, None], torch.zeros_like(v_new), v_new)
            return x_new, v_new

        # leapfrog
        _, vdot = self._dynamics(
            x,
            v,
            nparticles_per_example,
            particle_types=particle_types,
            vel_hist_mean=vel_hist_mean,
        )
        v_half = v + 0.5 * dt * vdot
        x_new = x + dt * v_half
        _, vdot2 = self._dynamics(
            x_new,
            v_half,
            nparticles_per_example,
            particle_types=particle_types,
            vel_hist_mean=vel_hist_mean,
        )
        v_new = v_half + 0.5 * dt * vdot2
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC or self._rollout_reflect_walls:
            x_new, v_new = self._apply_boundary_conditions(x_new, v_new, mask_exclude=wall_mask)
        if wall_mask is not None and self._wall_particles_cfg.freeze_walls:
            x_new = torch.where(wall_mask[:, None], x, x_new)
            v_new = torch.where(wall_mask[:, None], torch.zeros_like(v_new), v_new)
        return x_new, v_new

    # ------------------------------------------------------------
    # Public API (GNS-compatible)
    def predict_positions(
        self,
        current_positions: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del material_property

        dt_safe = max(self._dt, 1e-8)
        x = current_positions[:, -1].to(self._device)
        prev = current_positions[:, -2].to(self._device)
        v = self._relative_displacement(x, prev) / dt_safe
        # 直近 5 ステップの速度ノルム平均を計算（ダンピング用ゲート）
        with torch.no_grad():
            vel_hist = self._relative_displacement(
                current_positions[:, 1:], current_positions[:, :-1]
            ).to(self._device)
            vel_hist_mean = torch.linalg.norm(vel_hist, dim=-1).mean(dim=1) / dt_safe

        nparticles_per_example = nparticles_per_example.to(self._device)
        particle_types = particle_types.to(self._device)

        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                x_new, _ = self._integrate(
                    x,
                    v,
                    nparticles_per_example,
                    particle_types=particle_types,
                    vel_hist_mean=vel_hist_mean,
                )
        finally:
            if was_training:
                self.train()

        return x_new.detach()

    def predict_accelerations(
        self,
        next_positions: torch.Tensor,
        position_sequence_noise: torch.Tensor,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None,
    ):
        del material_property

        particle_types = particle_types.to(self._device)

        wall_mask = self._wall_particle_mask(
            particle_types, position_sequence.shape[0], self._device
        )
        if wall_mask is not None:
            position_sequence_noise = position_sequence_noise.clone()
            position_sequence_noise[wall_mask] = 0.0

        # --- Input world (noisy) ---
        noisy_seq = position_sequence + position_sequence_noise
        boundaries = self._boundaries.to(device=self._device, dtype=noisy_seq.dtype)
        lower, upper = boundaries[:, 0], boundaries[:, 1]

        x_raw = noisy_seq[:, -1].to(self._device)
        prev_raw = noisy_seq[:, -2].to(self._device)
        next_raw = (next_positions + position_sequence_noise[:, -1]).to(self._device)

        x = self._clamp_positions(x_raw, lower, upper, wall_mask)
        prev = self._clamp_positions(prev_raw, lower, upper, wall_mask)
        next_noisy = self._clamp_positions(next_raw, lower, upper, wall_mask)

        dt_safe = max(self._dt, 1e-8)
        v = self._relative_displacement(x, prev) / dt_safe
        vel_hist = self._relative_displacement(noisy_seq[:, 1:], noisy_seq[:, :-1]).to(self._device)
        vel_hist_mean = torch.linalg.norm(vel_hist, dim=-1).mean(dim=1) / dt_safe

        nparticles_per_example = nparticles_per_example.to(self._device)

        # --- Model predicts instantaneous a(x_t, v_t) ---
        _, vdot = self._dynamics(
            x,
            v,
            nparticles_per_example,
            particle_types=particle_types,
            vel_hist_mean=vel_hist_mean,
        )  # (N,dim)
        acc = vdot * (self._dt * self._dt)  # (N,dim) == Δv

        # normalize
        acc_stats = self._normalization_stats["acceleration"]
        acc_mean = torch.as_tensor(acc_stats["mean"], device=acc.device, dtype=acc.dtype)
        acc_std = torch.as_tensor(acc_stats["std"], device=acc.device, dtype=acc.dtype)
        normalized_acc = (acc - acc_mean) / acc_std

        # --- Target in the SAME noisy world ---
        with torch.no_grad():
            v_next = self._relative_displacement(next_noisy, x)
            v_prev = self._relative_displacement(x, prev)
            target_acc = v_next - v_prev  # == a*dt^2
            target_normalized = (target_acc - acc_mean) / acc_std

        return normalized_acc, target_normalized

    def save(self, path: str = "model.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


# 後方互換用のエイリアス（既存コードは HamiltonianSPHSimulator 名を期待）
HamiltonianSPHSimulator = HamiltonianSPHVarAWithDissipation

__all__ = [
    "ConservativeConfig",
    "ContactConfig",
    "CutoffConfig",
    "DissipationConfig",
    "HamiltonianSPHSimulator",
    "HamiltonianSPHVarAWithDissipation",
    "IntegratorConfig",
    "PairConservativePhiNetDistOnly",
    "PairConservativePhiNetWithDensity",
    "PairContactAlphaNet",
    "PairDissipationAlphaNet",
    "PairPressureRepulsionNet",
    "PressureConfig",
    "SPHConfig",
    "TermConfig",
    "WallConfig",
    "WallMagnitudeNet",
    "WallParticleConfig",
]
