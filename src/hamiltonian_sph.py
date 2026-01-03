"""Central Conservative + Folded Dissipation (rotation-safe v1 baseline)

Goal: stop option explosion and give a stable, learnable default.

Key decisions (v1):
  1) Use undirected unique pairs (never drop interactions due to edge direction).
  2) Smooth cutoff weight w(d) (cosine) to avoid force discontinuities at cutoff.
  3) Conservative term learns center-force magnitude phi(d) directly (dist-only, rotation-safe),
     instead of U->grad(x). This avoids differentiating wrt positions and is much lighter/stabler.
  4) Dissipation uses folded radial damping:
        s = v_rel · r_hat
        alpha(d, |s|) in [0, alpha_max]
        delta = w(d) * (alpha/dt) * s_clipped * r_hat
     Apply action-reaction:
        a_i += delta
        a_j -= delta
     This is guaranteed energy-dissipative for radial motion (alpha>=0).
  5) Training target consistency: compute target Δv (= a*dt^2) in the SAME "noisy world"
     used for model input.

Notes:
  - This keeps GNS-compatible BaseSimulator interface: predict_positions / predict_accelerations.
  - Leapfrog is used for rollout; training uses instantaneous a(x_t,v_t) (no integrator mismatch).

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import radius_graph

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
    phi_max_multiplier: float = 5.0  # phi_max = multiplier * (pos_scale / dt^2)


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
class CutoffConfig:
    """Smooth cutoff weight w(d) with cosine ramp:
    w(d)=0.5*(cos(pi*d/rc)+1) for d<rc else 0
    """

    kind: Literal["cosine"] = "cosine"


@dataclass
class IntegratorConfig:
    dt: float = 1e-3
    type: Literal["leapfrog", "symplectic_euler"] = "leapfrog"


# ------------------------------------------------------------
# Networks
class PairConservativePhiNetDistOnly(nn.Module):
    """phi = f(dist) -> scalar (acceleration magnitude along r_hat)"""

    def __init__(self, cfg: ConservativeConfig, *, pos_scale: float, dt: float) -> None:
        super().__init__()
        self._pos_scale = float(pos_scale)
        self._dt = float(dt)
        self._drop = nn.Dropout(cfg.dropout)

        # acceleration scale ~ pos / dt^2
        self._phi_max = float(cfg.phi_max_multiplier) * (
            self._pos_scale / max(self._dt * self._dt, 1e-12)
        )

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
        return self._phi_max * torch.tanh(raw)  # bounded, units: acceleration


class PairDissipationAlphaNet(nn.Module):
    """alpha = g(d, |s|) in [0, alpha_max]
      d = ||r||
      s = v_rel · r_hat

    We feed |s| to enforce even symmetry (more physical, avoids odd artifacts).
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
        raw = self._mlp(x).squeeze(-1)  # (P,)
        return self._alpha_max * torch.sigmoid(raw)  # (P,) in [0, alpha_max]


# ------------------------------------------------------------
# Simulator
class HamiltonianSPHVarAWithDissipation(BaseSimulator):
    """v1 baseline:
    - conservative: center-force magnitude phi(dist)
    - dissipation: folded radial damping with bounded alpha(d,|s|)
    - undirected unique pairs
    - smooth cutoff
    - training uses instantaneous acceleration a(x_t,v_t)
    """

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
        device: torch.device | str = "cpu",
        *,
        sph: SPHConfig | None = None,
        conservative: ConservativeConfig | None = None,
        dissipation: DissipationConfig | None = None,
        cutoff: CutoffConfig | None = None,
        integrator: IntegratorConfig | None = None,
        particle_mass: float = 1.0,
        gravity: torch.Tensor | None = None,
        pos_feature_scale: float | None = None,
        vel_feature_scale: float | None = None,
        include_external_acceleration: bool = True,
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

        self._dim = int(particle_dimensions)
        self._device = torch.device(device)
        self._connectivity_radius = float(connectivity_radius)
        self._normalization_stats = normalization_stats
        self._particle_mass = float(particle_mass)

        boundaries_arr = torch.as_tensor(boundaries, dtype=torch.float32)
        if boundaries_arr.ndim != 2 or boundaries_arr.shape != (self._dim, 2):
            raise ValueError(
                f"Expected boundaries shape ({self._dim},2); got {tuple(boundaries_arr.shape)}"
            )
        self._boundaries = boundaries_arr
        self._boundary_restitution = 0.5

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

        # Scales for stable feature scaling
        self._pos_scale = float(pos_feature_scale or (2.0 * float(self._sph.smoothing_length)))
        self._vel_scale = float(vel_feature_scale or (self._pos_scale / max(self._dt, 1e-8)))

        # Nets
        self._phi_net = PairConservativePhiNetDistOnly(
            cfg=self._cons_cfg,
            pos_scale=self._pos_scale,
            dt=self._dt,
        ).to(self._device)

        self._alpha_net = PairDissipationAlphaNet(
            cfg=self._diss_cfg,
            pos_scale=self._pos_scale,
            vel_scale=self._vel_scale,
        ).to(self._device)

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

    # ------------------------------------------------------------
    # Graph construction
    def _compute_graph_connectivity(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        *,
        add_self_edges: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = nparticles_per_example.to(node_position.device, dtype=torch.long)
        batch_ids = torch.repeat_interleave(
            torch.arange(len(counts), device=node_position.device), counts
        )
        try:
            edge_index = radius_graph(
                node_position,
                r=radius,
                batch=batch_ids,
                loop=add_self_edges,
                max_num_neighbors=self._sph.max_num_neighbors,
            )
            receivers = edge_index[0, :]
            senders = edge_index[1, :]
        except (ImportError, RuntimeError):
            receivers, senders = self._dense_radius_graph(
                node_position, nparticles_per_example, radius, add_self_edges
            )
        return receivers, senders

    def _dense_radius_graph(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        add_self_edges: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = node_position.device
        receivers_list: list[torch.Tensor] = []
        senders_list: list[torch.Tensor] = []
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
                receivers_list.append(r_idx + start_idx)
                senders_list.append(s_idx + start_idx)
            start_idx += n

        if receivers_list:
            receivers = torch.cat(receivers_list).to(device)
            senders = torch.cat(senders_list).to(device)
        else:
            receivers = torch.empty(0, dtype=torch.long, device=device)
            senders = torch.empty(0, dtype=torch.long, device=device)
        return receivers, senders

    def _build_edges(self, x: torch.Tensor, nparticles_per_example: torch.Tensor) -> torch.Tensor:
        radius = 2.0 * float(self._sph.smoothing_length)  # cutoff radius rc
        receivers, senders = self._compute_graph_connectivity(
            x, nparticles_per_example, radius, add_self_edges=False
        )
        # directed edges as (senders -> receivers)
        edge_index = torch.stack([senders, receivers], dim=0)  # (2,E)
        return edge_index

    def _build_pairs(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Directed edges -> unique undirected pairs.
        This prevents "dropping" interactions due to edge direction / neighbor truncation.
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

        # unique undirected pairs
        pairs = torch.unique(pairs, dim=0)
        return pairs[:, 0], pairs[:, 1]

    # ------------------------------------------------------------
    # Smooth cutoff
    def _cutoff_weight(self, dist: torch.Tensor) -> torch.Tensor:
        rc = 2.0 * float(self._sph.smoothing_length)
        rc_t = torch.as_tensor(rc, device=dist.device, dtype=dist.dtype)
        # cosine ramp: w=0..1, smooth at endpoints
        x = dist / (rc_t + 1e-12)
        w = 0.5 * (torch.cos(math.pi * x) + 1.0)
        w = torch.where(dist < rc_t, w, torch.zeros_like(w))
        return w

    # ------------------------------------------------------------
    # Boundary conditions (simple damped reflection) for rollout
    def _apply_boundary_conditions(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundaries = self._boundaries.to(device=x.device, dtype=x.dtype)
        lower = boundaries[:, 0]
        upper = boundaries[:, 1]

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
        return x_ref, v_ref

    # ------------------------------------------------------------
    # Conservative + dissipative accelerations (instantaneous)
    def _conservative_acc(self, *, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pairs = self._build_pairs(edge_index)
        if pairs is None:
            return torch.zeros_like(x)

        i_idx, j_idx = pairs
        rel = x[j_idx] - x[i_idx]  # (P,dim)
        dist = torch.sqrt((rel * rel).sum(dim=-1) + 1e-12)  # (P,)
        r_hat = rel / (dist.unsqueeze(-1) + 1e-8)  # (P,dim)

        w = self._cutoff_weight(dist)  # (P,)
        phi = self._phi_net(dist=dist)  # (P,) accel magnitude
        delta = (w * phi).unsqueeze(-1) * r_hat  # (P,dim)

        a = torch.zeros_like(x)
        a = a.index_add(0, i_idx, +delta)
        a = a.index_add(0, j_idx, -delta)
        return a

    def _dissipative_acc(
        self, *, x: torch.Tensor, v: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        pairs = self._build_pairs(edge_index)
        if pairs is None:
            return torch.zeros_like(v)

        i_idx, j_idx = pairs
        rel = x[j_idx] - x[i_idx]  # (P,dim)
        dist = torch.sqrt((rel * rel).sum(dim=-1) + 1e-12)  # (P,)
        r_hat = rel / (dist.unsqueeze(-1) + 1e-8)  # (P,dim)

        v_rel = v[j_idx] - v[i_idx]  # (P,dim)
        s = (v_rel * r_hat).sum(dim=-1)  # (P,) radial relative speed

        # scaled clipping for stability
        s_scaled = s / max(self._vel_scale, 1e-12)
        s_scaled = torch.clamp(
            s_scaled, -self._diss_cfg.s_clip_scaled, +self._diss_cfg.s_clip_scaled
        )
        s_used = s_scaled * self._vel_scale  # (P,) back to physical units

        # even symmetry in s: feed |s|
        alpha = self._alpha_net(dist=dist, s_abs=torch.abs(s_used))  # (P,) in [0, alpha_max]

        w = self._cutoff_weight(dist)  # (P,)
        delta = (w * (alpha / max(self._dt, 1e-8)) * s_used).unsqueeze(-1) * r_hat  # (P,dim)

        a = torch.zeros_like(v)
        a = a.index_add(0, i_idx, +delta)
        a = a.index_add(0, j_idx, -delta)
        return a

    def _dynamics(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        nparticles_per_example: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = self._build_edges(x, nparticles_per_example)
        a_cons = self._conservative_acc(x=x, edge_index=edge_index)
        a_diss = self._dissipative_acc(x=x, v=v, edge_index=edge_index)

        a = a_cons + a_diss
        if self._include_external_acceleration and self._gravity is not None:
            a = a + self._gravity.to(device=x.device, dtype=x.dtype)

        xdot = v
        vdot = a
        return xdot, vdot

    # ------------------------------------------------------------
    # Integrator (rollout only)
    def _integrate(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        nparticles_per_example: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self._dt

        if self._int_cfg.type == "symplectic_euler":
            _, vdot = self._dynamics(x, v, nparticles_per_example)
            v_new = v + dt * vdot
            x_new = x + dt * v_new
            x_new, v_new = self._apply_boundary_conditions(x_new, v_new)
            return x_new, v_new

        # leapfrog
        _, vdot = self._dynamics(x, v, nparticles_per_example)
        v_half = v + 0.5 * dt * vdot
        x_new = x + dt * v_half
        _, vdot2 = self._dynamics(x_new, v_half, nparticles_per_example)
        v_new = v_half + 0.5 * dt * vdot2
        x_new, v_new = self._apply_boundary_conditions(x_new, v_new)
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
        del particle_types, material_property

        x = current_positions[:, -1].to(self._device)
        prev = current_positions[:, -2].to(self._device)
        v = (x - prev) / max(self._dt, 1e-8)

        nparticles_per_example = nparticles_per_example.to(self._device)

        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                x_new, _ = self._integrate(x, v, nparticles_per_example)
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
        del particle_types, material_property

        # --- Input world (noisy) ---
        noisy_seq = position_sequence + position_sequence_noise
        x = noisy_seq[:, -1].to(self._device)
        prev = noisy_seq[:, -2].to(self._device)
        v = (x - prev) / max(self._dt, 1e-8)

        nparticles_per_example = nparticles_per_example.to(self._device)

        # --- Model predicts instantaneous a(x_t, v_t) ---
        _, vdot = self._dynamics(x, v, nparticles_per_example)  # (N,dim)

        # GNS expects a*dt^2 (== Δv) to compare with second-difference target
        acc = vdot * (self._dt * self._dt)  # (N,dim)

        # normalize
        acc_stats = self._normalization_stats["acceleration"]
        acc_mean = torch.as_tensor(acc_stats["mean"], device=acc.device, dtype=acc.dtype)
        acc_std = torch.as_tensor(acc_stats["std"], device=acc.device, dtype=acc.dtype)
        normalized_acc = (acc - acc_mean) / acc_std

        # --- Target in the SAME noisy world ---
        with torch.no_grad():
            next_noisy = (next_positions + position_sequence_noise[:, -1]).to(self._device)
            v_next = next_noisy - noisy_seq[:, -1].to(self._device)
            v_prev = noisy_seq[:, -1].to(self._device) - noisy_seq[:, -2].to(self._device)
            target_acc = v_next - v_prev  # == a*dt^2 (second difference)
            target_normalized = (target_acc - acc_mean) / acc_std

        return normalized_acc, target_normalized

    def save(self, path: str = "model.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


# 互換性のためのエイリアスを公開しておく
HamiltonianSPHSimulator = HamiltonianSPHVarAWithDissipation

__all__ = [
    "HamiltonianSPHSimulator",
    "HamiltonianSPHVarAWithDissipation",
    "SPHConfig",
    "ConservativeConfig",
    "DissipationConfig",
    "CutoffConfig",
    "IntegratorConfig",
]
