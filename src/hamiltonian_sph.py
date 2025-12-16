"""Hamiltonian-SPH ベースの新シミュレータ実装。

既存 GNS と同じ BaseSimulator インタフェースで、
ハミルトニアンのポテンシャル項を GNN で可変に学習できるようにする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torch_geometric.nn import radius_graph

import graph_network
from learned_simulator import BaseSimulator


def _cubic_spline_kernel(dist: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    """Cubic spline カーネル（Wendland ではなく標準 SPH）。

    dim に合わせた正規化係数を使い、距離 dist (>=0) で値を返す。
    """

    q = dist / (h + 1e-8)
    result = torch.zeros_like(q)

    mask1 = q <= 1.0
    mask2 = (q > 1.0) & (q <= 2.0)

    result = torch.where(
        mask1,
        1 - 1.5 * q**2 + 0.75 * q**3,
        result,
    )
    result = torch.where(mask2, 0.25 * (2 - q) ** 3, result)

    if dim == 2:
        norm = 10.0 / (7.0 * math.pi * (h**2))
    elif dim == 3:
        norm = 1.0 / (math.pi * (h**3))
    else:  # fallback: 1D もしくはその他
        norm = 2.0 / (3.0 * h)
    return norm * result


def _cubic_spline_kernel_grad(
    rel: torch.Tensor, dist: torch.Tensor, h: float, dim: int
) -> torch.Tensor:
    """cubic spline の勾配 ∇W を返す。rel は x_j - x_i。"""

    q = dist / (h + 1e-8)
    grad_q = torch.zeros_like(q)
    mask1 = q <= 1.0
    mask2 = (q > 1.0) & (q <= 2.0)
    grad_q = torch.where(mask1, -3.0 * q + 2.25 * q**2, grad_q)
    grad_q = torch.where(mask2, -0.75 * (2.0 - q) ** 2, grad_q)

    if dim == 2:
        norm = 10.0 / (7.0 * math.pi * (h**2))
    elif dim == 3:
        norm = 1.0 / (math.pi * (h**3))
    else:
        norm = 2.0 / (3.0 * h)

    dW_dr = norm * grad_q / (h + 1e-8)  # dW/dr
    rel_norm = dist + 1e-8
    return dW_dr.unsqueeze(-1) * rel / rel_norm.unsqueeze(-1)


def _tait_internal_energy(
    rho: torch.Tensor,
    rho0: float,
    sound_speed: float,
    gamma: float,
) -> torch.Tensor:
    """タイト方程式から内部エネルギー密度 u を返す（単位: エネルギー/質量）。"""

    # B = (rho0 * c0^2) / gamma, u = B/(gamma-1)*[(rho/rho0)^{gamma-1}-1]
    B = (rho0 * (sound_speed**2)) / gamma
    exponent = torch.clamp(rho / rho0, min=1e-6)
    return B / (gamma - 1.0) * (torch.pow(exponent, gamma - 1.0) - 1.0)


def _index_add_zero(dst_size: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """torch_scatter 依存を避けつつ scatter_add 相当を行う。"""

    out = torch.zeros(dst_size, device=src.device, dtype=src.dtype)
    return out.index_add(0, index, src)


@dataclass
class SPHConfig:
    kernel: Literal["cubic"] = "cubic"
    smoothing_length: Optional[float] = None
    rest_density: float = 1.0
    sound_speed: float = 10.0
    gamma: float = 7.0
    max_num_neighbors: int = 128
    # Monaghan artificial viscosity 係数
    alpha_viscosity: float = 0.1
    beta_viscosity: float = 0.0
    visc_eps: float = 0.01


@dataclass
class HamiltonianNetConfig:
    variant: Literal["varA", "varB", "varC"] = "varB"
    latent_dim: int = 128
    message_passing_steps: int = 4
    mlp_layers: int = 2
    mlp_hidden_dim: int = 128
    # 速度依存項をハミルトニアンに入れると正準形から外れるためデフォルトでオフ
    use_velocity_in_H: bool = False
    use_density_in_H: bool = True
    node_dropout: float = 0.0
    edge_dropout: float = 0.0


@dataclass
class IntegratorConfig:
    dt: float = 1e-3
    type: Literal["leapfrog", "symplectic_euler"] = "leapfrog"


class HamiltonianPotentialNet(nn.Module):
    """ポテンシャル補正 ΔU を出力する小型 GNN."""

    def __init__(
        self,
        dim: int,
        nparticle_types: int,
        type_emb_dim: int,
        net_cfg: HamiltonianNetConfig,
        pos_scale: float,
        vel_scale: float,
        rho_scale: float,
    ) -> None:
        super().__init__()
        self._dim = dim
        self._pos_scale = pos_scale
        self._vel_scale = vel_scale
        self._rho_scale = rho_scale
        self._cfg = net_cfg
        self._type_emb = nn.Embedding(nparticle_types, type_emb_dim)

        node_in = dim  # 位置
        if net_cfg.use_velocity_in_H:
            node_in += dim
        if net_cfg.use_density_in_H:
            node_in += 1
        node_in += type_emb_dim

        edge_in = dim + 2  # 相対位置(dim) + 距離 + kernel 値

        self._gnn = graph_network.EncodeProcessDecode(
            nnode_in_features=node_in,
            nnode_out_features=1,  # Δu_i
            nedge_in_features=edge_in,
            latent_dim=net_cfg.latent_dim,
            nmessage_passing_steps=net_cfg.message_passing_steps,
            nmlp_layers=net_cfg.mlp_layers,
            mlp_hidden_dim=net_cfg.mlp_hidden_dim,
        )

        self._node_dropout = nn.Dropout(net_cfg.node_dropout)
        self._edge_dropout = nn.Dropout(net_cfg.edge_dropout)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        rho: torch.Tensor,
        particle_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        feats = [x / self._pos_scale]
        if self._cfg.use_velocity_in_H:
            feats.append(v / self._vel_scale)
        if self._cfg.use_density_in_H:
            feats.append(rho.unsqueeze(-1) / self._rho_scale)
        feats.append(self._type_emb(particle_type))

        node_feat = torch.cat(feats, dim=-1)
        node_feat = self._node_dropout(node_feat)
        edge_feat = self._edge_dropout(edge_features)
        return self._gnn(node_feat, edge_index, edge_feat)


class HamiltonianSPHSimulator(BaseSimulator):
    """ハミルトニアン + SPH のハイブリッドモデル。

    predict_positions / predict_accelerations のインタフェースは既存 GNS と互換。
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
        hamiltonian_net: HamiltonianNetConfig | None = None,
        integrator: IntegratorConfig | None = None,
        particle_mass: float = 1.0,
        gravity: Optional[torch.Tensor] = None,
        pos_feature_scale: Optional[float] = None,
        vel_feature_scale: Optional[float] = None,
        rho_feature_scale: Optional[float] = None,
        include_external_potential: bool = True,
    ) -> None:
        super().__init__()
        self._dim = particle_dimensions
        self._connectivity_radius = float(connectivity_radius)
        self._normalization_stats = normalization_stats
        self._device = torch.device(device)
        self._particle_mass = float(particle_mass)
        boundaries_arr = torch.as_tensor(boundaries, dtype=torch.float32)
        if boundaries_arr.ndim != 2 or boundaries_arr.shape[1] != 2 or boundaries_arr.shape[0] != self._dim:
            msg = f"Expected boundaries with shape ({self._dim}, 2); got {tuple(boundaries_arr.shape)}"
            raise ValueError(msg)
        self._boundaries = boundaries_arr
        self._boundary_clamp_limit = float(boundary_clamp_limit)
        self._boundary_restitution = 0.5  # 水面と壁の衝突を想定した減衰付き反射係数

        def _coerce(obj, cls, fallback):
            if obj is None:
                return fallback
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, dict):
                return cls(**obj)
            raise TypeError(f"Expected {cls.__name__} or dict, got {type(obj)}")

        self._sph = _coerce(sph, SPHConfig, SPHConfig())
        if self._sph.smoothing_length is None:
            # cubic spline の有効半径 2h が connectivity_radius に一致するように設定
            self._sph.smoothing_length = self._connectivity_radius * 0.5
        self._ham_cfg = _coerce(hamiltonian_net, HamiltonianNetConfig, HamiltonianNetConfig())
        self._int_cfg = _coerce(integrator, IntegratorConfig, IntegratorConfig())

        pos_scale = pos_feature_scale or self._connectivity_radius
        vel_scale = vel_feature_scale or 1.0
        rho_scale = rho_feature_scale or max(self._sph.rest_density, 1.0)

        self._potential_net = HamiltonianPotentialNet(
            dim=self._dim,
            nparticle_types=nparticle_types,
            type_emb_dim=particle_type_embedding_size,
            net_cfg=self._ham_cfg,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            rho_scale=rho_scale,
        ).to(self._device)

        if gravity is not None:
            g = torch.as_tensor(gravity, dtype=torch.float32, device=self._device)
            if g.numel() >= self._dim:
                g = g[: self._dim]
            elif g.numel() == 1:
                g = g.repeat(self._dim)
            else:
                raise ValueError("gravity の次元が粒子次元と合いません。例: 2Dなら2要素。")
            self._gravity = g
        else:
            self._gravity = None
        self._include_external_potential = include_external_potential

    # ------------------------------------------------------------------
    # グラフ構築まわり（既存 GNS と同じ半径グラフ）
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
            dist = torch.cdist(slice_pos, slice_pos)
            mask = dist <= (radius + eps)
            if not add_self_edges:
                idx = torch.arange(n, device=device)
                mask[idx, idx] = False
            r_idx, s_idx = torch.nonzero(mask, as_tuple=True)
            if r_idx.numel() > 0:
                receivers_list.append(r_idx + start_idx)
                senders_list.append(s_idx + start_idx)
            start_idx += n

        if receivers_list:
            receivers_cat = torch.cat(receivers_list).to(device)
            senders_cat = torch.cat(senders_list).to(device)
        else:
            receivers_cat = torch.empty(0, dtype=torch.long, device=device)
            senders_cat = torch.empty(0, dtype=torch.long, device=device)
        return receivers_cat, senders_cat

    def _apply_boundary_conditions(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """境界を超えた粒子を反射し、速度を反転・減衰させる。"""

        boundaries = self._boundaries.to(device=x.device, dtype=x.dtype)
        lower = boundaries[:, 0]
        upper = boundaries[:, 1]

        below = x < lower
        above = x > upper
        if not (below.any() or above.any()):
            return x, v

        x_reflected = x.clone()
        v_reflected = v.clone()

        x_reflected = torch.where(below, lower + (lower - x_reflected), x_reflected)
        v_reflected = torch.where(
            below, -v_reflected * self._boundary_restitution, v_reflected
        )

        x_reflected = torch.where(above, upper - (x_reflected - upper), x_reflected)
        v_reflected = torch.where(
            above, -v_reflected * self._boundary_restitution, v_reflected
        )

        x_reflected = torch.min(torch.max(x_reflected, lower), upper)
        return x_reflected, v_reflected

    # ------------------------------------------------------------------
    # SPH 基本項
    def _build_edges(self, x: torch.Tensor, nparticles_per_example: torch.Tensor):
        # cubic spline の支持が 2h までなので、近傍半径は 2 * smoothing_length に合わせる
        radius = 2.0 * self._sph.smoothing_length
        receivers, senders = self._compute_graph_connectivity(
            x, nparticles_per_example, radius, add_self_edges=False
        )
        rel = x[senders] - x[receivers]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        w = _cubic_spline_kernel(dist, self._sph.smoothing_length, self._dim)
        edge_features = torch.cat(
            [rel / self._sph.smoothing_length, dist / self._sph.smoothing_length, w],
            dim=-1,
        )
        edge_index = torch.stack([senders, receivers])
        return edge_index, edge_features, dist.squeeze(-1), w.squeeze(-1), rel

    def _compute_density(
        self,
        edge_index: torch.Tensor,
        kernel_vals: torch.Tensor,
        nparticles: int,
    ) -> torch.Tensor:
        senders = edge_index[0]
        receivers = edge_index[1]
        contrib = kernel_vals * self._particle_mass
        rho = _index_add_zero(nparticles, receivers, contrib)
        # 自己寄与 m * W(0) を加えて密度の過小推定を防ぐ
        w0 = _cubic_spline_kernel(
            torch.zeros(1, device=rho.device, dtype=rho.dtype),
            self._sph.smoothing_length,
            self._dim,
        )
        rho = rho + self._particle_mass * w0.squeeze(0)
        return rho

    def _artificial_viscosity_acc(
        self,
        v: torch.Tensor,
        rho: torch.Tensor,
        edge_index: torch.Tensor,
        rel: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """Monaghan 型人工粘性による加速度項。"""

        if self._sph.alpha_viscosity == 0.0 and self._sph.beta_viscosity == 0.0:
            return torch.zeros_like(v)

        senders = edge_index[0]
        receivers = edge_index[1]

        v_rel = v[senders] - v[receivers]  # v_j - v_i
        v_dot_r = (v_rel * rel).sum(dim=-1)
        mask = v_dot_r < 0  # 収束しているペアのみ
        if not mask.any():
            return torch.zeros_like(v)

        h = self._sph.smoothing_length
        c = self._sph.sound_speed
        mu = h * v_dot_r / (dist**2 + self._sph.visc_eps * h * h)
        rho_ij = 0.5 * (rho[senders] + rho[receivers])
        Pi = (-self._sph.alpha_viscosity * c * mu + self._sph.beta_viscosity * mu * mu) / (
            rho_ij + 1e-8
        )
        Pi = torch.where(mask, Pi, torch.zeros_like(Pi))

        grad_w = _cubic_spline_kernel_grad(rel, dist, h, self._dim)
        acc_edges = -self._particle_mass * Pi.unsqueeze(-1) * grad_w
        acc = _index_add_zero(v.shape[0], receivers, acc_edges)
        return acc

    def _sph_potential(self, rho: torch.Tensor) -> torch.Tensor:
        u = _tait_internal_energy(
            rho=rho,
            rho0=self._sph.rest_density,
            sound_speed=self._sph.sound_speed,
            gamma=self._sph.gamma,
        )
        mass = torch.full_like(rho, self._particle_mass, dtype=rho.dtype)
        return (mass * u).sum()

    # ------------------------------------------------------------------
    # ハミルトニアンとその勾配
    def _hamiltonian(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        rho: torch.Tensor,
        particle_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        mass = torch.as_tensor(self._particle_mass, device=x.device, dtype=x.dtype)
        p = mass * v
        kinetic = 0.5 * ((p**2).sum(dim=-1) / mass).sum()

        potential_terms = []
        if self._ham_cfg.variant in {"varB", "varC"}:
            potential_terms.append(self._sph_potential(rho))

        delta_u = self._potential_net(
            x=x,
            v=v,
            rho=rho,
            particle_type=particle_type,
            edge_index=edge_index,
            edge_features=edge_features,
        )
        if self._ham_cfg.variant in {"varA", "varB", "varC"}:
            potential_terms.append((mass * delta_u.squeeze(-1)).sum())

        if self._include_external_potential and self._gravity is not None:
            g = self._gravity.to(x.device)
            potential_terms.append(-(mass * g * x).sum())

        return kinetic + sum(potential_terms)

    def _hamiltonian_dynamics(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        particle_type: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if create_graph:
            x = x.requires_grad_(True)
            v = v.requires_grad_(True)
        else:
            # 勾配グラフは保持しないが，力計算に必要な一次勾配は取る
            x = x.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)

        edge_index, edge_features, dist, kernel_vals, rel = self._build_edges(
            x, nparticles_per_example
        )
        rho = self._compute_density(edge_index, kernel_vals, x.shape[0])

        H = self._hamiltonian(
            x=x,
            v=v,
            rho=rho,
            particle_type=particle_type,
            edge_index=edge_index,
            edge_features=edge_features,
        )
        dH_dv, dH_dx = torch.autograd.grad(
            H, (v, x), create_graph=create_graph, retain_graph=create_graph
        )
        mass = torch.as_tensor(self._particle_mass, device=x.device, dtype=x.dtype)
        xdot = dH_dv / mass
        vdot = -dH_dx / mass
        # 非保存な人工粘性を加えて数値安定性を確保
        vdot = vdot + self._artificial_viscosity_acc(v, rho, edge_index, rel, dist)
        return xdot, vdot, rho.detach()

    # ------------------------------------------------------------------
    # 積分器
    def _integrate(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        particle_type: torch.Tensor,
        nparticles: torch.Tensor,
        *,
        create_graph: bool,
    ):
        dt = self._int_cfg.dt
        if self._int_cfg.type == "symplectic_euler":
            xdot, vdot, rho = self._hamiltonian_dynamics(
                x, v, particle_type, nparticles, create_graph=create_graph
            )
            v_new = v + dt * vdot
            x_new = x + dt * v_new
            x_new, v_new = self._apply_boundary_conditions(x_new, v_new)
            return x_new, v_new, vdot, rho

        # default: leapfrog
        xdot, vdot, rho = self._hamiltonian_dynamics(
            x, v, particle_type, nparticles, create_graph=create_graph
        )
        v_half = v + 0.5 * dt * vdot
        x_new = x + dt * v_half
        xdot_new, vdot_new, rho_new = self._hamiltonian_dynamics(
            x_new, v_half, particle_type, nparticles, create_graph=create_graph
        )
        v_new = v_half + 0.5 * dt * vdot_new
        x_new, v_new = self._apply_boundary_conditions(x_new, v_new)
        # 勾配計算用に最後の vdot_new を返す
        return x_new, v_new, vdot_new, rho_new

    # ------------------------------------------------------------------
    # 公開インタフェース
    def predict_positions(
        self,
        current_positions: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del material_property  # 現状では未使用
        x = current_positions[:, -1].to(self._device)
        prev = current_positions[:, -2].to(self._device)
        dt = self._int_cfg.dt
        v = (x - prev) / dt
        particle_types = particle_types.to(self._device)
        nparticles_per_example = nparticles_per_example.to(self._device)

        was_training = self.training
        if was_training:
            self.eval()
        try:
            # 予測時でも内部で力計算のための勾配が必要なので grad を有効化
            with torch.enable_grad():
                x_new, v_new, _, _ = self._integrate(
                    x,
                    v,
                    particle_types,
                    nparticles_per_example,
                    create_graph=False,
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
        del material_property  # 現状では未使用
        noisy_position_sequence = position_sequence + position_sequence_noise
        x = noisy_position_sequence[:, -1].to(self._device)
        prev = noisy_position_sequence[:, -2].to(self._device)
        dt = self._int_cfg.dt
        v = (x - prev) / dt
        particle_types = particle_types.to(self._device)
        nparticles_per_example = nparticles_per_example.to(self._device)

        create_graph = True
        with torch.set_grad_enabled(create_graph):
            _, _, vdot, _ = self._integrate(
                x,
                v,
                particle_types,
                nparticles_per_example,
                create_graph=create_graph,
            )

        acc = vdot
        acc_stats = self._normalization_stats["acceleration"]
        acc_mean = torch.as_tensor(acc_stats["mean"], device=acc.device, dtype=acc.dtype)
        acc_std = torch.as_tensor(acc_stats["std"], device=acc.device, dtype=acc.dtype)
        normalized_acc = (acc - acc_mean) / acc_std

        with torch.no_grad():
            next_position_adjusted = next_positions + position_sequence_noise[:, -1]
            target_acc = (next_position_adjusted - position_sequence[:, -1]) / dt
            target_acc = target_acc - (position_sequence[:, -1] - position_sequence[:, -2]) / dt
            target_acc = target_acc / dt
            target_normalized = (target_acc - acc_mean) / acc_std

        return normalized_acc, target_normalized

    def save(self, path: str = "model.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
