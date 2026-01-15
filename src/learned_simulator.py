from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import radius_graph

import graph_network

BOUNDARY_MODE_WALLS = "walls"
BOUNDARY_MODE_PERIODIC = "periodic"


class BaseSimulator(nn.Module, ABC):
    """手法ごとの共通インタフェース。新規手法はこれを継承して実装する。"""

    @abstractmethod
    def predict_positions(
        self,
        current_positions: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def predict_accelerations(
        self,
        next_positions: torch.Tensor,
        position_sequence_noise: torch.Tensor,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None,
    ): ...

    # DDP で forward 経由でも呼び出せるように predict_accelerations を委譲
    def forward(
        self,
        next_positions: torch.Tensor,
        position_sequence_noise: torch.Tensor,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None = None,
    ):
        return self.predict_accelerations(
            next_positions=next_positions,
            position_sequence_noise=position_sequence_noise,
            position_sequence=position_sequence,
            nparticles_per_example=nparticles_per_example,
            particle_types=particle_types,
            material_property=material_property,
        )

    @abstractmethod
    def save(self, path: str = "model.pt"): ...

    @abstractmethod
    def load(self, path: str): ...


class GNSSimulator(BaseSimulator):
    def __init__(
        self,
        particle_dimensions: int,  # 粒子の位置次元数
        nnode_in: int,  # ノード入力特徴量の次元数
        nedge_in: int,  # エッジ入力特徴量の次元数
        latent_dim: int,  # 潜在空間の次元数
        nmessage_passing_steps: int,  # メッセージパッシングのステップ数
        nmlp_layers: int,  # MLPの層数
        mlp_hidden_dim: int,  # MLPの隠れ層の次元数
        connectivity_radius: float,  # 接続半径
        normalization_stats: dict,  # 正規化のための統計情報
        nparticle_types: int,  # 粒子の種類数
        particle_type_embedding_size: int,  # 粒子タイプ埋め込みの次元数
        boundaries: np.ndarray,
        boundary_clamp_limit: float = 1.0,
        boundary_mode: str = BOUNDARY_MODE_WALLS,
        edge_relative_velocity: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._normalization_stats = normalization_stats
        self._particle_type_embedding = nn.Embedding(
            nparticle_types, particle_type_embedding_size
        )  # 粒子タイプの埋め込み層
        self._nparticle_types = nparticle_types
        self._connectivity_radius = connectivity_radius
        self._encode_process_decode = graph_network.EncodeProcessDecode(
            nnode_in_features=nnode_in,
            nnode_out_features=particle_dimensions,
            nedge_in_features=nedge_in,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        boundaries_arr = np.asarray(boundaries, dtype=np.float32)
        if boundaries_arr.ndim != 2 or boundaries_arr.shape[1] != 2:
            msg = f"Expected boundaries with shape (dim, 2); got {boundaries_arr.shape}"
            raise ValueError(msg)
        self._boundaries = boundaries_arr
        self._boundary_mode = self._normalize_boundary_mode(boundary_mode)
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            lengths = boundaries_arr[:, 1] - boundaries_arr[:, 0]
            if np.any(lengths <= 0):
                raise ValueError("Periodic boundaries require bounds with positive length.")
            self._periodic_length = lengths.astype(np.float32, copy=False)
            self._boundary_clamp_limit = 0.0
        else:
            self._periodic_length = None
            self._boundary_clamp_limit = boundary_clamp_limit
        self._device = device
        self._edge_relative_velocity = bool(edge_relative_velocity)
        self._neighbor_debug_logged = False
        self._neighbor_debug_info = None

    def forward(
        self,
        next_positions: torch.Tensor,
        position_sequence_noise: torch.Tensor,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None = None,
    ):
        return super().forward(
            next_positions=next_positions,
            position_sequence_noise=position_sequence_noise,
            position_sequence=position_sequence,
            nparticles_per_example=nparticles_per_example,
            particle_types=particle_types,
            material_property=material_property,
        )

    def _compute_graph_connectivity(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,  # 粒子数のリスト
        radius: float,  # 接続半径
        *,
        add_self_edges: bool = True,  # 自己エッジを追加するかどうか
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nparticles_per_example = nparticles_per_example.to("cpu")
        batch_ids = torch.cat(
            [
                torch.full((int(n),), i, dtype=torch.long)
                for i, n in enumerate(nparticles_per_example.tolist())
            ]
        ).to(self._device)
        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            backend = "dense_radius_graph_periodic"
            backend_note = None
            receivers, senders = self._dense_radius_graph_periodic(
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
                    max_num_neighbors=64,
                )
                receivers = edge_index[0, :]
                senders = edge_index[1, :]
            except (ImportError, RuntimeError) as e:
                backend = "dense_radius_graph"
                backend_note = f"{type(e).__name__}: {e}"
                receivers, senders = self._dense_radius_graph(
                    node_position, nparticles_per_example, radius, add_self_edges
                )
        self._log_neighbor_backend(
            backend=backend,
            node_position=node_position,
            batch_device=batch_ids.device,
            note=backend_note,
        )
        return receivers, senders

    def _dense_radius_graph(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        add_self_edges: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback radius graph construction without torch-cluster dependency."""
        device = node_position.device
        receivers: list[torch.Tensor] = []
        senders: list[torch.Tensor] = []
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
                receivers.append(r_idx + start_idx)
                senders.append(s_idx + start_idx)
            start_idx += n

        if receivers:
            receivers_cat = torch.cat(receivers).to(device)
            senders_cat = torch.cat(senders).to(device)
        else:
            receivers_cat = torch.empty(0, dtype=torch.long, device=device)
            senders_cat = torch.empty(0, dtype=torch.long, device=device)
        return receivers_cat, senders_cat

    def _dense_radius_graph_periodic(
        self,
        node_position: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        radius: float,
        add_self_edges: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = node_position.device
        receivers: list[torch.Tensor] = []
        senders: list[torch.Tensor] = []
        start_idx = 0
        eps = 1e-8
        for n in nparticles_per_example.tolist():
            n = int(n)
            if n == 0:
                continue
            slice_pos = node_position[start_idx : start_idx + n]
            delta = slice_pos[:, None, :] - slice_pos[None, :, :]
            delta = self._minimum_image_displacement(delta)
            dist = torch.linalg.norm(delta, dim=-1)
            mask = dist <= (radius + eps)
            if not add_self_edges:
                idx = torch.arange(n, device=device)
                mask[idx, idx] = False
            r_idx, s_idx = torch.nonzero(mask, as_tuple=True)
            if r_idx.numel() > 0:
                receivers.append(r_idx + start_idx)
                senders.append(s_idx + start_idx)
            start_idx += n

        if receivers:
            receivers_cat = torch.cat(receivers).to(device)
            senders_cat = torch.cat(senders).to(device)
        else:
            receivers_cat = torch.empty(0, dtype=torch.long, device=device)
            senders_cat = torch.empty(0, dtype=torch.long, device=device)
        return receivers_cat, senders_cat

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
        info = {
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
        print("[neighbor][GNS]", " ".join(parts))
        self._neighbor_debug_logged = True

    def get_neighbor_debug_info(self):
        """Return cached neighbor-search device info (first call only)."""
        return self._neighbor_debug_info

    def _encoder_preprocessor(
        self,
        position_sequence: torch.Tensor,  # [N, 6, D]
        nparticles_per_example: torch.Tensor,  # [粒子数, 粒子数, ...]。複数のシーンを同時に扱うため、近傍計算の際にシーンを区別できるようにするため。  # noqa: E501
        particle_types: torch.Tensor,  # [N],
        material_property: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # node_features、edge_features、edge_indexを生成。
        # ------node_featuresの計算------
        node_features_list = []
        nparticles = position_sequence.shape[0]  # 粒子数
        most_recent_position = position_sequence[:, -1]  # 最新の位置
        velocity_sequence = time_diff(
            position_sequence,
            boundaries=self._boundaries,
            boundary_mode=self._boundary_mode,
        )  # 速度列...

        # 速度列の正規化とフラット化
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
        node_features_list.append(flat_velocity_sequence)

        if self._boundary_mode == BOUNDARY_MODE_PERIODIC:
            boundary_feature_dim = int(self._boundaries.shape[0]) * 2
            node_features_list.append(
                torch.zeros(
                    (nparticles, boundary_feature_dim),
                    device=self._device,
                    dtype=most_recent_position.dtype,
                )
            )
        else:
            boundaries = (
                torch.tensor(self._boundaries, requires_grad=False)
                .float()
                .to(self._device)
            )
            distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
            distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
            distance_to_boundaries = torch.cat(
                [distance_to_lower_boundary, distance_to_upper_boundary], dim=1
            )
            normalized_clipped_distance_to_boundaries = torch.clamp(
                distance_to_boundaries / self._connectivity_radius,
                -self._boundary_clamp_limit,
                self._boundary_clamp_limit,
            )
            node_features_list.append(normalized_clipped_distance_to_boundaries)

        # 粒子タイプの埋め込みを計算
        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features_list.append(particle_type_embeddings)

        if material_property is not None:
            material_property = material_property.view(nparticles, 1)
            node_features_list.append(material_property)

        node_features = torch.cat(node_features_list, dim=-1)

        # ------edge_featuresの計算------
        senders, receivers = self._compute_graph_connectivity(
            most_recent_position, nparticles_per_example, self._connectivity_radius
        )
        relative_displacements = self._relative_displacement(
            most_recent_position[senders, :], most_recent_position[receivers, :]
        )
        normalized_relative_displacements = (
            relative_displacements / self._connectivity_radius
        )  # [E, D]
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )  # [E, 1]

        edge_features_list = [normalized_relative_displacements, normalized_relative_distances]
        if self._edge_relative_velocity:
            # 法線/接線分解は「生の速度差分」で行い、最後にスカラーで正規化する
            most_recent_velocity_raw = velocity_sequence[:, -1]  # [N, D]
            rel_v = most_recent_velocity_raw[senders, :] - most_recent_velocity_raw[receivers, :]  # [E, D]
            eps = 1e-8
            dist = torch.norm(relative_displacements, dim=-1, keepdim=True)  # [E, 1]
            r_hat = relative_displacements / (dist + eps)  # 単位ベクトル（幾何に忠実）
            v_n = (rel_v * r_hat).sum(dim=-1, keepdim=True)  # [E, 1]
            v_t = rel_v - v_n * r_hat  # [E, D]
            v_t_mag = torch.norm(v_t, dim=-1, keepdim=True)  # [E, 1]

            v_std = velocity_stats["std"]
            if torch.is_tensor(v_std):
                v_scale = v_std.mean()
            else:
                v_scale = torch.as_tensor(v_std, device=rel_v.device, dtype=rel_v.dtype).mean()

            v_n = v_n / (v_scale + eps)
            v_t_mag = v_t_mag / (v_scale + eps)

            edge_features_list.extend([v_n, v_t_mag])
        edge_features = torch.cat(edge_features_list, dim=-1)

        # ------edge_indexの計算------
        edge_index = torch.stack([senders, receivers])  # [2, E]

        return node_features, edge_index, edge_features

    def _decoder_postprocessor(
        self, normalized_acceleration: torch.Tensor, position_sequence: torch.Tensor
    ) -> torch.Tensor:  # 正規化された加速度から次の位置を計算
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (normalized_acceleration * acceleration_stats["std"]) + acceleration_stats[
            "mean"
        ]

        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = self._relative_displacement(
            position_sequence[:, -1], position_sequence[:, -2]
        )

        new_velocity = most_recent_velocity + acceleration
        new_position = most_recent_position + new_velocity
        return new_position

    def _inverse_decoder_postprocessor(
        self, next_position: torch.Tensor, position_sequence: torch.Tensor
    ) -> torch.Tensor:  # 次の位置、ノイズ付き位置列から正規化された加速度を計算
        previous_position = position_sequence[:, -1]
        previous_velocity = self._relative_displacement(
            position_sequence[:, -1], position_sequence[:, -2]
        )
        next_velocity = self._relative_displacement(next_position, previous_position)
        acceleration = next_velocity - previous_velocity

        # 正規化
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (acceleration - acceleration_stats["mean"]) / acceleration_stats[
            "std"
        ]
        return normalized_acceleration

    def predict_positions(
        self,
        current_positions: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None = None,
    ):  # 次の位置を予測
        if material_property is not None:
            node_features, edge_index, edge_features = self._encoder_preprocessor(
                current_positions,
                nparticles_per_example,
                particle_types,
                material_property,
            )
        else:
            node_features, edge_index, edge_features = self._encoder_preprocessor(
                current_positions, nparticles_per_example, particle_types
            )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features
        )  # 正規化された加速度を予測
        predicted_positions = self._decoder_postprocessor(
            predicted_normalized_acceleration, current_positions
        )  # 次の位置を計算
        predicted_positions = self._wrap_positions(predicted_positions)
        return predicted_positions

    def predict_accelerations(
        self,
        next_positions: torch.Tensor,
        position_sequence_noise: torch.Tensor,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor | None,
    ):
        noisy_position_sequence = position_sequence + position_sequence_noise
        noisy_position_sequence = self._wrap_positions(noisy_position_sequence)
        if material_property is not None:
            node_features, edge_index, edge_features = self._encoder_preprocessor(
                noisy_position_sequence,
                nparticles_per_example,
                particle_types,
                material_property,
            )
        else:
            node_features, edge_index, edge_features = self._encoder_preprocessor(
                noisy_position_sequence, nparticles_per_example, particle_types
            )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features
        )

        with torch.no_grad():
            next_position_adjusted = (
                next_positions + position_sequence_noise[:, -1]
            )  # 正確な教師データを用意するために、next_positionsにノイズを加算することでノイズの影響を相殺させる。  # noqa: E501, RUF003
            next_position_adjusted = self._wrap_positions(next_position_adjusted)
            target_normalized_acceleration = self._inverse_decoder_postprocessor(
                next_position_adjusted, noisy_position_sequence
            )  # 正規化された目標加速度を計算

        return predicted_normalized_acceleration, target_normalized_acceleration

    def save(self, path: str = "model.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=torch.device("cpu"))

        # 互換性: edge_relative_velocity 導入前に学習したチェックポイントは
        # エッジ入力次元が 3 のまま。現在のモデルが 6 次元を期待する場合は、
        # 既存重みをコピーし、新規チャネルを 0 でパディングして読み込む。
        weight_key = "_encode_process_decode.encoder.edge_fn.0.NN-0.weight"
        if weight_key in state_dict:
            current_weight = self._encode_process_decode.encoder.edge_fn[0][0].weight
            target_in = current_weight.shape[1]
            loaded_weight = state_dict[weight_key]
            loaded_in = loaded_weight.shape[1]
            if loaded_in != target_in:
                if loaded_in < target_in and self._edge_relative_velocity:
                    pad_cols = target_in - loaded_in
                    pad = torch.zeros(
                        loaded_weight.shape[0],
                        pad_cols,
                        device=loaded_weight.device,
                        dtype=loaded_weight.dtype,
                    )
                    state_dict[weight_key] = torch.cat([loaded_weight, pad], dim=1)
                    print(
                        "[learned_simulator] 旧チェックポイントを edge_relative_velocity "
                        "対応に合わせるため、edge MLP 入力重みをパディングしました。"
                    )
                else:
                    raise RuntimeError(
                        f"Edge MLP 入力次元が一致しません (checkpoint {loaded_in} != model {target_in}). "
                        "モデル再学習か設定の見直しが必要です。"
                    )

        self.load_state_dict(state_dict)

    def _normalize_boundary_mode(self, mode: str) -> str:
        value = str(mode).strip().lower()
        if value not in {BOUNDARY_MODE_WALLS, BOUNDARY_MODE_PERIODIC}:
            raise ValueError(
                f"Unsupported boundary_mode '{mode}'. Use '{BOUNDARY_MODE_WALLS}' or '{BOUNDARY_MODE_PERIODIC}'."
            )
        return value

    def _wrap_positions(self, positions: torch.Tensor) -> torch.Tensor:
        if self._boundary_mode != BOUNDARY_MODE_PERIODIC:
            return positions
        boundaries = torch.as_tensor(self._boundaries, device=positions.device, dtype=positions.dtype)
        lower = boundaries[:, 0]
        length = boundaries[:, 1] - boundaries[:, 0]
        return lower + torch.remainder(positions - lower, length)

    def _minimum_image_displacement(self, delta: torch.Tensor) -> torch.Tensor:
        if self._boundary_mode != BOUNDARY_MODE_PERIODIC:
            return delta
        if self._periodic_length is None:
            raise RuntimeError("Periodic lengths are not available.")
        length = torch.as_tensor(self._periodic_length, device=delta.device, dtype=delta.dtype)
        half = 0.5 * length
        return torch.remainder(delta + half, length) - half

    def _relative_displacement(self, pos_a: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
        delta = pos_a - pos_b
        return self._minimum_image_displacement(delta)


def time_diff(  # 速度列を計算
    position_sequence: torch.Tensor,
    *,
    boundaries: np.ndarray | torch.Tensor | None = None,
    boundary_mode: str = BOUNDARY_MODE_WALLS,
) -> torch.Tensor:
    delta = position_sequence[:, 1:] - position_sequence[:, :-1]
    if boundaries is None or boundary_mode != BOUNDARY_MODE_PERIODIC:
        return delta
    boundaries_t = torch.as_tensor(
        boundaries, device=position_sequence.device, dtype=position_sequence.dtype
    )
    if boundaries_t.ndim != 2 or boundaries_t.shape[1] != 2:
        raise ValueError("Expected boundaries with shape (dim, 2) for periodic diff.")
    length = boundaries_t[:, 1] - boundaries_t[:, 0]
    half = 0.5 * length
    return torch.remainder(delta + half, length) - half


# 互換性のためのエイリアス
LearnedSimulator = GNSSimulator


def get_simulator_class(method_name: str):
    """手法名から対応するクラスを取得する。"""
    if method_name == "gns":
        return GNSSimulator
    if method_name == "hamiltonian_sph":
        # 循環 import を避けるため遅延 import
        from hamiltonian_sph import HamiltonianSPHSimulator

        return HamiltonianSPHSimulator

    known = ", ".join(["gns", "hamiltonian_sph"])
    raise ValueError(f"Unknown method '{method_name}'. known: {known}")
