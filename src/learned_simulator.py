import torch
from torch import nn
from torch_geometric.nn import radius_graph

import graph_network


class LearnedSimulator(nn.Module):
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
        self._device = device

    def forward(self):
        pass

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
        try:
            edge_index = radius_graph(
                node_position,
                r=radius,
                batch=batch_ids,
                loop=add_self_edges,
                max_num_neighbors=128,
            )
            receivers = edge_index[0, :]
            senders = edge_index[1, :]
        except ImportError:
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
        velocity_sequence = time_diff(position_sequence)  # 速度列...

        # 速度列の正規化とフラット化
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
        node_features_list.append(flat_velocity_sequence)

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
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius  # [E, D]
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )  # [E, 1]

        edge_features = torch.cat(
            [normalized_relative_displacements, normalized_relative_distances], dim=-1
        )

        # ------edge_indexの計算------
        edge_index = torch.stack([senders, receivers])  # [2, E]

        return node_features, edge_index, edge_features

    def _decoder_postprocessor(
        self, normalized_acceleration: torch.Tensor, position_sequence: torch.Tensor
    ) -> torch.Tensor:  # 正規化された加速度から次の位置を計算
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats["std"]
        ) + acceleration_stats["mean"]

        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = position_sequence[:, -1] - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration
        new_position = most_recent_position + new_velocity
        return new_position

    def _inverse_decoder_postprocessor(
        self, next_position: torch.Tensor, position_sequence: torch.Tensor
    ) -> torch.Tensor:  # 次の位置、ノイズ付き位置列から正規化された加速度を計算
        previous_position = position_sequence[:, -1]
        previous_velocity = position_sequence[:, -1] - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        # 正規化
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats["mean"]
        ) / acceleration_stats["std"]
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
            target_normalized_acceleration = self._inverse_decoder_postprocessor(
                next_position_adjusted, noisy_position_sequence
            )  # 正規化された目標加速度を計算

        return predicted_normalized_acceleration, target_normalized_acceleration

    def save(self, path: str = "model.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


def time_diff(  # 速度列を計算
    position_sequence: torch.Tensor,
) -> torch.Tensor:
    return position_sequence[:, 1:] - position_sequence[:, :-1]
