from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn
Tensor = torch.Tensor


class GraphInputs(NamedTuple):
    node_features: Tensor
    edge_features: Tensor
    senders: Tensor
    receivers: Tensor
    num_nodes: int
    num_edges: int


def radius_graph(
    positions: Tensor,
    radius: float,
    *,
    include_self_edges: bool = False,
    max_neighbors: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    if positions.ndim != 2:
        raise ValueError(f"positions must be [N, D], got shape {tuple(positions.shape)}")
    num_nodes = positions.shape[0]

    disp = positions.unsqueeze(0) - positions.unsqueeze(1)
    sq_dist = disp.pow(2).sum(-1)

    mask = sq_dist <= radius * radius
    if not include_self_edges:
        mask.fill_diagonal_(False)

    receivers, senders = mask.nonzero(as_tuple=True)
    if receivers.numel() == 0:
        empty = positions.new_empty((0,), dtype=torch.long)
        return empty, empty, positions.new_empty((0, positions.shape[1]), dtype=positions.dtype)

    if max_neighbors is not None and max_neighbors > 0:
        keep_mask = torch.zeros(receivers.numel(), dtype=torch.bool, device=positions.device)
        for node in range(num_nodes):
            idx = (receivers == node).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            if idx.numel() <= max_neighbors:
                keep_mask[idx] = True
                continue
            node_sq = sq_dist[node, senders[idx]]
            topk = torch.topk(node_sq.neg(), max_neighbors).indices
            keep_mask[idx[topk]] = True
        receivers = receivers[keep_mask]
        senders = senders[keep_mask]

    rel_disp = positions[senders] - positions[receivers]
    return senders, receivers, rel_disp


def build_gns_graph_inputs(
    positions: Tensor,
    velocities: Tensor,
    *,
    radius: float,
    max_neighbors: Optional[int] = None,
    include_self_edges: bool = False,
) -> GraphInputs:
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must share shape [N, D]")
    senders, receivers, rel_pos = radius_graph(
        positions,
        radius,
        include_self_edges=include_self_edges,
        max_neighbors=max_neighbors,
    )
    num_nodes = positions.shape[0]
    num_edges = senders.numel()

    if num_edges == 0:
        node_deg = torch.zeros((num_nodes, 1), dtype=positions.dtype, device=positions.device)
    else:
        deg = torch.zeros((num_nodes,), dtype=positions.dtype, device=positions.device)
        deg.index_add_(0, receivers, torch.ones_like(receivers, dtype=positions.dtype))
        node_deg = deg.unsqueeze(-1)

    node_center = positions.mean(dim=0, keepdim=True)
    node_features = torch.cat((velocities, positions - node_center, node_deg), dim=-1)

    edge_dim = positions.shape[1] * 2 + 2
    if num_edges == 0:
        edge_features = positions.new_zeros((0, edge_dim))
    else:
        rel_vel = velocities[senders] - velocities[receivers]
        dist = rel_pos.norm(dim=-1, keepdim=True)
        sq_dist = (rel_pos * rel_pos).sum(-1, keepdim=True)
        edge_features = torch.cat((rel_pos, rel_vel, dist, sq_dist), dim=-1)

    return GraphInputs(
        node_features=node_features,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        *,
        num_layers: int = 2,
        activation: nn.Module = nn.ReLU(),
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        output_dim = output_dim or hidden_dim
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != num_layers - 1:
                layers.append(activation)
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GraphNetBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        layer_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.edge_mlp = MLP(
            input_dim=latent_dim * 3,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            num_layers=2,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.node_mlp = MLP(
            input_dim=latent_dim * 2,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            num_layers=2,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.residual = residual

    @staticmethod
    def _aggregate(messages: Tensor, receivers: Tensor, num_nodes: int) -> Tensor:
        agg = messages.new_zeros((num_nodes, messages.shape[-1]))
        if messages.numel() == 0:
            return agg
        agg.index_add_(0, receivers, messages)
        return agg

    def forward(
        self,
        node_latent: Tensor,
        edge_latent: Tensor,
        senders: Tensor,
        receivers: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        sender_feat = node_latent[senders]
        receiver_feat = node_latent[receivers]

        edge_input = torch.cat((edge_latent, sender_feat, receiver_feat), dim=-1)
        edge_update = self.edge_mlp(edge_input)
        new_edge = edge_latent + edge_update if self.residual else edge_update

        aggregated = self._aggregate(new_edge, receivers, node_latent.shape[0])
        node_input = torch.cat((node_latent, aggregated), dim=-1)
        node_update = self.node_mlp(node_input)
        new_node = node_latent + node_update if self.residual else node_update

        return new_node, new_edge


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        latent_dim: int,
        *,
        message_passing_steps: int = 10,
        layer_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
        decoder_layers: int = 2,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.node_encoder = MLP(
            input_dim=node_input_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            num_layers=2,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.edge_encoder = MLP(
            input_dim=edge_input_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            num_layers=2,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.processor = nn.ModuleList(
            [
                GraphNetBlock(
                    latent_dim,
                    layer_norm=layer_norm,
                    dropout=dropout,
                    residual=residual,
                )
                for _ in range(message_passing_steps)
            ]
        )
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dim=latent_dim,
            output_dim=output_dim,
            num_layers=decoder_layers,
            layer_norm=layer_norm,
            dropout=dropout,
        )

    def forward(
        self,
        graph_inputs: GraphInputs,
    ) -> Tensor:
        node_latent = self.node_encoder(graph_inputs.node_features)
        edge_latent = self.edge_encoder(graph_inputs.edge_features)
        for block in self.processor:
            node_latent, edge_latent = block(
                node_latent,
                edge_latent,
                graph_inputs.senders,
                graph_inputs.receivers,
            )
        return self.decoder(node_latent)


class GNSModel(nn.Module):
    def __init__(
        self,
        dim_world: int,
        *,
        latent_dim: int = 128,
        message_passing_steps: int = 10,
        radius: float = 0.08,
        include_rigid_features: bool = True,
        num_rigid_ids: int = 64,
        rigid_embedding_dim: int = 16,
        max_neighbors: Optional[int] = None,
        layer_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = True,
        decoder_layers: int = 2,
        output_dim: int = None,
    ) -> None:
        super().__init__()
        self.dim_world = dim_world
        self.radius = radius
        self.include_rigid_features = include_rigid_features
        self.max_neighbors = max_neighbors

        node_dim = dim_world * 2 + 1
        edge_dim = dim_world * 2 + 2
        if include_rigid_features:
            if num_rigid_ids <= 0 or rigid_embedding_dim <= 0:
                raise ValueError("num_rigid_ids and rigid_embedding_dim must be positive")
            self.rigid_emb = nn.Embedding(num_rigid_ids, rigid_embedding_dim)
            node_dim += self.rigid_emb.embedding_dim
            edge_dim += 1
        else:
            self.rigid_emb = None

        self.model = EncodeProcessDecode(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            latent_dim=latent_dim,
            message_passing_steps=message_passing_steps,
            layer_norm=layer_norm,
            dropout=dropout,
            residual=residual,
            decoder_layers=decoder_layers,
            output_dim=output_dim or dim_world,
        )

    def _prepare_graph(
        self,
        positions: Tensor,
        velocities: Tensor,
        rigid_id: Optional[Tensor],
    ) -> GraphInputs:
        if self.rigid_emb is None:
            rigid_feature = None
        elif rigid_id is None:
            raise ValueError("rigid_id must be provided when include_rigid_features=True")
        else:
            if rigid_id.ndim != 1 or rigid_id.shape[0] != positions.shape[0]:
                raise ValueError("rigid_id must be [N]")
            rigid_index = rigid_id.to(device=positions.device, dtype=torch.long)
            if rigid_index.max() >= self.rigid_emb.num_embeddings:
                raise ValueError(
                    f"rigid_id contains value {int(rigid_index.max())} "
                    f"but embedding supports only {self.rigid_emb.num_embeddings} ids"
                )
            rigid_feature = self.rigid_emb(rigid_index)

        base_graph = build_gns_graph_inputs(
            positions,
            velocities,
            radius=self.radius,
            max_neighbors=self.max_neighbors,
        )
        if self.rigid_emb is None:
            return base_graph

        node_features = torch.cat((base_graph.node_features, rigid_feature), dim=-1)
        if base_graph.num_edges == 0:
            edge_features = torch.cat(
                (
                    base_graph.edge_features,
                    base_graph.edge_features.new_zeros((0, 1)),
                ),
                dim=-1,
            )
        else:
            same_rigid = (rigid_index[base_graph.senders] == rigid_index[base_graph.receivers]).to(
                base_graph.edge_features.dtype
            ).unsqueeze(-1)
            edge_features = torch.cat((base_graph.edge_features, same_rigid), dim=-1)

        return GraphInputs(
            node_features=node_features,
            edge_features=edge_features,
            senders=base_graph.senders,
            receivers=base_graph.receivers,
            num_nodes=base_graph.num_nodes,
            num_edges=base_graph.num_edges,
        )

    def forward(
        self,
        noisy_position: Tensor,
        velocity: Tensor,
        rigid_id: Optional[Tensor] = None,
    ) -> Tensor:
        if noisy_position.ndim == 3:
            outputs = []
            for idx in range(noisy_position.shape[0]):
                rid = None
                if rigid_id is not None:
                    if rigid_id.ndim == 2:
                        rid = rigid_id[idx]
                    else:
                        rid = rigid_id
                outputs.append(self.forward(noisy_position[idx], velocity[idx], rid))
            return torch.stack(outputs, dim=0)
        if noisy_position.ndim != 2 or velocity.ndim != 2:
            raise ValueError("noisy_position and velocity must have shape [N, D] or [B, N, D]")
        graph = self._prepare_graph(noisy_position, velocity, rigid_id)
        return self.model(graph)
