from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


# --- Encoders -------------------------------------------------
def build_mlp(
    input_size: int,
    hidden_layer_sizes: list[int],
    output_size: Optional[int] = None,
    output_activation: type[nn.Module] = nn.Identity,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    # Size of each layer
    layer_sizes = [input_size, *hidden_layer_sizes]
    if output_size is not None:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())
    return mlp


class Encoder(nn.Module):
    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        nedge_out_features: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()

        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in_features,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out_features,
                ),
                nn.LayerNorm(nnode_out_features),
            ]
        )
        # Encode edge features as an MLP
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nedge_in_features,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out_features,
                ),
                nn.LayerNorm(nedge_out_features),
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_nodes = self.node_fn(x)
        encoded_edges = self.edge_fn(edge_features)
        return encoded_nodes, encoded_edges


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__(aggr="add")
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nedge_out,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out,
                ),
                nn.LayerNorm(nnode_out),
            ]
        )
        # Edge MLP
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nnode_in + nedge_in,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out,
                ),
                nn.LayerNorm(nedge_out),
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # edge_indexは[2, E]
        x_residual = x  # 残差接続のために元のノード特徴量を保存
        edge_residual = edge_features  # 残差接続のために元のエッジ特徴量を保存

        # message → aggregate → update の順にメソッドを自動で実行
        self._edge_features = None
        x = self.propagate(edge_index, x=x, edge_attr=edge_features)
        if self._edge_features is None:
            raise RuntimeError("edge features were not computed during message passing.")
        edge_features = self._edge_features
        x += x_residual  # ノード特徴量に残差接続を追加
        edge_features += edge_residual  # エッジ特徴量に残差接続を追加
        return x, edge_features

    # エッジの更新
    def message(
        self,
        x_j: torch.Tensor,
        x_i: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_i is None:
            msg = "x_i is required"
            raise ValueError(msg)
        if edge_attr is None:
            msg = "edge_attr is required"
            raise ValueError(msg)
        # x_i: 受信ノードの特徴量, x_j: 送信ノードの特徴量
        edge_input = torch.cat([x_j, x_i, edge_attr], dim=-1)
        self._edge_features = self.edge_fn(edge_input)
        return self._edge_features

    # ノードの更新
    def update(
        self,
        inputs: torch.Tensor,  # ← PyGはここに集約後を渡す(aggr_out)
        x: Optional[torch.Tensor] = None,  # ← もとのノード特徴を任意で受ける
    ) -> torch.Tensor:
        if x is None:
            msg = "x is required"
            raise ValueError(msg)
        out = torch.cat([x, inputs], dim=-1)
        return self.node_fn(out)


class Processer(nn.Module):
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        nmessage_passing_steps: int,
    ) -> None:
        super().__init__()

        self.gnn_stack = nn.ModuleList(
            [
                InteractionNetwork(
                    nnode_in=nnode_in,
                    nnode_out=nnode_out,
                    nedge_in=nedge_in,
                    nedge_out=nedge_out,
                    nmlp_layers=nmlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(nmessage_passing_steps)  # message passingの回数
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        for gnn in self.gnn_stack:
            x, edge_features = gnn(x, edge_index, edge_features)
        return x


class Decoder(nn.Module):
    def __init__(
        self, nnode_in: int, nnode_out: int, nmlp_layers: int, mlp_hidden_dim: int
    ) -> None:
        super().__init__()
        self.node_fn = build_mlp(nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            nnode_in_features=nnode_in_features,
            nnode_out_features=latent_dim,
            nedge_in_features=nedge_in_features,
            nedge_out_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self.processer = Processer(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            nmessage_passing_steps=nmessage_passing_steps,
        )

        self.decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        x, edge_features = self.encoder(x, edge_features)
        x = self.processer(x, edge_index, edge_features)
        x = self.decoder(x)
        return x
