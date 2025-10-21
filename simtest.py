import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict

class LearnedSimulator(nn.Module):
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
          boundaries: np.ndarray,
          normalization_stats: dict,
          nparticle_types: int,
          particle_type_embedding_size: int,
          boundary_clamp_limit: float = 1.0,
          device="cpu"
  ):
    super(LearnedSimulator, self).__init__()
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types
    self._boundary_clamp_limit = boundary_clamp_limit
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)
    self._device = device

  def forward(self):
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)
    edge_index = radius_graph(
        node_features, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=128)
    receivers = edge_index[0, :]
    senders = edge_index[1, :]
    return receivers, senders

  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None):
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]
    velocity_sequence = time_diff(position_sequence)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    node_features = []
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(
        nparticles, -1)
    node_features.append(flat_velocity_sequence)
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
        most_recent_position - boundaries[:, 0][None])
    distance_to_upper_boundary = (
        boundaries[:, 1][None] - most_recent_position)
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    node_features.append(normalized_clipped_distance_to_boundaries)
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)
    if material_property is not None:
        material_property = material_property.view(nparticles, 1)
        node_features.append(material_property)
    edge_features = []
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius
    edge_features.append(normalized_relative_displacements)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)
    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor) -> torch.tensor:
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]
    new_velocity = most_recent_velocity + acceleration
    new_position = most_recent_position + new_velocity
    return new_position

  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None) -> torch.tensor:
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types, material_property)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types)
    predicted_normalized_acceleration = self._encode_process_decode(
        node_features, edge_index, edge_features)
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    return next_positions

  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None):
    noisy_position_sequence = position_sequence + position_sequence_noise
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types, material_property)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types)
    predicted_normalized_acceleration = self._encode_process_decode(
        node_features, edge_index, edge_features)
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity
    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']
    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  return position_sequence[:, 1:] - position_sequence[:, :-1]