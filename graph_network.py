from typing import List
import torch
import torch.nn as nn

# --- Encoders -------------------------------------------------
def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:

  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
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
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
    layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp

