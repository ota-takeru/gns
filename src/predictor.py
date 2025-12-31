import pickle
from pathlib import Path
from typing import Any

import torch

import data_loader
import reading_utils
from rollout_utils import rollout
from simulator_factory import _get_simulator
from train_config import INPUT_SEQUENCE_LENGTH, Config
from train_paths import _resolve_model_path, _resolve_output_directory
from train_utils import _resolve_rollout_dataset_path


def _assert_model_finite(simulator: torch.nn.Module) -> None:
    """NaN/Inf を含む重み・バッファを検知して早期に失敗させる"""
    bad: list[str] = []
    for name, param in simulator.named_parameters():
        if param is None or not param.is_floating_point():
            continue
        if not torch.isfinite(param).all():
            bad.append(name)
    for name, buf in simulator.named_buffers():
        if buf is None or not buf.is_floating_point():
            continue
        if not torch.isfinite(buf).all():
            bad.append(f"[buffer]{name}")
    if bad:
        joined = ", ".join(bad)
        raise RuntimeError(f"Model contains NaN/Inf tensors: {joined}")


def predict(cfg: Config, device: torch.device):
    """valid / rollout モードの入口"""
    metadata_key = (
        cfg.active_scenario.rollout_metadata_split
        if cfg.active_scenario and cfg.active_scenario.rollout_metadata_split
        else "rollout"
    )
    metadata = reading_utils.read_metadata(cfg.data_path, metadata_key)
    simulator = _get_simulator(metadata, cfg.noise_std, cfg.noise_std, device, cfg)

    model_path = _resolve_model_path(cfg)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model does not exist at {model_path}")
    simulator.load(model_path)
    _assert_model_finite(simulator)
    simulator.to(device)
    simulator.eval()

    output_dir = _resolve_output_directory(cfg)

    dataset_path = _resolve_rollout_dataset_path(cfg)
    if dataset_path is None:
        raise FileNotFoundError(
            "No suitable dataset found for rollout/valid. Expected one of valid.npz, test.npz, or train.npz under the configured data_path, or a path specified via rollout_dataset."
        )

    ds = data_loader.get_data_loader_by_trajectories(dataset_path)

    first = ds.dataset[0]
    if len(first) == 4:
        material_property_as_feature = True
    elif len(first) == 3:
        material_property_as_feature = False
    else:
        raise NotImplementedError

    eval_loss = []
    with torch.no_grad():
        for example_i, features in enumerate(ds):
            if (
                cfg.mode == "rollout"
                and cfg.rollout_inference_max_examples is not None
                and example_i >= int(cfg.rollout_inference_max_examples)
            ):
                print(
                    f"Reached rollout_inference_max_examples={cfg.rollout_inference_max_examples}. Stopping."
                )
                break
            print(f"processing example number {example_i}")
            positions = features[0].to(device)
            if metadata.get("sequence_length") is not None:
                nsteps = int(metadata["sequence_length"]) - INPUT_SEQUENCE_LENGTH
            else:
                sequence_length = positions.shape[1]
                nsteps = int(sequence_length) - INPUT_SEQUENCE_LENGTH

            particle_type = features[1].to(device)
            if material_property_as_feature:
                material_property = features[2].to(device)
                n_particles_per_example = torch.tensor(
                    [int(features[3])], dtype=torch.int32
                ).to(device)
            else:
                material_property = None
                n_particles_per_example = torch.tensor(
                    [int(features[2])], dtype=torch.int32
                ).to(device)

            example_rollout, loss = rollout(
                simulator,
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
                nsteps,
                device,
            )
            example_rollout["metadata"] = metadata
            loss_mean = float(loss.mean().detach().cpu())
            print(f"Predicting example {example_i} loss: {loss_mean:.6f}")
            eval_loss.append(torch.flatten(loss))

            if cfg.mode == "rollout":
                example_rollout["loss"] = loss_mean
                filename = f"{cfg.output_filename}_ex{example_i}.pkl"
                with (output_dir / filename).open("wb") as f:
                    pickle.dump(example_rollout, f)

    print(f"Mean loss on rollout prediction: {torch.mean(torch.cat(eval_loss))}")


__all__ = ["predict"]
