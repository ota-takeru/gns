from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import noise_utils
from learned_simulator import BaseSimulator
from losses import get_loss
from train_config import (
    INPUT_SEQUENCE_LENGTH,
    KINEMATIC_PARTICLE_ID,
    Config,
)
from train_utils import _unwrap_simulator


class RolloutEvaluator:
    """rollout 指標をオンデマンドで計算するユーティリティ"""

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_examples: int = 1,
    ):
        self._loader = loader
        self._device = device
        self._max_examples = max(1, int(max_examples))
        self._iterator: Any | None = None

    def evaluate(self, simulator: BaseSimulator) -> dict[str, float | None]:
        metrics_list: list[dict[str, float]] = []
        for _ in range(self._max_examples):
            example = self._next_example()
            if example is None:
                break
            metrics = self._evaluate_example(simulator, example)
            if metrics is not None:
                metrics_list.append(metrics)

        if not metrics_list:
            return {
                "rollout_rmse_mean": None,
                "rollout_rmse_last": None,
                "rollout_instability": None,
            }

        aggregated: dict[str, float] = {}
        for key in metrics_list[0]:
            aggregated[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return aggregated

    def _next_example(self):
        if self._iterator is None:
            self._iterator = iter(self._loader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            try:
                return next(self._iterator)
            except StopIteration:
                return None

    def _evaluate_example(
        self, simulator: BaseSimulator, example: tuple
    ) -> dict[str, float] | None:
        if len(example) == 4:
            positions, particle_type, material_property, n_particles = example
            material_property_tensor = material_property.to(self._device)
            n_particles_tensor = torch.tensor(
                [int(n_particles)], dtype=torch.int32, device=self._device
            )
        elif len(example) == 3:
            positions, particle_type, n_particles = example
            material_property_tensor = None
            n_particles_tensor = torch.tensor(
                [int(n_particles)], dtype=torch.int32, device=self._device
            )
        else:
            return None

        positions = positions.to(self._device)
        particle_type = particle_type.to(self._device)

        nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH
        if nsteps <= 0:
            return None

        _, loss = rollout(
            simulator,
            positions,
            particle_type,
            material_property_tensor,
            n_particles_tensor,
            nsteps,
            self._device,
            show_progress=False,
        )

        if loss.numel() == 0:
            return None

        rmse_per_step = torch.sqrt(loss.mean(dim=(1, 2)))
        rmse_mean = float(rmse_per_step.mean().item())
        rmse_last = float(rmse_per_step[-1].item())
        instability = float(rmse_last / rmse_mean) if rmse_mean > 0.0 else 0.0

        return {
            "rollout_rmse_mean": rmse_mean,
            "rollout_rmse_last": rmse_last,
            "rollout_instability": instability,
        }


def rollout(
    simulator: BaseSimulator | DDP,
    position: torch.Tensor,
    particle_types: torch.Tensor,
    material_property: torch.Tensor | None,
    n_particles_per_example: torch.Tensor,
    nsteps: int,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> tuple[dict[str, Any], torch.Tensor]:
    """逐次1ステップ予測をシフト窓で積み上げるロールアウト"""
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

    current_positions = initial_positions
    predictions_list = []

    step_iterator = tqdm(range(nsteps), total=nsteps) if show_progress else range(nsteps)

    core_simulator = _unwrap_simulator(simulator)

    for step_idx in step_iterator:
        next_position = core_simulator.predict_positions(
            current_positions,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_types,
            material_property=material_property,
        )
        kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).to(device)
        next_position_gt = ground_truth_positions[:, step_idx]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
        next_position = torch.where(kinematic_mask, next_position_gt, next_position)
        predictions_list.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

    predictions = torch.stack(predictions_list)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

    loss = (predictions - ground_truth_positions) ** 2

    output = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": particle_types.cpu().numpy(),
        "material_property": material_property.cpu().numpy()
        if material_property is not None
        else None,
    }
    return output, loss


def validation(
    simulator: BaseSimulator,
    example: tuple,
    feature_components: int,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    loss_fn = get_loss(cfg.loss)
    position = example[0][0].to(device)
    particle_type = example[0][1].to(device)
    if feature_components == 4:
        material_property = example[0][2].to(device)
        n_particles_per_example = example[0][3].to(device)
    elif feature_components == 3:
        material_property = None
        n_particles_per_example = example[0][2].to(device)
    else:
        raise NotImplementedError
    labels = example[1].to(device)

    noise_sampler = noise_utils.get_noise(cfg.noise)
    sampled_noise = noise_sampler(position, noise_std_last_step=cfg.noise_std).to(device)
    non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).to(device)
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

    with torch.no_grad():
        pred_acc, target_acc = simulator(
            next_positions=labels,
            position_sequence_noise=sampled_noise,
            position_sequence=position,
            nparticles_per_example=n_particles_per_example,
            particle_types=particle_type,
            material_property=material_property,
        )
    loss = loss_fn(pred_acc, target_acc, non_kinematic_mask)
    return loss


__all__ = ["RolloutEvaluator", "rollout", "validation"]
