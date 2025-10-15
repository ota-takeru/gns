import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from data_loader import GNSDataSample, load_gns_scene

FeaturesDict = Dict[str, np.ndarray]


def _parse_noise_std(values: Sequence[float]) -> Union[float, Tuple[float, ...]]:
    """Convert argparse float sequence into scalar or tuple."""
    cleaned = tuple(float(v) for v in values)
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned


def _select_scene_file(: Path, index: int, pattern: str = "scene_*.npz") -> Path:
    files = sorted(dataset_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No scene files matching '{pattern}' in {dataset_dir}")
    if index < 0:
        index += len(files)
    if index < 0 or index >= len(files):
        raise IndexError(f"Scene index {index} is out of range for {len(files)} files")
    return files[index]


def prepare_training_tensors(sample: GNSDataSample) -> Tuple[FeaturesDict, np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(sample.valid_targets)
    if valid_idx.size == 0:
        raise ValueError("Sample does not contain any valid target time steps.")

    rigid_over_time = np.broadcast_to(
        sample.rigid_id,
        (sample.noisy_positions.shape[0], sample.rigid_id.shape[0]),
    )

    features: FeaturesDict = {
        "noisy_position": sample.noisy_positions[valid_idx],
        "velocity": sample.velocities[valid_idx],
        "rigid_id": rigid_over_time[valid_idx],
    }
    targets = sample.accelerations[valid_idx]
    time_index = valid_idx.astype(np.int32, copy=False)
    return features, targets, time_index


def _report_scene(
    scene_path: Path,
    sample: GNSDataSample,
    features: FeaturesDict,
    targets: np.ndarray,
    time_index: np.ndarray,
) -> None:
    print(f"Loaded scene: {scene_path}")
    t_steps, num_particles, dims = sample.positions.shape
    print(f"Frames: {t_steps}, Particles: {num_particles}, Dimensions: {dims}")
    print(f"Valid target steps: {time_index.size}/{t_steps}")
    print(f"dt: {sample.dt}")
    meta_keys = ", ".join(sorted(sample.meta)) if sample.meta else "(none)"
    print(f"Metadata keys: {meta_keys}")
    print("Prepared feature tensors:")
    for name, array in features.items():
        print(f"  - {name}: shape={array.shape}, dtype={array.dtype}")
    print(f"Targets: shape={targets.shape}, dtype={targets.dtype}")
    print(f"Time indices used for training: {time_index[:8]}{'...' if time_index.size > 8 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and prepare one GNS scene for training.")
    default_dataset = Path(__file__).resolve().parent / "datasets" / "out" / "train"
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset,
        help=f"Directory containing scene npz files (default: {default_dataset})",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Index of the scene to load (supports negative indexing).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs="*",
        default=(0.0,),
        help="Optional noise standard deviation (scalar or per-dimension).",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=None,
        help="Clip magnitude applied to sampled noise before adding it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for noise generation.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    scene_path = _select_scene_file(, args.scene_index)
    noise_std = _parse_noise_std(args.noise_std)

    sample = load_gns_scene(
        scene_path,
        noise_std=noise_std,
        noise_clip=args.noise_clip,
        seed=args.seed,
    )

    features, targets, time_index = prepare_training_tensors(sample)
    _report_scene(scene_path, sample, features, targets, time_index)


if __name__ == "__main__":
    main()
