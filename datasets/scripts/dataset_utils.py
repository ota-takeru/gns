import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np 


def _compute_differences(sequence: np.ndarray) -> np.ndarray:
    """Compute first-order differences along the time axis without dt scaling."""
    return sequence[1:] - sequence[:-1]


def collect_statistics(
    trajectories: Iterable[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate velocity/acceleration stats across trajectories for metadata."""
    velocity_rows: List[np.ndarray] = []
    acceleration_rows: List[np.ndarray] = []
    for pos in trajectories:
        vel = _compute_differences(pos)
        if vel.size:
            velocity_rows.append(vel.reshape(-1, vel.shape[-1]))
            acc = _compute_differences(vel)
            if acc.size:
                acceleration_rows.append(acc.reshape(-1, acc.shape[-1]))

    dim = trajectories[0].shape[-1]
    if velocity_rows:
        vel_stack = np.concatenate(velocity_rows, axis=0)
        vel_mean = vel_stack.mean(axis=0)
        vel_std = vel_stack.std(axis=0)
    else:
        vel_mean = vel_std = np.zeros((dim,), dtype=np.float32)

    if acceleration_rows:
        acc_stack = np.concatenate(acceleration_rows, axis=0)
        acc_mean = acc_stack.mean(axis=0)
        acc_std = acc_stack.std(axis=0)
    else:
        acc_mean = acc_std = np.zeros((dim,), dtype=np.float32)

    return vel_mean, vel_std, acc_mean, acc_std


def estimate_connectivity_radius(trajectories: Iterable[np.ndarray]) -> float:
    """Heuristic connectivity radius based on nearest-neighbour distance."""
    nearest_distances: List[np.ndarray] = []
    for pos in trajectories:
        frames = pos[: min(5, pos.shape[0])]
        for frame in frames:
            n = frame.shape[0]
            if n < 2:
                continue
            diff = frame[:, None, :] - frame[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dist, np.inf)
            nn = np.min(dist, axis=1)
            nn = nn[np.isfinite(nn)]
            if nn.size:
                nearest_distances.append(nn)

    if not nearest_distances:
        return 1.0

    all_nn = np.concatenate(nearest_distances, axis=0)
    # Use a generous multiplier so that the message graph connects over shapes.
    return float(np.percentile(all_nn, 90) * 1.5)


def export_dataset(
    trajectories: List[np.ndarray],
    particle_types_list: List[np.ndarray],
    out_dir: Path,
    split: str,
    dt: float,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Save aggregated dataset compatible with data_loader along with metadata.

    Note: We store particle_types as an array per trajectory to support mixed types
    (dynamic and kinematic particles in the same scene).
    """
    if not trajectories:
        raise ValueError("No trajectories provided for dataset export.")

    entries: List[Tuple[np.ndarray, np.ndarray]] = []
    for pos, ptypes in zip(trajectories, particle_types_list):
        entries.append((pos.astype(np.float32, copy=False), ptypes.astype(np.int32)))

    gns_data = np.array(entries, dtype=object)
    dataset_path = out_dir / f"{split}.npz"
    np.savez_compressed(dataset_path, gns_data=gns_data)

    vel_mean, vel_std, acc_mean, acc_std = collect_statistics(trajectories)
    dim = int(trajectories[0].shape[-1])
    input_sequence_length = 6  # Must match INPUT_SEQUENCE_LENGTH in training.
    particle_type_embedding = 16  # Must match particle_type_embedding_size in simulator.
    velocity_feature_dim = dim * (input_sequence_length - 1)
    boundary_feature_dim = dim * 2
    metadata = {
        "dim": dim,
        "sequence_length": int(trajectories[0].shape[0]),
        "dt": float(dt),
        "default_connectivity_radius": estimate_connectivity_radius(trajectories),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
        "nnode_in": int(
            velocity_feature_dim + boundary_feature_dim + particle_type_embedding
        ),
        "nedge_in": int(dim + 1),
        "boundary_feature_dim": int(boundary_feature_dim),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump({"train": metadata, "rollout": metadata}, fp, indent=2)

    return dataset_path, meta_path
