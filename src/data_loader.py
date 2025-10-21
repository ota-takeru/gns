import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


class NormalizationStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray


class GNSDataSample(NamedTuple):
    """Container for one scene worth of preprocessed arrays."""

    positions: np.ndarray  # (T, N, D)
    velocities: np.ndarray  # (T, N, D)
    accelerations: np.ndarray  # (T, N, D)
    noisy_positions: np.ndarray  # (T, N, D)
    noise: np.ndarray  # (T, N, D)
    rigid_id: np.ndarray  # (N,)
    dt: float
    meta: Dict[str, Any]
    valid_targets: np.ndarray  # (T,) bool mask where derivatives are reliable
    normalizers: Dict[str, NormalizationStats]


def _ensure_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got {path}")
    return path


def _parse_meta(meta_arr: np.ndarray) -> Dict[str, Any]:
    if meta_arr.ndim == 0:
        raw = meta_arr.item()
    else:
        raw = meta_arr[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"raw_meta": raw}


def _central_difference(arr: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return derivative and validity mask for central difference along axis 0."""
    deriv = np.zeros_like(arr)
    t_steps = arr.shape[0]
    valid = np.zeros((t_steps,), dtype=bool)
    if t_steps < 2:
        return deriv, valid

    # Forward/backward difference at boundaries
    deriv[0] = (arr[1] - arr[0]) / dt
    deriv[-1] = (arr[-1] - arr[-2]) / dt
    valid[[0, t_steps - 1]] = True

    if t_steps >= 3:
        mid = (arr[2:] - arr[:-2]) / (2.0 * dt)
        deriv[1:-1] = mid
        valid[1:-1] = True

    return deriv, valid


def _prepare_noise(
    shape: Tuple[int, ...],
    scale: ArrayLike,
    rng: Optional[np.random.Generator],
    dtype: np.dtype,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    scale_arr = np.asarray(scale, dtype=dtype)
    if scale_arr.ndim == 0:
        scale_arr = np.full((shape[-1],), float(scale_arr), dtype=dtype)
    else:
        scale_arr = np.broadcast_to(scale_arr, (shape[-1],)).astype(dtype, copy=False)
    noise = rng.normal(loc=0.0, scale=1.0, size=shape).astype(dtype, copy=False)
    broadcast_shape = (1,) * (noise.ndim - 1) + (scale_arr.shape[0],)
    noise *= scale_arr.reshape(broadcast_shape)
    return noise


def _compute_normalizer(
    arr: np.ndarray,
    axis: Tuple[int, ...],
    eps: float = 1e-6,
) -> NormalizationStats:
    mean = arr.mean(axis=axis).astype(np.float32, copy=True)
    std = arr.std(axis=axis).astype(np.float32, copy=True)
    std = np.where(std < eps, 1.0, std)
    return NormalizationStats(mean=mean, std=std)


def _apply_normalizer(arr: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    target_shape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
    mean = stats.mean.reshape(target_shape).astype(np.float32, copy=False)
    std = stats.std.reshape(target_shape).astype(np.float32, copy=False)
    arr = arr.astype(np.float32, copy=True)
    return (arr - mean) / std


def load_gns_scene(
    npz_path: Union[str, Path],
    noise_std: ArrayLike = 0.0,
    noise_clip: Optional[float] = None,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> GNSDataSample:
    """Load and preprocess one npz scene for GNS training/evaluation.

    Parameters
    ----------
    npz_path:
        Path to a compressed numpy archive produced by dataset generation.
    noise_std:
        Standard deviation of Gaussian noise added to positions. Can be a scalar
        or per-dimension sequence. Set to 0.0 to disable noise augmentation.
    noise_clip:
        Optional absolute clip value applied to sampled noise before adding it.
        Useful to avoid extreme perturbations.
    seed:
        Optional random seed for deterministic noise generation.
    normalize:
        Whether to standardize positions, velocities, and accelerations to zero
        mean and unit variance (computed per-scene across particles and time).
        Normalization statistics are returned alongside the sample.
    """

    path = _ensure_path(npz_path)
    with np.load(path) as data:
        positions = data["position"].astype(np.float32, copy=True)
        rigid_id = data["rigid_id"].astype(np.int32, copy=False)
        meta = _parse_meta(data["meta"])

    dt = float(meta.get("dt", 1.0))
    rng = np.random.default_rng(seed)

    velocities, vel_valid = _central_difference(positions, dt)
    accelerations, acc_valid = _central_difference(velocities, dt)

    scale = np.asarray(noise_std, dtype=np.float32)
    use_noise = scale.size > 0 and np.any(scale != 0.0)
    if use_noise:
        noise = _prepare_noise(positions.shape, scale, rng, positions.dtype)
    else:
        noise = np.zeros_like(positions)

    if noise_clip is not None and noise_clip > 0:
        noise = np.clip(noise, -noise_clip, noise_clip)

    noisy_positions = positions + noise

    valid_targets = vel_valid & acc_valid

    stats_axes = (0, 1)
    pos_stats = _compute_normalizer(positions, stats_axes)
    vel_stats = _compute_normalizer(velocities, stats_axes)
    acc_stats = _compute_normalizer(accelerations, stats_axes)

    normalizers = {
        "position": pos_stats,
        "velocity": vel_stats,
        "acceleration": acc_stats,
    }

    if normalize:
        positions = _apply_normalizer(positions, pos_stats)
        velocities = _apply_normalizer(velocities, vel_stats)
        accelerations = _apply_normalizer(accelerations, acc_stats)
        noisy_positions = _apply_normalizer(noisy_positions, pos_stats)
        noise = noisy_positions - positions
    else:
        positions = positions.astype(np.float32, copy=False)
        velocities = velocities.astype(np.float32, copy=False)
        accelerations = accelerations.astype(np.float32, copy=False)
        noisy_positions = noisy_positions.astype(np.float32, copy=False)
        noise = noise.astype(np.float32, copy=False)

    return GNSDataSample(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        noisy_positions=noisy_positions,
        noise=noise,
        rigid_id=rigid_id,
        dt=dt,
        meta=meta,
        valid_targets=valid_targets,
        normalizers=normalizers,
    )


def load_gns_dataset(
    directory: Union[str, Path],
    pattern: str = "scene_*.npz",
    normalize: bool = True,
    **kwargs: Any,
) -> Tuple[GNSDataSample, ...]:
    """Load all npz scenes under the given directory matching a glob pattern."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(dir_path)

    samples = []
    for file_path in sorted(dir_path.glob(pattern)):
        samples.append(load_gns_scene(file_path, normalize=normalize, **kwargs))

    return tuple(samples)
