import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from gen_pymunk import DatasetConfig, load_dataset_config


def _compute_marker_size(spacing: float | None, scale: float = 1.0) -> float:
    """Convert particle spacing in data units to scatter size (points^2)."""
    scale = max(0.1, float(scale))
    if spacing is None or spacing <= 0:
        return 16.0 * scale**2  # sensible default scaled
    # Convert spacing (world units) to an on-screen radius.
    radius = max(1.5, spacing * 45.0)
    radius *= scale
    return min(72.0, radius**2)


def _resolve_particle_spacing(meta: dict | None, cfg: DatasetConfig | None = None) -> float | None:
    if meta is not None:
        spacing = meta.get("particle_spacing")
        if isinstance(spacing, (int, float)):
            return float(spacing)
        density = meta.get("particle_density")
        if isinstance(density, (int, float)) and density > 0:
            return 1.0 / math.sqrt(float(density))

    if cfg is None:
        cfg, _ = load_dataset_config()
    if cfg.particle_density > 0:
        return 1.0 / math.sqrt(cfg.particle_density)
    return None


def _resolve_marker_scale(meta: dict | None, cfg: DatasetConfig) -> float:
    if meta is not None:
        scale = meta.get("visualization_marker_scale")
        if isinstance(scale, (int, float)) and scale > 0:
            return float(scale)
    if cfg.visualization_marker_scale > 0:
        return float(cfg.visualization_marker_scale)
    return 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate a dataset scene.")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "valid", "test"),
        help="Dataset split directory under the output dir (default: train).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Overrides the dataset output root directory.",
    )
    parser.add_argument(
        "--case",
        help="Optional case subdirectory under the output root (useful for fluid datasets).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="0-based index within the split (default: 0).",
    )
    group.add_argument(
        "--scene-path",
        type=Path,
        help="Path to a specific scene npz file. If relative, interpreted within the split directory.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg, _cfg_path = load_dataset_config()
    output_root = Path(cfg.output_dir)
    if args.output_root is not None:
        output_root = args.output_root

    base_dir = output_root
    if args.case:
        base_dir = base_dir / args.case

    split_dir = base_dir / args.split

    if args.scene_path is not None:
        dataset_path = args.scene_path
        if not dataset_path.is_absolute():
            dataset_path = split_dir / dataset_path
        dataset_path = dataset_path.resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Scene file not found: {dataset_path}")
    else:
        scenes = sorted(split_dir.glob("scene_*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scene npz files found under {split_dir}")
        if args.scene_index < 0 or args.scene_index >= len(scenes):
            raise IndexError(
                f"scene_index {args.scene_index} out of range for {len(scenes)} scene files"
            )
        dataset_path = scenes[args.scene_index]

    dataset_dir = dataset_path.parent
    print(f"Animating {dataset_path}")

    with np.load(dataset_path) as data:
        pos = data["position"]
        rigid_id = data.get("rigid_id")
        particle_type = data.get("particle_type")
        meta_raw = data.get("meta")
        meta = None
        if meta_raw is not None:
            meta = json.loads(str(meta_raw.item()))

    marker_scale = _resolve_marker_scale(meta, cfg)
    marker_size = _compute_marker_size(_resolve_particle_spacing(meta, cfg), marker_scale)

    fig, ax = plt.subplots()
    colors = None
    color_source = None
    if rigid_id is not None:
        color_source = np.asarray(rigid_id)
    elif particle_type is not None:
        color_source = np.asarray(particle_type)

    if color_source is not None and color_source.size > 0:
        if len(np.unique(color_source)) > 1:
            colors = color_source
    scatter_kwargs = dict(s=marker_size, alpha=0.9, linewidths=0)
    if colors is None:
        sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], **scatter_kwargs)
    else:
        sc = ax.scatter(
            pos[0, :, 0],
            pos[0, :, 1],
            c=colors,
            cmap="tab20",
            **scatter_kwargs,
        )

    ax.set_xlim(pos[..., 0].min() - 1, pos[..., 0].max() + 1)
    ax.set_ylim(pos[..., 1].min() - 1, pos[..., 1].max() + 1)
    ax.set_aspect("equal")

    def update(frame: int):
        sc.set_offsets(pos[frame])
        ax.set_title(f"frame={frame}")
        return (sc,)

    ani = animation.FuncAnimation(fig, update, frames=len(pos), interval=30, blit=True)
    backend = matplotlib.get_backend().lower()

    if "agg" in backend:
        # Save as a single self-contained HTML (no external PNGs)
        html_dir = dataset_dir / "animations"
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_dir / f"{dataset_path.stem}.html"
        html = ani.to_jshtml(fps=100)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved animation to {html_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
