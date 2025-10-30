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


def main():
    cfg, _cfg_path = load_dataset_config()
    dataset_dir = Path(cfg.output_dir) / "train"
    scenes = sorted(dataset_dir.glob("scene_*.npz"))
    if not scenes:
        raise FileNotFoundError(f"No scene npz files found under {dataset_dir}")

    dataset_path = scenes[0]
    with np.load(dataset_path) as data:
        pos = data["position"]
        rigid_id = data["rigid_id"]
        meta_raw = data.get("meta")
        meta = None
        if meta_raw is not None:
            meta = json.loads(str(meta_raw.item()))

    marker_scale = _resolve_marker_scale(meta, cfg)
    marker_size = _compute_marker_size(_resolve_particle_spacing(meta, cfg), marker_scale)

    fig, ax = plt.subplots()
    colors = rigid_id if len(np.unique(rigid_id)) > 1 else None
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
