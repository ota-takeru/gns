#!/usr/bin/env python3
"""
Rollout結果の可視化スクリプト

rollouts/フォルダ内のpklファイルを読み込み、
予測結果と真値を比較してアニメーションまたは静止画として表示します。
"""

import argparse
import math
import pickle
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml


def _compute_marker_size(spacing: float | None, scale: float = 1.0) -> float:
    scale = max(0.1, float(scale))
    if spacing is None or spacing <= 0:
        return 16.0 * scale**2
    radius = max(1.5, spacing * 45.0)
    radius *= scale
    return min(72.0, radius**2)


def _load_viz_params_from_config() -> tuple[float | None, float | None]:
    config_path = Path("datasets/config.yaml")
    if not config_path.exists():
        return None, None
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    density = data.get("particle_density")
    spacing = (
        1.0 / math.sqrt(float(density))
        if isinstance(density, (int, float)) and density > 0
        else None
    )
    scale = data.get("visualization_marker_scale")
    if isinstance(scale, (int, float)) and scale > 0:
        scale = float(scale)
    else:
        scale = None
    return spacing, scale


def _resolve_particle_spacing(metadata: dict | None) -> float | None:
    if isinstance(metadata, dict):
        spacing = metadata.get("particle_spacing")
        if isinstance(spacing, (int, float)) and spacing > 0:
            return float(spacing)
        density = metadata.get("particle_density")
        if isinstance(density, (int, float)) and density > 0:
            return 1.0 / math.sqrt(float(density))
    spacing, _ = _load_viz_params_from_config()
    return spacing


def _resolve_marker_scale(metadata: dict | None) -> float:
    if isinstance(metadata, dict):
        scale = metadata.get("visualization_marker_scale")
        if isinstance(scale, (int, float)) and scale > 0:
            return float(scale)
    _, scale = _load_viz_params_from_config()
    if isinstance(scale, float) and scale > 0:
        return scale
    return 1.0


def load_rollout(pkl_path: Path) -> dict:
    """rollout結果をpklファイルから読み込む"""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data


def visualize_rollout(
    data: dict,
    output_path: Path | None = None,
    save_as_html: bool = False,
    show_initial: bool = True,
    use_blit: bool = False,
):
    """
    rollout結果を可視化

    Args:
        data: rollout結果の辞書
        output_path: 保存先パス（Noneの場合は表示のみ）
        save_as_html: HTMLとして保存するかどうか
        show_initial: 初期位置を表示するかどうか
    """
    # データを取得
    initial_positions = data["initial_positions"]  # (T_init, N, D)
    predicted_rollout = data["predicted_rollout"]  # (T_pred, B=1, N, D)
    ground_truth_rollout = data["ground_truth_rollout"]  # (T_pred, B=1, N, D)
    particle_types = data["particle_types"]  # (N,)

    # バッチ次元を除去（B=1を想定）
    if predicted_rollout.ndim == 4:
        predicted_rollout = predicted_rollout[:, 0, :, :]
    if ground_truth_rollout.ndim == 4:
        ground_truth_rollout = ground_truth_rollout[:, 0, :, :]

    # 初期位置と予測を結合
    if show_initial:
        # 初期位置の最後のフレームから開始
        all_positions_pred = np.concatenate(
            [initial_positions[-1:], predicted_rollout], axis=0
        )
        all_positions_gt = np.concatenate(
            [initial_positions[-1:], ground_truth_rollout], axis=0
        )
    else:
        all_positions_pred = predicted_rollout
        all_positions_gt = ground_truth_rollout

    n_frames = len(all_positions_pred)
    dim = all_positions_pred.shape[-1]
    metadata = data.get("metadata")
    particle_spacing = _resolve_particle_spacing(metadata)
    marker_scale = _resolve_marker_scale(metadata)

    # 2D or 3D
    if dim == 2:
        visualize_2d(
            all_positions_pred,
            all_positions_gt,
            particle_types,
            n_frames,
            particle_spacing,
            marker_scale,
            output_path,
            save_as_html,
            data,
            use_blit,
        )
    elif dim == 3:
        visualize_3d(
            all_positions_pred,
            all_positions_gt,
            particle_types,
            n_frames,
            particle_spacing,
            marker_scale,
            output_path,
            save_as_html,
            data,
        )
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def visualize_2d(
    positions_pred: np.ndarray,
    positions_gt: np.ndarray,
    particle_types: np.ndarray,
    n_frames: int,
    particle_spacing: float | None,
    marker_scale: float,
    output_path: Path | None,
    save_as_html: bool,
    data: dict,
    use_blit: bool = False,
):
    """2Dの可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # カラーマップの設定
    unique_types = np.unique(particle_types)
    colors = particle_types if len(unique_types) > 1 else None
    metadata = data.get("metadata") if isinstance(data, dict) else None
    bounds = None
    if metadata and isinstance(metadata, dict):
        raw_bounds = metadata.get("bounds")
        if raw_bounds is not None:
            bounds_arr = np.asarray(raw_bounds, dtype=float)
            if bounds_arr.ndim == 2 and bounds_arr.shape[0] >= 2:
                bounds = bounds_arr

    # 軸の範囲を設定（余白を広めに）
    all_pos = np.concatenate([positions_pred, positions_gt], axis=0)
    if bounds is not None:
        x_min, x_max = float(bounds[0, 0]), float(bounds[0, 1])
        y_min, y_max = float(bounds[1, 0]), float(bounds[1, 1])
        margin_x = max(0.1, 0.05 * max(1e-6, x_max - x_min))
        margin_y = max(0.1, 0.05 * max(1e-6, y_max - y_min))
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
    else:
        x_min, x_max = all_pos[..., 0].min() - 1, all_pos[..., 0].max() + 1
        y_min, y_max = all_pos[..., 1].min() - 1, all_pos[..., 1].max() + 1

    particle_size = _compute_marker_size(particle_spacing, marker_scale)
    particle_alpha = 0.9
    scatter_kwargs = dict(s=particle_size, alpha=particle_alpha, linewidths=0)

    if colors is None:
        sc1 = ax1.scatter(
            positions_pred[0, :, 0],
            positions_pred[0, :, 1],
            **scatter_kwargs,
        )
        sc2 = ax2.scatter(
            positions_gt[0, :, 0],
            positions_gt[0, :, 1],
            **scatter_kwargs,
        )
    else:
        sc1 = ax1.scatter(
            positions_pred[0, :, 0],
            positions_pred[0, :, 1],
            c=colors,
            cmap="tab20",
            **scatter_kwargs,
        )
        sc2 = ax2.scatter(
            positions_gt[0, :, 0],
            positions_gt[0, :, 1],
            c=colors,
            cmap="tab20",
            **scatter_kwargs,
        )

    def _configure_axis(axis: plt.Axes, title: str):
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("x", fontsize=10)
        axis.set_ylabel("y", fontsize=10)
        axis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        if bounds is not None:
            rect = plt.Rectangle(
                (float(bounds[0, 0]), float(bounds[1, 0])),
                float(bounds[0, 1] - bounds[0, 0]),
                float(bounds[1, 1] - bounds[1, 0]),
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="gray",
                alpha=0.8,
                label="Boundary",
            )
            axis.add_patch(rect)

    _configure_axis(ax1, "Predicted")
    _configure_axis(ax2, "Ground Truth")

    # ロス情報を表示
    if "loss" in data:
        fig.suptitle(f'Rollout Comparison (Mean Loss: {data["loss"]:.4f})', fontsize=14)
    else:
        fig.suptitle("Rollout Comparison", fontsize=14)

    def update(frame):
        sc1.set_offsets(positions_pred[frame])
        sc2.set_offsets(positions_gt[frame])
        ax1.set_title(f"Predicted (frame={frame}/{n_frames-1})")
        ax2.set_title(f"Ground Truth (frame={frame}/{n_frames-1})")
        return sc1, sc2

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=use_blit
    )

    # 保存または表示
    backend = matplotlib.get_backend().lower()
    if save_as_html or "agg" in backend:
        if output_path is None:
            output_path = Path("rollout_visualization.html")
        # Embed frames directly into a single self-contained HTML
        html = ani.to_jshtml(fps=20)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved animation to {output_path}")
        plt.close(fig)
    elif output_path is not None:
        # MP4やGIFとして保存
        if output_path.suffix == ".gif":
            ani.save(output_path, writer="pillow", fps=20)
        else:
            ani.save(output_path, writer="ffmpeg", fps=20)
        print(f"Saved animation to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_3d(
    positions_pred: np.ndarray,
    positions_gt: np.ndarray,
    particle_types: np.ndarray,
    n_frames: int,
    particle_spacing: float | None,
    marker_scale: float,
    output_path: Path | None,
    save_as_html: bool,
    data: dict,
    use_blit: bool = False,
):
    """3Dの可視化"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    # カラーマップの設定
    unique_types = np.unique(particle_types)
    colors = particle_types if len(unique_types) > 1 else None

    # 軸の範囲を設定
    all_pos = np.concatenate([positions_pred, positions_gt], axis=0)
    x_min, x_max = all_pos[..., 0].min() - 1, all_pos[..., 0].max() + 1
    y_min, y_max = all_pos[..., 1].min() - 1, all_pos[..., 1].max() + 1
    z_min, z_max = all_pos[..., 2].min() - 1, all_pos[..., 2].max() + 1

    particle_size = _compute_marker_size(particle_spacing, marker_scale)
    particle_alpha = 0.8
    scatter_kwargs = dict(s=particle_size, alpha=particle_alpha, edgecolors="none", linewidths=0)

    # 初期化
    if colors is None:
        sc1 = ax1.scatter(
            positions_pred[0, :, 0],
            positions_pred[0, :, 1],
            positions_pred[0, :, 2],
            **scatter_kwargs,
        )
        sc2 = ax2.scatter(
            positions_gt[0, :, 0],
            positions_gt[0, :, 1],
            positions_gt[0, :, 2],
            **scatter_kwargs,
        )
    else:
        sc1 = ax1.scatter(
            positions_pred[0, :, 0],
            positions_pred[0, :, 1],
            positions_pred[0, :, 2],
            c=colors,
            cmap="tab20",
            **scatter_kwargs,
        )
        sc2 = ax2.scatter(
            positions_gt[0, :, 0],
            positions_gt[0, :, 1],
            positions_gt[0, :, 2],
            c=colors,
            cmap="tab20",
            **scatter_kwargs,
        )

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_title("Predicted")

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    ax2.set_title("Ground Truth")

    # ロス情報を表示
    if "loss" in data:
        fig.suptitle(f'Rollout Comparison (Mean Loss: {data["loss"]:.4f})', fontsize=14)
    else:
        fig.suptitle("Rollout Comparison", fontsize=14)

    def update(frame):
        sc1._offsets3d = (
            positions_pred[frame, :, 0],
            positions_pred[frame, :, 1],
            positions_pred[frame, :, 2],
        )
        sc2._offsets3d = (
            positions_gt[frame, :, 0],
            positions_gt[frame, :, 1],
            positions_gt[frame, :, 2],
        )
        ax1.set_title(f"Predicted (frame={frame}/{n_frames-1})")
        ax2.set_title(f"Ground Truth (frame={frame}/{n_frames-1})")
        return sc1, sc2

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=use_blit
    )

    # 保存または表示
    backend = matplotlib.get_backend().lower()
    if save_as_html or "agg" in backend:
        if output_path is None:
            output_path = Path("rollout_visualization_3d.html")
        # Embed frames directly into a single self-contained HTML
        html = ani.to_jshtml(fps=20)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved animation to {output_path}")
        plt.close(fig)
    elif output_path is not None:
        # MP4やGIFとして保存
        if output_path.suffix == ".gif":
            ani.save(output_path, writer="pillow", fps=20)
        else:
            ani.save(output_path, writer="ffmpeg", fps=20)
        print(f"Saved animation to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GNS rollout predictions vs ground truth"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to rollout pickle file (e.g., rollouts/rollout_ex0.pkl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for saving animation (e.g., output.html or output.mp4)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Force saving as HTML animation",
    )
    parser.add_argument(
        "--no-initial",
        action="store_true",
        help="Don't show initial positions",
    )
    parser.add_argument(
        "--blit",
        action="store_true",
        help="Use blitting for animation (may break in some HTML viewers).",
    )

    args = parser.parse_args()

    pkl_path = Path(args.input)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Rollout file not found: {pkl_path}")

    print(f"Loading rollout from: {pkl_path}")
    data = load_rollout(pkl_path)

    # メタデータの表示
    if "metadata" in data:
        print(f"Metadata: {data['metadata']}")
    if "loss" in data:
        print(f"Mean Loss: {data['loss']:.6f}")

    output_path = Path(args.output) if args.output else None

    visualize_rollout(
        data,
        output_path=output_path,
        save_as_html=args.html,
        show_initial=not args.no_initial,
        use_blit=args.blit,
    )


if __name__ == "__main__":
    main()
