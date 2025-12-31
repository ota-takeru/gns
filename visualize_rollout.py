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


def _candidate_rollout_roots() -> list[Path]:
    """rolloutファイルの探索ルート候補を集める"""
    roots: list[Path] = [Path("rollouts"), Path("rollouts_small")]
    for cfg_name in ("config_rollout.yaml", "config_rollout_small.yaml", "config.yaml"):
        cfg_path = Path(cfg_name)
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as fp:
                cfg = yaml.safe_load(fp) or {}
            output_path = cfg.get("output_path")
            if output_path:
                roots.append(Path(output_path))
        except Exception:
            # 読み取り失敗時は無視して次へ
            continue
    # 重複を除去しつつ順序は維持
    seen = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def _guess_latest_rollout() -> Path | None:
    """最新の *_ex*.pkl を探索して返す（見つからなければ None）"""
    candidates: list[tuple[float, Path]] = []
    for root in _candidate_rollout_roots():
        if not root.exists():
            continue
        for p in root.rglob("*_ex*.pkl"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, p.resolve()))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


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


def _save_animation(
    fig: plt.Figure,
    ani: animation.FuncAnimation,
    output_path: Path | None,
    output_format: str,
    fps: int = 20,
):
    """アニメーションを指定フォーマットで保存"""
    suffix_map = {"mp4": ".mp4", "html": ".html", "gif": ".gif"}
    output_format = output_format.lower()
    if output_format not in suffix_map:
        raise ValueError(f"Unsupported output format: {output_format}")

    if output_path is None:
        print("No output path provided. Showing animation window instead.")
        plt.show()
        return

    # 出力先ディレクトリが無い場合は自動で作成する
    output_path.parent.mkdir(parents=True, exist_ok=True)

    desired_suffix = suffix_map[output_format]
    if output_path.suffix.lower() != desired_suffix:
        output_path = output_path.with_suffix(desired_suffix)

    if output_format == "html":
        html = ani.to_jshtml(fps=fps)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(html)
    elif output_format == "gif":
        ani.save(output_path, writer="pillow", fps=fps)
    else:
        ani.save(output_path, writer="ffmpeg", fps=fps)

    print(f"Saved {output_format.upper()} animation to {output_path}")
    plt.close(fig)


def visualize_rollout(
    data: dict,
    output_path: Path | None = None,
    output_format: str = "mp4",
    show_initial: bool = True,
    use_blit: bool = False,
):
    """
    rollout結果を可視化

    Args:
        data: rollout結果の辞書
        output_path: 保存先パス（Noneの場合は表示のみ。main経由では既定で入力ファイル名＋.mp4）
        output_format: 保存フォーマット（"mp4" | "html" | "gif"、既定は"mp4"）
        show_initial: 初期位置を表示するかどうか
    """
    output_format = output_format.lower()
    if output_format not in {"mp4", "html", "gif"}:
        raise ValueError(f"Unsupported output format: {output_format}")

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
            output_format,
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
            output_format,
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
    output_format: str,
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

    _save_animation(fig, ani, output_path, output_format, fps=20)


def visualize_3d(
    positions_pred: np.ndarray,
    positions_gt: np.ndarray,
    particle_types: np.ndarray,
    n_frames: int,
    particle_spacing: float | None,
    marker_scale: float,
    output_path: Path | None,
    output_format: str,
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

    _save_animation(fig, ani, output_path, output_format, fps=20)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GNS rollout predictions vs ground truth"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Path to rollout pickle file (省略時は最新の *_ex*.pkl を自動検出)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for saving animation (e.g., output.html or output.mp4)",
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "html", "gif"],
        default="mp4",
        help="出力フォーマットを指定（既定: mp4）",
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

    if args.input is None:
        pkl_path = _guess_latest_rollout()
        if pkl_path is None:
            raise FileNotFoundError(
                "入力が指定されておらず、*_ex*.pkl を自動検出できませんでした。"
            )
        print(f"No input specified. Using latest rollout: {pkl_path}")
    else:
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
    output_format = args.format

    # `--html` は後方互換のショートカット
    if args.html:
        output_format = "html"

    # 出力パスの拡張子からフォーマットを推測
    if output_path is not None:
        ext = output_path.suffix.lower()
        if ext in {".html", ".htm"}:
            output_format = "html"
        elif ext == ".gif":
            output_format = "gif"
        elif ext == ".mp4":
            output_format = "mp4"

    # ffmpeg が無い環境では mp4 生成に失敗するため自動で HTML に切り替える
    if output_format == "mp4" and not animation.writers.is_available("ffmpeg"):
        print("ffmpeg が見つからないため出力フォーマットを HTML に変更します。")
        output_format = "html"

    suffix_map = {"mp4": ".mp4", "html": ".html", "gif": ".gif"}
    desired_suffix = suffix_map[output_format]

    # 出力パスが指定されていない場合は、入力ファイル名に既定拡張子を付けて保存する
    if output_path is None:
        output_path = Path(pkl_path).with_suffix(desired_suffix)
    # フォーマットと拡張子が食い違う場合は合わせる
    elif output_path.suffix.lower() != desired_suffix:
        print(
            f"Warning: output path suffix {output_path.suffix} "
            f"does not match format {output_format}; using {desired_suffix}."
        )
        output_path = output_path.with_suffix(desired_suffix)

    visualize_rollout(
        data,
        output_path=output_path,
        output_format=output_format,
        show_initial=not args.no_initial,
        use_blit=args.blit,
    )


if __name__ == "__main__":
    main()
