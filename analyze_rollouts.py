#!/usr/bin/env python3
"""
Rollout結果の分析スクリプト

rollouts/フォルダ内のすべてのpklファイルを読み込み、
統計情報を表示します。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def _guess_rollouts_dir(cli_path: str | None) -> Path:
    """
    ロールアウト格納ディレクトリの推定

    優先順位:
      1. CLI 指定 (--rollouts-dir)
      2. 既存の設定ファイルに記載された output_path
      3. 既存のフォルダ (rollouts, rollouts_small)
      4. 既定の "rollouts"
    """
    if cli_path:
        return Path(cli_path)

    import yaml  # 遅延 import で依存のない環境でも動作するようにする

    for cfg_name in ("config_rollout.yaml", "config_rollout_small.yaml", "config.yaml"):
        cfg_path = Path(cfg_name)
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as fp:
                cfg = yaml.safe_load(fp) or {}
            output_path = cfg.get("output_path")
            if output_path:
                return Path(output_path)
        except Exception:
            # 設定読取に失敗しても他の候補を試す
            continue

    for candidate in (Path("rollouts"), Path("rollouts_small")):
        if candidate.exists():
            return candidate

    return Path("rollouts")


def analyze_rollout(pkl_path: Path, dt: float | None = None) -> dict:
    """rollout結果を分析

    Args:
        pkl_path: rollouts/*.pkl
        dt: 速度計算に用いる刻み幅（None で 1.0 とみなす）
    """
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    # dt が無効値ならデフォルト 1.0
    if dt is None or not isinstance(dt, (int, float)) or dt <= 0:
        dt = 1.0

    # データを取得
    predicted_rollout = data["predicted_rollout"]  # (T_pred, B=1, N, D)
    ground_truth_rollout = data["ground_truth_rollout"]  # (T_pred, B=1, N, D)

    # バッチ次元を除去（B=1を想定）
    if predicted_rollout.ndim == 4:
        predicted_rollout = predicted_rollout[:, 0, :, :]
    if ground_truth_rollout.ndim == 4:
        ground_truth_rollout = ground_truth_rollout[:, 0, :, :]

    # 誤差計算
    error = predicted_rollout - ground_truth_rollout  # (T, N, D)
    squared_error = error**2
    mse_per_timestep = squared_error.mean(axis=(1, 2))  # (T,)
    mse_total = squared_error.mean()

    # 距離誤差
    distance_error = np.sqrt((error**2).sum(axis=-1))  # (T, N)
    mean_distance_error_per_timestep = distance_error.mean(axis=1)  # (T,)
    mean_distance_error = distance_error.mean()

    # 速度誤差（位置差分 / dt）
    velocity_pred = np.diff(predicted_rollout, axis=0) / dt  # (T-1, N, D)
    velocity_gt = np.diff(ground_truth_rollout, axis=0) / dt
    velocity_error = velocity_pred - velocity_gt
    velocity_distance_error = np.sqrt((velocity_error**2).sum(axis=-1))  # (T-1, N)
    mean_velocity_error_per_timestep = velocity_distance_error.mean(axis=1)  # (T-1,)
    mean_velocity_error = velocity_distance_error.mean()

    # 加速度誤差（速度差分 / dt）
    acceleration_pred = np.diff(velocity_pred, axis=0) / dt  # (T-2, N, D)
    acceleration_gt = np.diff(velocity_gt, axis=0) / dt
    acceleration_error = acceleration_pred - acceleration_gt
    acceleration_distance_error = np.sqrt((acceleration_error**2).sum(axis=-1))  # (T-2, N)
    mean_acceleration_error_per_timestep = acceleration_distance_error.mean(axis=1)
    mean_acceleration_error = acceleration_distance_error.mean()

    return {
        "file": pkl_path.name,
        "n_timesteps": len(predicted_rollout),
        "n_particles": predicted_rollout.shape[1],
        "dimension": predicted_rollout.shape[2],
        "mse_total": mse_total,
        "mean_distance_error": mean_distance_error,
        "mse_per_timestep": mse_per_timestep,
        "distance_error_per_timestep": mean_distance_error_per_timestep,
        "mean_velocity_error": mean_velocity_error,
        "velocity_error_per_timestep": mean_velocity_error_per_timestep,
        "mean_acceleration_error": mean_acceleration_error,
        "acceleration_error_per_timestep": mean_acceleration_error_per_timestep,
        "loss_from_file": data.get("loss", None),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rollout結果の統計をまとめて表示します"
    )
    parser.add_argument(
        "--rollouts-dir",
        "-r",
        default=None,
        help="rollout結果を格納しているディレクトリ。省略時は設定ファイルの output_path か rollouts を自動検出。",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="ディレクトリが存在しない場合に自動作成しない（既定は自動作成）。",
    )
    args = parser.parse_args()

    rollouts_dir = _guess_rollouts_dir(args.rollouts_dir)
    if not rollouts_dir.exists():
        if args.no_create:
            print(f"Rolloutsディレクトリが見つかりません: {rollouts_dir}")
            return
        rollouts_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rolloutsディレクトリを作成しました: {rollouts_dir}")

    # サブディレクトリ（method/output_filename/など）も含めて検索
    # 出力ファイル名は設定の `output_filename` に依存するため、プレフィックスは固定しない
    pkl_files = sorted(rollouts_dir.rglob("*_ex*.pkl"))
    if not pkl_files:
        print("rolloutファイルが見つかりません")
        return

    print(f"{'='*80}")
    print(f"Rollout結果の分析")
    print(f"{'='*80}\n")

    results = []
    for pkl_file in pkl_files:
        print(f"分析中: {pkl_file.name}")
        result = analyze_rollout(pkl_file)
        results.append(result)

        print(f"  - タイムステップ数: {result['n_timesteps']}")
        print(f"  - 粒子数: {result['n_particles']}")
        print(f"  - 次元: {result['dimension']}D")
        print(f"  - MSE (total): {result['mse_total']:.6f}")
        print(f"  - 平均距離誤差: {result['mean_distance_error']:.6f}")
        print(f"  - 平均速度誤差: {result['mean_velocity_error']:.6f}")
        print(f"  - 平均加速度誤差: {result['mean_acceleration_error']:.6f}")
        if result["loss_from_file"] is not None:
            print(f"  - ファイル記録ロス: {result['loss_from_file']:.6f}")
        print()

    # 全体統計
    print(f"{'='*80}")
    print(f"全体統計")
    print(f"{'='*80}")
    print(f"  - 総rollout数: {len(results)}")
    avg_mse = np.mean([r["mse_total"] for r in results])
    avg_distance_error = np.mean([r["mean_distance_error"] for r in results])
    avg_velocity_error = np.mean([r["mean_velocity_error"] for r in results])
    avg_acceleration_error = np.mean([r["mean_acceleration_error"] for r in results])
    print(f"  - 平均MSE: {avg_mse:.6f}")
    print(f"  - 平均距離誤差: {avg_distance_error:.6f}")
    print(f"  - 平均速度誤差: {avg_velocity_error:.6f}")
    print(f"  - 平均加速度誤差: {avg_acceleration_error:.6f}")

    # タイムステップごとの誤差推移
    print(f"\n{'='*80}")
    print(f"タイムステップごとの誤差推移（最初の例）")
    print(f"{'='*80}")
    if results:
        first = results[0]
        n_steps = len(first["mse_per_timestep"])
        n_steps_vel = len(first["velocity_error_per_timestep"])
        n_steps_acc = len(first["acceleration_error_per_timestep"])
        # 10ステップごとに表示
        step_interval = max(1, n_steps // 10)
        print(f"{'Step':<10}{'MSE':<15}{'距離誤差':<15}{'速度誤差':<15}{'加速度誤差':<15}")
        print(f"{'-'*70}")
        for i in range(0, n_steps, step_interval):
            mse = first["mse_per_timestep"][i]
            dist_err = first["distance_error_per_timestep"][i]
            vel_err = (
                first["velocity_error_per_timestep"][min(i, n_steps_vel - 1)]
                if n_steps_vel
                else float("nan")
            )
            acc_err = (
                first["acceleration_error_per_timestep"][min(i, n_steps_acc - 1)]
                if n_steps_acc
                else float("nan")
            )
            print(f"{i:<10}{mse:<15.6f}{dist_err:<15.6f}{vel_err:<15.6f}{acc_err:<15.6f}")

    print(f"\n{'='*80}")
    print(f"分析完了")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
