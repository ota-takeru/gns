#!/usr/bin/env python3
"""
Rollout結果の分析スクリプト

rollouts/フォルダ内のすべてのpklファイルを読み込み、
統計情報を表示します。
"""

import pickle
from pathlib import Path

import numpy as np


def analyze_rollout(pkl_path: Path) -> dict:
    """rollout結果を分析"""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

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

    return {
        "file": pkl_path.name,
        "n_timesteps": len(predicted_rollout),
        "n_particles": predicted_rollout.shape[1],
        "dimension": predicted_rollout.shape[2],
        "mse_total": mse_total,
        "mean_distance_error": mean_distance_error,
        "mse_per_timestep": mse_per_timestep,
        "distance_error_per_timestep": mean_distance_error_per_timestep,
        "loss_from_file": data.get("loss", None),
    }


def main():
    rollouts_dir = Path("rollouts")
    if not rollouts_dir.exists():
        print(f"Rolloutsディレクトリが見つかりません: {rollouts_dir}")
        return

    # サブディレクトリ（method/output_filename/など）も含めて検索
    pkl_files = sorted(rollouts_dir.rglob("rollout_ex*.pkl"))
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
    print(f"  - 平均MSE: {avg_mse:.6f}")
    print(f"  - 平均距離誤差: {avg_distance_error:.6f}")

    # タイムステップごとの誤差推移
    print(f"\n{'='*80}")
    print(f"タイムステップごとの誤差推移（最初の例）")
    print(f"{'='*80}")
    if results:
        first = results[0]
        n_steps = len(first["mse_per_timestep"])
        # 10ステップごとに表示
        step_interval = max(1, n_steps // 10)
        print(f"{'Step':<10}{'MSE':<15}{'距離誤差':<15}")
        print(f"{'-'*40}")
        for i in range(0, n_steps, step_interval):
            mse = first["mse_per_timestep"][i]
            dist_err = first["distance_error_per_timestep"][i]
            print(f"{i:<10}{mse:<15.6f}{dist_err:<15.6f}")

    print(f"\n{'='*80}")
    print(f"分析完了")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
