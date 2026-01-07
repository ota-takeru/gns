# Codex 利用メモ（gns プロジェクト）

このリポジトリでエージェントが守るべき前提と、よく使うコマンドのまとめ。

## 前提
- 変更は既存ユーザー編集を壊さない範囲で最小限に行う。破壊的 git コマンド（`reset --hard` など）は禁止。
- Python 実行・依存管理は `uv` 優先（README 記載）。必要に応じて `uv run ...` で python を呼ぶ。
- サンドボックス: `workspace-write`。ネットワークは `restricted`（外部通信や Kaggle CLI 利用は事前に確認）。
- 言語は日本語で回答する。

## 主要スクリプト・コマンド
- 依存インストール: `uv sync` （ネット要確認）。
- データ生成
  - 剛体: `uv run python datasets/scripts/gen_pymunk.py`（設定: `datasets/config.yaml`）。
  - 流体: `uv run python datasets/scripts/gen_fluid.py`（設定: `datasets/config/fluid.yaml`）。
- 学習
  - 標準: `uv run python src/train.py --config config.yaml`
  - 小規模/ローカル: `--config config_small.yaml` / `config_local.yaml`
- 推論〜可視化一括: `bash run_inference.sh`
  - 内部で `config_rollout.yaml` を読み、`rollouts/` に pkl / HTML (MP4 既定) を生成。
- 個別実行
  - 推論: `uv run python src/train.py --config config_rollout.yaml`
  - 分析: `uv run python analyze_rollouts.py --rollouts-dir <dir>`（省略時デフォルト）
  - 可視化: `uv run python visualize_rollout.py <pkl> --format {mp4,html,gif} --output <path>`

## Kaggle パイプライン (`tools/kaggle_pipeline.sh`)
- 実行例:  
  `bash tools/kaggle_pipeline.sh [--dataset-dir PATH] [--kernel-dir PATH] [--kernel-ref REF] [--interval SEC] [--timeout SEC]`
- 既定パス: dataset=`kaggle/dataset`, kernel=`kaggle/kernal`。`dataset-metadata.json` の `id` 必須。
- `KAGGLE_KERNEL_REF` または `KAGGLE_USERNAME`+`KAGGLE_KERNEL_SLUG` で参照指定。`kernel-metadata.json` に `id` があれば自動取得。
- 実行フロー:  
  1) `git add/commit/push`（未コミットがあれば自動コミット）  
  2) コードを dataset に同期し `code.zip` を添付  
  3) `kaggle datasets version/create` → ready まで待機  
  4) `kaggle kernels push`  
  5) `kaggle kernels status` で完了待ち（失敗/403/timeout を検出）  
  6) `kaggle kernels output` を `runs/<timestamp>_<sha>/output/` へ保存し不要ファイルを削除  
- ログ: `runs/<timestamp>_<sha>/01_git_push.log` 〜 `06_kernel_output.log`。`.pipeline_version` は差分回避用スタンプ。
- Kaggle CLI インストール・認証とネット接続が前提（要事前確認）。

## よく使う設定ファイル
- 学習: `config.yaml`（シナリオ切替は `scenario`/`scenario_options`）、軽量版 `config_small.yaml`・`config_local.yaml`。
- 推論: `config_rollout.yaml`（出力 `output_path`、モデル `model_file: latest` が既定）。
- 追加の小設定: `config_rollout_small.yaml`（省リソース推論）、`config_rollout.yaml` と併用可。

## ログ・出力場所
- 学習/推論の標準出力は適宜 `runs/` に保存されることが多い。Kaggle パイプラインは `runs/<timestamp>_<sha>/` に集約。
- 推論結果: `rollouts/rollout_ex*.pkl` と可視化 `*.html`/`*.mp4`。

## 作業時の確認ポイント
- Kaggle 用データセットに学習済みモデルを含める場合、`.gitignore` に影響されないようパイプラインが zip へ同梱する仕様を把握する。
- 既存の未コミット変更を自動コミットする挙動を理解し、意図しないコミットを避けたいときは先に確認を取る。
