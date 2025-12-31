# 推論（Inference）ガイド

このガイドでは、学習済みモデルを使用して推論を実行し、結果を可視化する方法を説明します。

## 概要

学習済みの GNS モデルを使用して、物理シミュレーションの推論（rollout）を実行し、予測結果と真値を比較することができます。

## ファイル構成

- `config_rollout.yaml`: 推論用の設定ファイル
- `visualize_rollout.py`: rollout 結果を可視化するスクリプト
- `analyze_rollouts.py`: rollout 結果を分析するスクリプト
- `rollouts/`: 推論結果が保存されるディレクトリ

## 使用方法

### 1. 推論の実行

学習済みモデルを使用して推論を実行します：

```bash
uv run python src/train.py --config config_rollout.yaml
```

このコマンドは、`models/`ディレクトリから最新のモデルを自動的に読み込み、テストデータに対して推論を実行します。

**設定ファイルのポイント:**

- `mode: rollout`: rollout モードで実行
- `model_file: latest`: 最新のモデルを使用
- `scenario`: 使用するデータセット（例: `rigid` / `fluid`）。未指定の場合は `data_path` を参照
- `data_path`: 既定のデータセット場所（シナリオがパスを上書きする場合もあります）
- `output_path`: 結果の保存先

`scenario_options` を設定することで、新しい推論ケース（例: 別の流体ケース）を追加し、`scenario` の値を切り替えるだけで簡単に入力データを交換できます。

### 2. 結果の分析

推論結果の統計情報を表示します：

```bash
uv run python analyze_rollouts.py
```

**出力される情報:**

- タイムステップ数、粒子数、次元
- MSE（平均二乗誤差）
- 平均距離誤差
- タイムステップごとの誤差推移

**サンプル出力:**

```
================================================================================
Rollout結果の分析
================================================================================

分析中: rollout_ex0.pkl
  - タイムステップ数: 234
  - 粒子数: 64
  - 次元: 2D
  - MSE (total): 296.048737
  - 平均距離誤差: 18.867802
  - ファイル記録ロス: 296.048767

================================================================================
全体統計
================================================================================
  - 総rollout数: 3
  - 平均MSE: 311.423431
  - 平均距離誤差: 19.332863
```

### 3. 結果の可視化

推論結果をアニメーションとして可視化します：

```bash
# 単一のrolloutを可視化（既定は MP4 出力: rollouts/rollout_ex0.mp4）
uv run python visualize_rollout.py rollouts/rollout_ex0.pkl

# HTML 形式で保存したい場合
uv run python visualize_rollout.py rollouts/rollout_ex0.pkl --format html

# すべてのrolloutを可視化
for i in 0 1 2; do
    uv run python visualize_rollout.py rollouts/rollout_ex${i}.pkl
done
```

**可視化オプション:**

- `--output PATH`: 出力ファイルのパス（拡張子に合わせて mp4/html/gif を自動選択）
- `--format {mp4,html,gif}`: 出力フォーマットを明示（既定: mp4）。`--html` は `--format html` のエイリアス
- `--no-initial`: 初期位置を表示しない
- `--blit`: blitting を使用して描画を高速化（環境によっては表示崩れの可能性あり）

推論時に生成するシーン数を制限したい場合は、`config_rollout.yaml` に以下を設定します。

```
rollout_inference_max_examples: 3  # 例: 先頭3シーンのみ生成
```
この設定により `rollouts/rollout_ex{i}.pkl` の生成数が最大 N 件に制限され、可視化対象もその範囲になります。

**可視化の特徴:**

- 左側：モデルの予測結果
- 右側：真値（Ground Truth）
- 粒子タイプごとに色分け表示
- フレームごとの時間推移を表示

### 4. 結果の確認

生成された MP4 ファイルは任意の動画プレーヤーで再生できます。HTML で保存した場合はブラウザで開いて確認します：

```bash
# WSL2の場合
explorer.exe rollouts/rollout_ex0.html

# Linuxの場合
xdg-open rollouts/rollout_ex0.html
```

## 推論結果の解釈

### MSE（平均二乗誤差）

- **低い値（< 100）**: 予測が真値に近く、モデルが良好に学習できている
- **中程度（100-500）**: 実用的な精度だが、さらなる学習が必要
- **高い値（> 500）**: 予測精度が低く、モデルの改善が必要

### 誤差の時間推移

```
Step      MSE            距離誤差
----------------------------------------
0         0.000003       0.002387
23        0.266645       0.728790
46        3.796991       2.749992
...
230       1201.236938    48.840919
```

- 時間が経過するにつれて誤差が累積していくのは正常です
- 急激な誤差の増加は、モデルが物理法則を正確に捉えられていない可能性があります

## トラブルシューティング

### モデルが見つからない

```
FileNotFoundError: Model does not exist at ...
```

**解決方法:**

1. `models/`ディレクトリにモデルファイルが存在するか確認
2. `config_rollout.yaml`の`model_file`設定を確認

### test.npz が見つからない

```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/out/test.npz'
```

**解決方法:**

```bash
# valid.npzをtest.npzとしてコピー
cp datasets/out/valid.npz datasets/out/test.npz
```

### CUDA メモリ不足

```
RuntimeError: CUDA out of memory
```

**解決方法:**

1. バッチサイズを小さくする（config_rollout.yaml で調整）
2. CPU モードで実行する（`cuda_device_number: null`）

## ディレクトリ構成

```
gns/
├── config_rollout.yaml       # 推論用設定ファイル
├── visualize_rollout.py      # 可視化スクリプト
├── analyze_rollouts.py       # 分析スクリプト
├── models/                   # 学習済みモデル
│   ├── model-0.pt
│   ├── model-50.pt
│   └── model-100.pt
├── datasets/out/             # データセット
│   ├── test.npz
│   ├── valid.npz
│   └── metadata.json
└── rollouts/                 # 推論結果
    ├── rollout_ex0.pkl       # 推論データ
    ├── rollout_ex0.html      # 可視化結果
    ├── rollout_ex1.pkl
    ├── rollout_ex1.html
    └── ...
```

## まとめ

1. **推論実行**: `uv run python src/train.py --config config_rollout.yaml`
2. **分析**: `uv run python analyze_rollouts.py`
3. **可視化**: `uv run python visualize_rollout.py rollouts/rollout_ex0.pkl --output rollouts/rollout_ex0.html --html`
4. **確認**: ブラウザで HTML ファイルを開く

これで、学習済みモデルの性能を視覚的に確認し、改善点を見つけることができます。
