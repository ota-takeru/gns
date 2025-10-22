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
- `data_path`: テストデータの場所（test.npz または valid.npz）
- `output_path`: 結果の保存先

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
# 単一のrolloutを可視化（HTML形式）
uv run python visualize_rollout.py rollouts/rollout_ex0.pkl --output rollouts/rollout_ex0.html --html

# すべてのrolloutを可視化
for i in 0 1 2; do
    uv run python visualize_rollout.py rollouts/rollout_ex${i}.pkl \
        --output rollouts/rollout_ex${i}.html --html
done
```

**可視化オプション:**

- `--output PATH`: 出力ファイルのパス（.html, .gif, .mp4 など）
- `--html`: HTML 形式で保存（強制）
- `--no-initial`: 初期位置を表示しない

**可視化の特徴:**

- 左側：モデルの予測結果
- 右側：真値（Ground Truth）
- 粒子タイプごとに色分け表示
- フレームごとの時間推移を表示

### 4. 結果の確認

生成された HTML ファイルをブラウザで開いて確認します：

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
