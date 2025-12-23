#!/bin/bash
# 推論を実行して結果を可視化するスクリプト

set -e

echo "========================================="
echo "GNS モデル推論スクリプト"
echo "========================================="
echo ""

# 1. 推論を実行
echo "1. 推論を実行中..."
python src/train.py --config config_rollout.yaml

echo ""
echo "2. 結果を分析中..."
python analyze_rollouts.py

echo ""
echo "3. 結果を可視化中..."
find rollouts -type f -name "rollout_ex*.pkl" | sort | while read -r pkl_file; do
    base_name=$(basename "$pkl_file" .pkl)
    html_file="$(dirname "$pkl_file")/${base_name}.html"
    echo "  - $pkl_file -> $html_file"
    python visualize_rollout.py "$pkl_file" --output "$html_file" --html
done

echo ""
echo "========================================="
echo "完了！"
echo "========================================="
echo ""
echo "結果を確認するには、以下のファイルをブラウザで開いてください："
find rollouts -type f -name "rollout_ex*.html" | sort | while read -r html_file; do
    echo "  - $html_file"
done
echo ""
