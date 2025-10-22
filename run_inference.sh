#!/bin/bash
# 推論を実行して結果を可視化するスクリプト

set -e

echo "========================================="
echo "GNS モデル推論スクリプト"
echo "========================================="
echo ""

# 1. 推論を実行
echo "1. 推論を実行中..."
uv run python src/train.py --config config_rollout.yaml

echo ""
echo "2. 結果を分析中..."
uv run python analyze_rollouts.py

echo ""
echo "3. 結果を可視化中..."
for pkl_file in rollouts/rollout_ex*.pkl; do
    if [ -f "$pkl_file" ]; then
        base_name=$(basename "$pkl_file" .pkl)
        html_file="rollouts/${base_name}.html"
        echo "  - $pkl_file -> $html_file"
        uv run python visualize_rollout.py "$pkl_file" --output "$html_file" --html
    fi
done

echo ""
echo "========================================="
echo "完了！"
echo "========================================="
echo ""
echo "結果を確認するには、以下のファイルをブラウザで開いてください："
for html_file in rollouts/rollout_ex*.html; do
    if [ -f "$html_file" ]; then
        echo "  - $html_file"
    fi
done
echo ""

