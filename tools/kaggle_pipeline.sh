#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Kaggle パイプライン実行スクリプト

使い方:
  tools/kaggle_pipeline.sh [options]

オプション:
  --dataset-dir PATH   kaggle datasets version に渡すパス (デフォルト: kaggle/dataset)
  --kernel-dir PATH    kaggle kernels push に渡すパス (デフォルト: kaggle/kernal)
  --kernel-ref REF     カーネル参照 (user/slug)。指定がなければ
                       KAGGLE_KERNEL_REF もしくは KAGGLE_USERNAME + KAGGLE_KERNEL_SLUG を使用
  --interval SEC       ステータス確認間隔秒 (デフォルト: 30)
  --timeout SEC        ステータス待ちタイムアウト秒 (デフォルト: 3600)
  -h, --help           このヘルプを表示

備考:
  - 実行時に未コミットの変更があれば自動で `git add -A && git commit` し、続けて push します。
  - 変更がない場合は既存の HEAD をそのまま push します。
  - Kaggle へ送るファイルは一時ディレクトリに展開してから push するため、リポジトリの .gitignore には影響されません。
EOF
}

DATASET_DIR="kaggle/dataset"
KERNEL_DIR="kaggle/kernal"
KERNEL_REF="${KAGGLE_KERNEL_REF:-}"
INTERVAL=30
TIMEOUT=3600

CODE_DST_SUBDIR="code"
CODE_SRCS=(
  "src"
  "analyze_rollouts.py"
  "visualize_rollout.py"
  "run_inference.sh"
  "config.yaml"
  "config_rollout.yaml"
  "config_rollout_small.yaml"
  "config_small.yaml"
  "config_local.yaml"
  "pyproject.toml"
  "uv.lock"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2;;
    --kernel-dir) KERNEL_DIR="$2"; shift 2;;
    --kernel-ref) KERNEL_REF="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$REPO_ROOT" ]]; then
  echo "Git リポジトリ内で実行してください。" >&2
  exit 1
fi
cd "$REPO_ROOT"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI が見つかりません。事前にインストール・認証してください。" >&2
  exit 1
fi

if [[ ! -d "$KERNEL_DIR" ]]; then
  echo "カーネルディレクトリが存在しません: ${KERNEL_DIR}" >&2
  exit 1
fi

if [[ -z "$KERNEL_REF" ]]; then
  META_PATH="${KERNEL_DIR}/kernel-metadata.json"
  if [[ -f "$META_PATH" ]]; then
    # kernel-metadata.json の id をデフォルト参照として使う
    KERNEL_REF=$(
      python3 - "$META_PATH" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
meta = Path(sys.argv[1])
try:
    data = json.loads(meta.read_text())
    ref = data.get("id", "") if isinstance(data, dict) else ""
    if ref:
        print(ref)
except Exception:
    pass
PY
    ) || true
  fi
fi

if [[ -z "$KERNEL_REF" ]]; then
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KERNEL_SLUG:-}" ]]; then
    KERNEL_REF="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
  else
    echo "カーネル参照が未指定です。--kernel-ref か KAGGLE_KERNEL_REF もしくは KAGGLE_USERNAME + KAGGLE_KERNEL_SLUG を設定してください。" >&2
    exit 1
  fi
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "データセットディレクトリが存在しません: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/dataset-metadata.json" ]]; then
  echo "dataset-metadata.json が ${DATASET_DIR} に存在しません。既存データセットのメタデータを配置してください。" >&2
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Step 1: git add / commit / push
LOG1_TMP=$(mktemp)
{
  echo "[1/6] git add/commit/push を実行します。"
  status_output=$(git status --porcelain)
  if [[ -z "$status_output" ]]; then
    echo "変更なし: 既存の HEAD をプッシュします。"
  else
    echo "変更を検出: 自動コミットを実施します。"
    git add -A
    git commit -m "kaggle pipeline auto-commit ${TIMESTAMP}"
  fi
  git push
} 2>&1 | tee "$LOG1_TMP"

GIT_SHA=$(git rev-parse --short HEAD)
RUN_DIR="${REPO_ROOT}/runs/${TIMESTAMP}_${GIT_SHA}"
OUT_DIR="${RUN_DIR}/output"
TMP_KERNEL_DIR="${RUN_DIR}/kernel_payload"
CODE_DST="${DATASET_DIR}/${CODE_DST_SUBDIR}"

mkdir -p "$RUN_DIR"

LOG1="${RUN_DIR}/01_git_push.log"
LOG2="${RUN_DIR}/02_sync_code.log"
LOG3="${RUN_DIR}/03_dataset_version.log"
LOG4="${RUN_DIR}/04_kernel_push.log"
LOG5="${RUN_DIR}/05_kernel_wait.log"
LOG6="${RUN_DIR}/06_kernel_output.log"

mv "$LOG1_TMP" "$LOG1"

echo "Run ID: ${TIMESTAMP}_${GIT_SHA}"
echo "Logs:   ${RUN_DIR}"
echo "Output: ${OUT_DIR}"
echo "Code:   ${CODE_DST}"
echo "Dataset: ${DATASET_DIR}"

# Step 2: sync code into dataset (uncompressed)
: > "$LOG2"
echo "[2/6] コードをデータセットにそのまま配置します (${CODE_DST})." | tee -a "$LOG2"
rm -f "${DATASET_DIR}/code.zip"
rm -rf "$CODE_DST"
mkdir -p "$CODE_DST"
missing=()
for src in "${CODE_SRCS[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${src}" ]]; then
    missing+=("$src")
  fi
done
if (( ${#missing[@]} )); then
  echo "存在しないパスがあります: ${missing[*]}" | tee -a "$LOG2"
  exit 1
fi
rsync -a --delete "${CODE_SRCS[@]/#/${REPO_ROOT}/}" "${CODE_DST}/" 2>&1 | tee -a "$LOG2"

# Step 3: kaggle datasets version
: > "$LOG3"
echo "[3/6] kaggle datasets version を実行します (${DATASET_DIR})." | tee -a "$LOG3"
file_count=$(find "$DATASET_DIR" -type f ! -name 'dataset-metadata.json' | wc -l)
if (( file_count == 0 )); then
  echo "データセットに実ファイルがありません。dataset-metadata.json 以外のファイルを配置してください。" | tee -a "$LOG3"
  exit 1
fi
kaggle datasets version -p "$DATASET_DIR" --dir-mode zip -m "auto ${TIMESTAMP} ${GIT_SHA}" 2>&1 | tee -a "$LOG3"

# Step 4: kaggle kernels push
: > "$LOG4"
echo "[4/6] kaggle kernels push を実行します (${TMP_KERNEL_DIR})." | tee -a "$LOG4"
rm -rf "$TMP_KERNEL_DIR"
mkdir -p "$TMP_KERNEL_DIR"
rsync -a --delete "${KERNEL_DIR}/" "${TMP_KERNEL_DIR}/" 2>&1 | tee -a "$LOG4"
kaggle kernels push -p "$TMP_KERNEL_DIR" 2>&1 | tee -a "$LOG4"

# Step 5: wait for kernel completion
: > "$LOG5"
echo "[5/6] カーネル完了待ちを開始します (${KERNEL_REF}). interval=${INTERVAL}s timeout=${TIMEOUT}s" | tee -a "$LOG5"
start_ts=$(date +%s)
while true; do
  # kaggle CLI が 403/500 を返しても原因を出したいので、一時的に pipefail を外して exit コードを捕捉する
  set +o pipefail
  status_output=$(kaggle kernels status "$KERNEL_REF" 2>&1 | tee -a "$LOG5")
  status_rc=${PIPESTATUS[0]}
  set -o pipefail

  if (( status_rc != 0 )); then
    if echo "$status_output" | grep -qi "403"; then
      echo "ステータス取得で 403 Forbidden。kernel-metadata.json の id が正しいか、カーネルが公開/自分のものか確認してください (期待値例: ${KERNEL_REF})." | tee -a "$LOG5"
    else
      echo "ステータス取得に失敗しました (exit=${status_rc})。" | tee -a "$LOG5"
    fi
    exit 1
  fi

  if echo "$status_output" | grep -qiE "complete|success"; then
    echo "ステータス: 完了を検知しました。" | tee -a "$LOG5"
    break
  fi
  if echo "$status_output" | grep -qiE "error|fail|cancel"; then
    echo "ステータス: 失敗を検知しました。" | tee -a "$LOG5"
    exit 1
  fi
  now_ts=$(date +%s)
  if (( now_ts - start_ts > TIMEOUT )); then
    echo "タイムアウト: ${TIMEOUT} 秒待機しましたが完了しませんでした。" | tee -a "$LOG5"
    exit 1
  fi
  sleep "$INTERVAL"
done

# Step 6: download kernel output
mkdir -p "$OUT_DIR"

: > "$LOG6"
echo "[6/6] kaggle kernels output を実行します (${OUT_DIR})." | tee -a "$LOG6"
kaggle kernels output "$KERNEL_REF" -p "$OUT_DIR" 2>&1 | tee -a "$LOG6"

# Kaggle の出力ログは run.log / output.log / stdout.txt など環境で異なるので柔軟に判定
log_file=""
for candidate in run.log output.log stdout.txt; do
  if [[ -f "${OUT_DIR}/${candidate}" ]]; then
    log_file="$candidate"
    break
  fi
done

if [[ -z "$log_file" ]]; then
  echo "ログファイル(run.log / output.log / stdout.txt)が出力に存在しません。失敗扱いとします。" | tee -a "$LOG6"
  ls -lah "$OUT_DIR" | tee -a "$LOG6"
  exit 1
fi

echo "ログファイルを検出: ${log_file}" | tee -a "$LOG6"
echo "✅ 完了: 出力=${OUT_DIR} ログ=${RUN_DIR}"
