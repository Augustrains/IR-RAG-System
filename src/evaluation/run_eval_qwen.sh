#!/bin/bash
set -euo pipefail

MODE="${1:-base}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/IR-RAG-System/models/Qwen3-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/autodl-tmp/IR-RAG-System/output/qwen3_8b_lora_ir_rag}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/IR-RAG-System/LlamaFactory-main/data/ir_rag/test_augmented_with_neg_alpaca.jsonl}"
RUNS_DIR="${RUNS_DIR:-/root/autodl-tmp/IR-RAG-System/src/evaluation/runs}"
RUN_NAME="${RUN_NAME:-}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPORT_MAX_SAMPLES="${REPORT_MAX_SAMPLES:-20}"

if [[ "$MODE" != "base" && "$MODE" != "lora" ]]; then
  echo "[error] MODE must be 'base' or 'lora', got: $MODE" >&2
  exit 1
fi

CMD=(
  python3 -m src.evaluation.eval_qwen_qa
  --mode "$MODE"
  --base-model-path "$BASE_MODEL_PATH"
  --dataset-path "$DATASET_PATH"
  --runs-dir "$RUNS_DIR"
  --limit "$LIMIT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --report-max-samples "$REPORT_MAX_SAMPLES"
)

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run-name "$RUN_NAME")
fi

if [[ "$MODE" == "lora" ]]; then
  CMD+=(--adapter-path "$ADAPTER_PATH")
fi

echo ">>> Running Qwen evaluation"
echo ">>> mode=$MODE"
echo ">>> base_model_path=$BASE_MODEL_PATH"
if [[ "$MODE" == "lora" ]]; then
  echo ">>> adapter_path=$ADAPTER_PATH"
fi
echo ">>> dataset_path=$DATASET_PATH"
echo ">>> runs_dir=$RUNS_DIR"

"${CMD[@]}"
