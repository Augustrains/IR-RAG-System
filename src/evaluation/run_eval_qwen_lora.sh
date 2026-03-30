#!/bin/bash
set -euo pipefail

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/IR-RAG-System/models/Qwen3-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/autodl-tmp/IR-RAG-System/output/qwen3_8b_lora_ir_rag_round2_clean}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/IR-RAG-System/LlamaFactory-main/data/ir_rag/test_augmented_with_neg_alpaca.jsonl}"
RUNS_DIR="${RUNS_DIR:-/root/autodl-tmp/IR-RAG-System/src/evaluation/runs}"
RUN_NAME="${RUN_NAME:-lora_eval}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPORT_MAX_SAMPLES="${REPORT_MAX_SAMPLES:-20}"

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "[error] LoRA adapter path not found: $ADAPTER_PATH" >&2
  exit 1
fi

CMD=(
  python3 -m src.evaluation.eval_qwen_qa
  --mode lora
  --base-model-path "$BASE_MODEL_PATH"
  --adapter-path "$ADAPTER_PATH"
  --dataset-path "$DATASET_PATH"
  --runs-dir "$RUNS_DIR"
  --run-name "$RUN_NAME"
  --limit "$LIMIT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --report-max-samples "$REPORT_MAX_SAMPLES"
)

echo ">>> Running Qwen LoRA evaluation"
echo ">>> base_model_path=$BASE_MODEL_PATH"
echo ">>> adapter_path=$ADAPTER_PATH"
echo ">>> dataset_path=$DATASET_PATH"
echo ">>> runs_dir=$RUNS_DIR"

"${CMD[@]}"
