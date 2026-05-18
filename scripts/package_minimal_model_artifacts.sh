#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/IR-RAG-System"
RELEASE_DIR="${1:-$ROOT_DIR/release_artifacts}"
LORA_SRC="${LORA_SRC:-$ROOT_DIR/output/qwen3_8b_lora_ir_rag_round2_clean}"
RERANKER_SRC="${RERANKER_SRC:-$ROOT_DIR/output/rag_retrieval_bge_reranker_v2_m3/model}"
MAKE_TARBALLS="${MAKE_TARBALLS:-1}"

LORA_DST="$RELEASE_DIR/lora_minimal"
RERANKER_DST="$RELEASE_DIR/reranker_minimal"

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required file: $path" >&2
    exit 1
  fi
}

copy_if_exists() {
  local src="$1"
  local dst_dir="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst_dir/"
  fi
}

rm -rf "$LORA_DST" "$RERANKER_DST"
mkdir -p "$LORA_DST" "$RERANKER_DST"

# LoRA minimal inference set: adapter weights + adapter config.
need_file "$LORA_SRC/adapter_model.safetensors"
need_file "$LORA_SRC/adapter_config.json"
cp "$LORA_SRC/adapter_model.safetensors" "$LORA_DST/"
cp "$LORA_SRC/adapter_config.json" "$LORA_DST/"
copy_if_exists "$LORA_SRC/README.md" "$LORA_DST"
copy_if_exists "$LORA_SRC/chat_template.jinja" "$LORA_DST"

# Reranker minimal inference set: model config + weights + tokenizer assets.
need_file "$RERANKER_SRC/config.json"
cp "$RERANKER_SRC/config.json" "$RERANKER_DST/"

if [[ -f "$RERANKER_SRC/model.safetensors" ]]; then
  cp "$RERANKER_SRC/model.safetensors" "$RERANKER_DST/"
elif [[ -f "$RERANKER_SRC/pytorch_model.bin" ]]; then
  cp "$RERANKER_SRC/pytorch_model.bin" "$RERANKER_DST/"
else
  echo "[error] missing reranker weight file under: $RERANKER_SRC" >&2
  exit 1
fi

copy_if_exists "$RERANKER_SRC/tokenizer.json" "$RERANKER_DST"
copy_if_exists "$RERANKER_SRC/tokenizer_config.json" "$RERANKER_DST"
copy_if_exists "$RERANKER_SRC/special_tokens_map.json" "$RERANKER_DST"
copy_if_exists "$RERANKER_SRC/sentencepiece.bpe.model" "$RERANKER_DST"
copy_if_exists "$RERANKER_SRC/vocab.json" "$RERANKER_DST"
copy_if_exists "$RERANKER_SRC/merges.txt" "$RERANKER_DST"
copy_if_exists "$RERANKER_SRC/README.md" "$RERANKER_DST"

cat > "$LORA_DST/MANIFEST.txt" <<EOF
Source: $LORA_SRC
Contents:
- adapter_model.safetensors
- adapter_config.json
- optional README.md
- optional chat_template.jinja
Excluded on purpose:
- optimizer.pt
- scheduler.pt
- rng_state.pth
- trainer_state.json
- checkpoint directories
- training plots and logs
EOF

cat > "$RERANKER_DST/MANIFEST.txt" <<EOF
Source: $RERANKER_SRC
Contents:
- config.json
- final model weight file
- tokenizer assets when present
Excluded on purpose:
- training logs
- intermediate checkpoints
- unrelated base-model files outside final export dir
EOF

if [[ "$MAKE_TARBALLS" == "1" ]]; then
  tar -czf "$RELEASE_DIR/lora_minimal.tar.gz" -C "$RELEASE_DIR" lora_minimal
  tar -czf "$RELEASE_DIR/reranker_minimal.tar.gz" -C "$RELEASE_DIR" reranker_minimal
fi

echo "[done] minimal artifacts prepared under: $RELEASE_DIR"
echo "[lora] $LORA_DST"
echo "[reranker] $RERANKER_DST"
if [[ "$MAKE_TARBALLS" == "1" ]]; then
  echo "[archive] $RELEASE_DIR/lora_minimal.tar.gz"
  echo "[archive] $RELEASE_DIR/reranker_minimal.tar.gz"
fi
