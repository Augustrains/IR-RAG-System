#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/IR-RAG-System"
RAG_DIR="$ROOT_DIR/RAG-Retrieval-master"
TRAIN_DIR="$RAG_DIR/rag_retrieval/train/reranker"
ACCELERATE_CONFIG="$ROOT_DIR/scripts/configs/rag_retrieval_single_gpu.yaml"
TRAIN_CONFIG="$ROOT_DIR/scripts/configs/rag_retrieval_bge_reranker_v2_m3.yaml"
LOG_DIR="$ROOT_DIR/logs/rag_retrieval_reranker"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
LATEST_LOG="$LOG_DIR/train.log"

echo "[info] log_file=$LOG_FILE"

cd "$TRAIN_DIR"

accelerate launch   --config_file "$ACCELERATE_CONFIG"   train_reranker.py   --config "$TRAIN_CONFIG"   "$@" 2>&1 | tee "$LOG_FILE"

ln -sfn "$LOG_FILE" "$LATEST_LOG"
