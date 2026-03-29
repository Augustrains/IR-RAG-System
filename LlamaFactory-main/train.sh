#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

source /root/autodl-tmp/IR-RAG-System/scripts/cache_env.sh

LLAMA_FACTORY_DIR="/root/autodl-tmp/IR-RAG-System/LlamaFactory-main"
CONFIG_PATH="examples/train_lora/qwen3_ir_rag_lora_sft.yaml"

LOG_DIR="/root/autodl-tmp/IR-RAG-System/logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_$TIMESTAMP.log"

echo "======================================"
echo "Training started at $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "======================================"

cd $LLAMA_FACTORY_DIR

llamafactory-cli train $CONFIG_PATH 2>&1 | tee $LOG_FILE

echo "======================================"
echo "Training finished"
echo "======================================"