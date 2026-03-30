#!/bin/bash
set -euo pipefail

source /root/autodl-tmp/IR-RAG-System/scripts/cache_env.sh

# 下载 Qwen3-Reranker-4B
modelscope download \
    --model Qwen/Qwen3-Reranker-4B \
    --local_dir /root/autodl-tmp/IR-RAG-System/models/Qwen3-Reranker-4B

# 下载 Qwen3-8B
modelscope download \
    --model Qwen/Qwen3-8B \
    --local_dir /root/autodl-tmp/IR-RAG-System/models/Qwen3-8B

# 下载 BGE-M3
huggingface-cli download BAAI/bge-m3 \
  --local-dir /root/autodl-tmp/IR-RAG-System/models/bge-m3 \
  --local-dir-use-symlinks False

# 下载版面检测模型
huggingface-cli download unstructuredio/yolo_x_layout \
  yolox_l0.05.onnx \
  --local-dir /root/autodl-tmp/IR-RAG-System/models/yolox \
  --local-dir-use-symlinks False

echo "[info] 跳过基座 bge-reranker-v2-m3 下载。"
echo "[info] 如需当前仓库对应的微调 reranker，请执行: bash scripts/download_reranker_release.sh"
