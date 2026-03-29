#!/bin/bash

# Keep downloads, model caches, and temp files on the larger autodl-tmp mount.
export IR_RAG_CACHE_ROOT="/root/autodl-tmp/IR-RAG-System/.cache"
export XDG_CACHE_HOME="${IR_RAG_CACHE_ROOT}/xdg"
export HF_HOME="${IR_RAG_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MODELSCOPE_CACHE="${IR_RAG_CACHE_ROOT}/modelscope"
export PIP_CACHE_DIR="${IR_RAG_CACHE_ROOT}/pip"
export TMPDIR="${IR_RAG_CACHE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${MODELSCOPE_CACHE}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"
