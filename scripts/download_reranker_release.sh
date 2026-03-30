#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/IR-RAG-System"
REPO_SLUG="${REPO_SLUG:-Augustrains/IR-RAG-System}"
RELEASE_TAG="${RELEASE_TAG:-latest}"
ASSET_NAME="${ASSET_NAME:-reranker_minimal.tar.gz}"
TARGET_DIR="${TARGET_DIR:-$ROOT_DIR/models/bge-reranker-v2-m3}"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/.cache/tmp}"

mkdir -p "$TMP_DIR"
ARCHIVE_PATH="$TMP_DIR/$ASSET_NAME"
EXTRACT_DIR="$TMP_DIR/reranker_extract"

if [[ "$RELEASE_TAG" == "latest" ]]; then
  DOWNLOAD_URL="https://github.com/$REPO_SLUG/releases/latest/download/$ASSET_NAME"
else
  DOWNLOAD_URL="https://github.com/$REPO_SLUG/releases/download/$RELEASE_TAG/$ASSET_NAME"
fi

echo "[info] downloading: $DOWNLOAD_URL"
curl -fL "$DOWNLOAD_URL" -o "$ARCHIVE_PATH"

rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"

SRC_DIR="$EXTRACT_DIR/reranker_minimal"
if [[ ! -d "$SRC_DIR" ]]; then
  echo "[error] extracted directory not found: $SRC_DIR" >&2
  exit 1
fi

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
cp -R "$SRC_DIR"/. "$TARGET_DIR"/

echo "[done] reranker downloaded to: $TARGET_DIR"
ls -1 "$TARGET_DIR"
