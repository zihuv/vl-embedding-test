#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

BACKEND="${FGCLIP2_ORT_BACKEND:-auto}"
COREML_CACHE_DIR="${FGCLIP2_ORT_COREML_CACHE_DIR:-$REPO_ROOT/.cache/ort-coreml}"
EXTRA_ORT_ARGS=()

if [[ -n "${FGCLIP2_ORT_PROVIDERS:-}" ]]; then
  PROVIDERS="$FGCLIP2_ORT_PROVIDERS"
  EXTRA_ORT_ARGS+=(--fg-onnx-providers "$PROVIDERS")
  echo "Launching app_compare_clip.py with ORT providers: $PROVIDERS"
else
  echo "Launching app_compare_clip.py with ORT backend profile: $BACKEND"
fi

if python -c "import gradio, numpy, onnxruntime, torch, transformers" >/dev/null 2>&1; then
  exec python app_compare_clip.py \
    --fg-onnx-mode split-text \
    --fg-onnx-backend "$BACKEND" \
    "${EXTRA_ORT_ARGS[@]}" \
    --fg-onnx-coreml-cache-dir "$COREML_CACHE_DIR" \
    "$@"
fi

exec uv run --with onnxruntime python app_compare_clip.py \
  --fg-onnx-mode split-text \
  --fg-onnx-backend "$BACKEND" \
  "${EXTRA_ORT_ARGS[@]}" \
  --fg-onnx-coreml-cache-dir "$COREML_CACHE_DIR" \
  "$@"
