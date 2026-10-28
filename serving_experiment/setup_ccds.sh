#!/bin/bash
# Run on the CCDS login host; GPU inference belongs in the Slurm job.
set -euo pipefail
TASK_ROOT="$HOME/cachewise-rerun"
TASK_CACHE="/local/cachewise-$USER"
UV="$HOME/.local/bin/uv"
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$TASK_ROOT/src" "$TASK_ROOT/runs" "$TASK_CACHE/uv" "$TASK_CACHE/tmp"
export TMPDIR="$TASK_CACHE/tmp" UV_CACHE_DIR="$TASK_CACHE/uv"
export UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_INSTALLS=4
if [ ! -d "$TASK_ROOT/src/vllm-cachewise/.git" ]; then
    git clone --depth 1 --single-branch --branch cachewise \
        https://github.com/chughtapan/vllm.git "$TASK_ROOT/src/vllm-cachewise"
fi
test "$(git -C "$TASK_ROOT/src/vllm-cachewise" rev-parse HEAD)" = \
    a4b3affa7f746db57c2f19de839f72094c8f3ce1
TASK_PATCH="$TASK_SCRIPT_DIR/patches/cachewise-session-lifecycle.patch"
if git -C "$TASK_ROOT/src/vllm-cachewise" apply --reverse --check "$TASK_PATCH" 2>/dev/null; then
    : # The exact patch is already installed.
else
    git -C "$TASK_ROOT/src/vllm-cachewise" apply --check "$TASK_PATCH"
    git -C "$TASK_ROOT/src/vllm-cachewise" apply "$TASK_PATCH"
fi
if [ ! -x "$TASK_ROOT/.venv/bin/python" ]; then
    "$UV" venv --python 3.12 "$TASK_ROOT/.venv"
fi
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=fe042382925000e5adfe530a1cc2b91d7a125fd5 \
    "$UV" pip install --python "$TASK_ROOT/.venv/bin/python" \
    -e "$TASK_ROOT/src/vllm-cachewise" aiohttp scikit-learn pytest
# FlashInfer JIT requires compiler/CRT headers matching PyTorch's CUDA 13.0.
"$UV" pip install --python "$TASK_ROOT/.venv/bin/python" \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
    nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85
"$TASK_ROOT/.venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    revision="aedcc2d42b622764e023cf882b6652e646b95671",
    ignore_patterns=["*.bin", "*.pt"],
))
PY
