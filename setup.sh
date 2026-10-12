#!/usr/bin/env bash
# Sets up LittleBit for a Tesla V100 (Volta, sm_70): Samsung's code at a pinned commit, our Volta patch on top,
# and a Python 3.12 venv. Run on the machine with the GPU. Everything lands in $LB_ROOT (default /mnt/steam/littlebit).
set -euo pipefail
LB_ROOT=${LB_ROOT:-/mnt/steam/littlebit}
UPSTREAM_COMMIT=42d658b0c79f76450b34b6a3547462c7cdc3e1a0   # SamsungLabs/LittleBit main, 2026-10-09
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$LB_ROOT"
cd "$LB_ROOT"
[ -d LittleBit ] || git clone https://github.com/SamsungLabs/LittleBit.git
cd LittleBit
git fetch -q origin
git checkout -q "$UPSTREAM_COMMIT"
git reset -q --hard "$UPSTREAM_COMMIT"
git apply "$HERE/patches/volta-fp16.patch"
echo "LittleBit at $(git rev-parse --short HEAD) + volta-fp16.patch"

cd "$LB_ROOT"
[ -d venv ] || uv venv -q -p 3.12 venv
# torch 2.6 + CUDA 12.4 still ships sm_70 kernels (newer CUDA 13 builds drop Volta).
VIRTUAL_ENV=$LB_ROOT/venv uv pip install -q \
    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "torch==2.6.0+cu124" "numpy==2.2.4" "transformers==4.51.3" "datasets==3.4.1" "lm-eval==0.4.8" \
    "accelerate>=1.6.0" sentencepiece tqdm "pytorch-optimizer>=3.5.0" "GPUtil>=1.4.0" safetensors
"$LB_ROOT/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list())"
