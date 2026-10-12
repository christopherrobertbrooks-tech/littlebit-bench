# Common environment for every run: the V100 only (PCI order puts it at index 1 on this host), big disk for caches.
export LB_ROOT=${LB_ROOT:-/mnt/steam/littlebit}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${LB_GPU:-1}
export HF_DATASETS_CACHE=$LB_ROOT/hf-datasets
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
PY=$LB_ROOT/venv/bin/python
UP=$LB_ROOT/LittleBit
RUNS=$LB_ROOT/runs
mkdir -p "$RUNS"
