#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.local/bin/env"

uv run torchrun \
  --nproc_per_node="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}" \
  main.py
