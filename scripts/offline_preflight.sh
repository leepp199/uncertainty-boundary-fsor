#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
python -m compileall -q boundary_fsor scripts
for config in configs/ls100.yaml configs/ns100.yaml configs/fsc89.yaml; do
  python -c 'from boundary_fsor.config import load_config; import sys; load_config(sys.argv[1])' "$config"
done
python -m unittest discover -s tests -v
if rg -n "https?://|hf_hub_download|download_url" boundary_fsor scripts/*.py configs; then
  echo "network-capable code found in executable paths" >&2
  exit 1
fi
echo "offline preflight passed"
