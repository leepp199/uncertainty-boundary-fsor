#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
config="${1:?usage: $0 CONFIG}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
for method in prototype energy global_boundary class_boundary uncertainty_boundary; do
  python scripts/evaluate_fsor.py --config "$config" --method "$method"
done
