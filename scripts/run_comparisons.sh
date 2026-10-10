#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
config="${1:?usage: $0 CONFIG [SHOTS]}"
shots="${2:-5}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
for method in prototype energy global_tbg uniform_tbg ces_tbg; do
  python scripts/evaluate_fsor.py --config "$config" --shots "$shots" --method "$method"
done
