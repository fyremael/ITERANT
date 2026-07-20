#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Phase 1 requires one visible H100; nvidia-smi is unavailable." >&2
  exit 3
fi
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#gpu_names[@]} -ne 1 ]] || [[ "${gpu_names[0]}" != *H100* ]]; then
  printf 'Phase 1 requires exactly one visible H100; found: %s\n' "${gpu_names[*]:-none}" >&2
  exit 3
fi
"$root/scripts/run_profile.sh" baseline_adamw easy e1 74
"$root/scripts/run_profile.sh" baseline_adamw medium m1 74
"$root/scripts/run_profile.sh" baseline_adamw medium m5 74
