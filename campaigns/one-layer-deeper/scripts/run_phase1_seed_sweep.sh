#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$root/UPSTREAM_PIN.env"
evaluator=${ONE_LAYER_EVALUATOR_ROOT:-"$root/.upstream/one-layer-deeper"}
seeds=(11 22 33)
submission="$root/artifacts/submissions/official_baseline/submission.py"
ONE_LAYER_EVALUATOR_ROOT="$evaluator" \
  bash "$root/scripts/materialize_official_baseline.sh" "$submission"

for item in "easy:e1" "medium:m1" "medium:m5"; do
  tier=${item%%:*}
  dataset=${item##*:}
  official="$evaluator/benchmark/manifests/h100_${tier}_${dataset}.json"
  for seed in "${seeds[@]}"; do
    adapted="$root/artifacts/manifests/h100_${tier}_${dataset}_seed_${seed}.json"
    output="$root/artifacts/runs/official_baseline-${tier}-${dataset}-adapted-s${seed}.json"
    python "$root/scripts/make_seed_manifest.py" --input "$official" --seed "$seed" --output "$adapted"
    python "$root/scripts/profile_h100.py" \
      --evaluator-root "$evaluator" --manifest "$adapted" --submission "$submission" \
      --output "$output" --upstream-commit "$ONE_LAYER_COMMIT" --profile official_baseline \
      --tier "$tier" --dataset "$dataset" --seed "$seed" \
      --classification resource-adapted-seed-sweep
  done
done
