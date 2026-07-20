#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$root/UPSTREAM_PIN.env"
evaluator=${ONE_LAYER_EVALUATOR_ROOT:-"$root/.upstream/one-layer-deeper"}
seeds=(11 22 33)
for item in "easy:e1" "medium:m1" "medium:m5"; do
  tier=${item%%:*}
  dataset=${item##*:}
  official="$evaluator/benchmark/manifests/h100_${tier}_${dataset}.json"
  for seed in "${seeds[@]}"; do
    adapted="$root/artifacts/manifests/h100_${tier}_${dataset}_seed_${seed}.json"
    submission="$root/artifacts/submissions/baseline_adamw/submission.py"
    output="$root/artifacts/runs/baseline_adamw-${tier}-${dataset}-adapted-s${seed}.json"
    python "$root/scripts/make_seed_manifest.py" --input "$official" --seed "$seed" --output "$adapted"
    old-campaign generate --profile "$root/profiles/baseline_adamw.json" --template "$root/templates/submission.py.tmpl" --output "$submission"
    python "$root/scripts/profile_h100.py" \
      --evaluator-root "$evaluator" --manifest "$adapted" --submission "$submission" \
      --output "$output" --upstream-commit "$ONE_LAYER_COMMIT" --profile baseline_adamw \
      --tier "$tier" --dataset "$dataset" --seed "$seed" \
      --classification resource-adapted-seed-sweep
  done
done
