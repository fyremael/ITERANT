#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:?usage: run_profile.sh PROFILE TIER DATASET SEED}
tier=${2:?}
dataset=${3:?}
seed=${4:?}
# shellcheck disable=SC1091
source "$root/UPSTREAM_PIN.env"
evaluator=${ONE_LAYER_EVALUATOR_ROOT:-"$root/.upstream/one-layer-deeper"}
submission="$root/artifacts/submissions/$profile/submission.py"
manifest="$evaluator/benchmark/manifests/h100_${tier}_${dataset}.json"
output="$root/artifacts/runs/${profile}-${tier}-${dataset}-s${seed}.json"

if [[ "$profile" == "official_baseline" ]]; then
  ONE_LAYER_EVALUATOR_ROOT="$evaluator" \
    bash "$root/scripts/materialize_official_baseline.sh" "$submission"
else
  profile_json="$root/profiles/$profile.json"
  old-campaign generate \
    --profile "$profile_json" \
    --template "$root/templates/submission.py.tmpl" \
    --output "$submission"
fi

python "$root/scripts/profile_h100.py" \
  --evaluator-root "$evaluator" \
  --manifest "$manifest" \
  --submission "$submission" \
  --output "$output" \
  --upstream-commit "$ONE_LAYER_COMMIT" \
  --profile "$profile" \
  --tier "$tier" \
  --dataset "$dataset" \
  --seed "$seed" \
  --classification official-faithful
