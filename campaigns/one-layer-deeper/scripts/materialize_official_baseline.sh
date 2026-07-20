#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$root/UPSTREAM_PIN.env"
evaluator=${ONE_LAYER_EVALUATOR_ROOT:-"$root/.upstream/one-layer-deeper"}
output=${1:-"$root/artifacts/submissions/official_baseline/submission.py"}

if [[ ! -d "$evaluator/.git" ]]; then
  echo "Pinned evaluator checkout not found at $evaluator; run scripts/bootstrap_upstream.sh first." >&2
  exit 2
fi

actual_commit=$(git -C "$evaluator" rev-parse HEAD)
if [[ "$actual_commit" != "$ONE_LAYER_COMMIT" ]]; then
  echo "Evaluator checkout mismatch: expected $ONE_LAYER_COMMIT, found $actual_commit" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"
git -C "$evaluator" show \
  "$ONE_LAYER_COMMIT:submissions/baseline_adamw/submission.py" > "$output"

expected_blob=$(git -C "$evaluator" rev-parse \
  "$ONE_LAYER_COMMIT:submissions/baseline_adamw/submission.py")
actual_blob=$(git hash-object "$output")
if [[ "$actual_blob" != "$expected_blob" ]]; then
  echo "Official baseline materialization is not byte-identical to the pinned upstream blob." >&2
  exit 2
fi

printf 'official baseline: %s\nblob: %s\n' "$output" "$actual_blob"
