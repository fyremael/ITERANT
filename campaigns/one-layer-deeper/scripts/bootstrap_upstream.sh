#!/usr/bin/env bash
set -euo pipefail

campaign_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$campaign_root/UPSTREAM_PIN.env"
target=${1:-"$campaign_root/.upstream/one-layer-deeper"}

if [[ -d "$target/.git" ]]; then
  git -C "$target" fetch --tags origin
else
  mkdir -p "$(dirname "$target")"
  git clone "$ONE_LAYER_REPOSITORY" "$target"
fi

git -C "$target" checkout --detach "$ONE_LAYER_COMMIT"
actual=$(git -C "$target" rev-parse HEAD)
[[ "$actual" == "$ONE_LAYER_COMMIT" ]] || {
  echo "upstream pin mismatch: expected $ONE_LAYER_COMMIT, got $actual" >&2
  exit 2
}
printf 'Pinned evaluator: %s\n' "$actual"
