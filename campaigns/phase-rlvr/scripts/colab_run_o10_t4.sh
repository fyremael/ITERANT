#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CAMPAIGN="$ROOT/campaigns/phase-rlvr"
PYTHONPATH_CAMPAIGN="$CAMPAIGN/src"
GPU="${PHASE_RLVR_COLAB_GPU:-T4}"
COLAB_AUTH="${COLAB_AUTH:-oauth2}"
SEGMENT_TIMEOUT="${PHASE_RLVR_SEGMENT_TIMEOUT:-7200}"
BOOTSTRAP_TIMEOUT="${PHASE_RLVR_BOOTSTRAP_TIMEOUT:-3600}"
RESUME_ARCHIVE="${PHASE_RLVR_RESUME_ARCHIVE:-}"

if [[ "$GPU" != "T4" ]]; then
  echo "O10-T4 runner rejects accelerator '$GPU'; use a separate governed profile for larger hardware." >&2
  exit 2
fi

for cmd in git python colab tar sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "missing required command: $cmd" >&2
    exit 2
  }
done

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "O10-T4 requires a clean worktree so the uploaded payload equals the declared ITERANT head." >&2
  exit 2
fi

ITERANT_HEAD="$(git rev-parse HEAD)"
case "$ITERANT_HEAD" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
    ;;
  *)
    echo "unable to resolve immutable ITERANT head" >&2
    exit 2
    ;;
esac
if [[ ${#ITERANT_HEAD} -ne 40 ]]; then
  echo "ITERANT head is not a full 40-character commit SHA: $ITERANT_HEAD" >&2
  exit 2
fi

STAMP="$(LC_ALL=C date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${PHASE_RLVR_RUN_ID:-PHASE-RLVR-WP01-O10-T4-${STAMP}}"
RUN_ROOT="$ROOT/runs/hosted/$RUN_ID"
mkdir -p "$RUN_ROOT"

MANIFEST="$RUN_ROOT/o10_manifest.json"
PYTHONPATH="$PYTHONPATH_CAMPAIGN" python "$CAMPAIGN/scripts/make_o10_t4_manifest.py" \
  --iterant-head "$ITERANT_HEAD" \
  --run-id "$RUN_ID" \
  --output "$MANIFEST"

PYTHONPATH="$PYTHONPATH_CAMPAIGN" python "$CAMPAIGN/scripts/check_o10_t4_manifest.py" "$MANIFEST"

SOURCE_ARCHIVE="$RUN_ROOT/phase_rlvr_source.tar.gz"
SOURCE_LIST="$RUN_ROOT/source_files.txt"
git ls-files \
  "campaigns/phase-rlvr/**" \
  "README.md" \
  "Makefile" \
  | LC_ALL=C sort > "$SOURCE_LIST"

if [[ ! -s "$SOURCE_LIST" ]]; then
  echo "source file list is empty" >&2
  exit 2
fi

tar \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -czf "$SOURCE_ARCHIVE" \
  -T "$SOURCE_LIST"

SOURCE_SHA256="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$SOURCE_SHA256" "$(basename "$SOURCE_ARCHIVE")" > "$RUN_ROOT/source_sha256.txt"

SESSION="gcl-phase-rlvr-o10-${STAMP,,}-$$"
cleanup() {
  set +e
  colab log -s "$SESSION" -o "$RUN_ROOT/colab-execution.md" >/dev/null 2>&1 || true
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[PHASE-RLVR] allocating session=$SESSION gpu=$GPU"
colab new -s "$SESSION" --gpu "$GPU"
colab status -s "$SESSION" > "$RUN_ROOT/colab-status.txt"

colab upload -s "$SESSION" "$SOURCE_ARCHIVE" /content/phase_rlvr_source.tar.gz
colab upload -s "$SESSION" "$MANIFEST" /content/o10_manifest.json

if [[ -n "$RESUME_ARCHIVE" ]]; then
  test -f "$RESUME_ARCHIVE"
  colab upload -s "$SESSION" "$RESUME_ARCHIVE" /content/o10_resume.tar.gz
fi

echo "[PHASE-RLVR] bootstrapping governed veRL environment"
set +e
colab exec -s "$SESSION" -f "$CAMPAIGN/colab/bootstrap_verl.py" --timeout "$BOOTSTRAP_TIMEOUT"
BOOTSTRAP_RC=$?
set -e

colab download -s "$SESSION" /content/o10/bootstrap_receipt.json "$RUN_ROOT/bootstrap_receipt.json" || true
if [[ $BOOTSTRAP_RC -ne 0 || ! -s "$RUN_ROOT/bootstrap_receipt.json" ]]; then
  echo "O10-T4 bootstrap failed or produced no authoritative receipt" >&2
  exit 3
fi
python - "$RUN_ROOT/bootstrap_receipt.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status")!="GREEN_BOOTSTRAP":
    raise SystemExit(f"bootstrap not green: {payload}")
gpu=payload.get("gpu", {})
if "T4" not in str(gpu.get("name", "")):
    raise SystemExit(f"wrong accelerator: {gpu}")
print("[PHASE-RLVR] bootstrap receipt admitted")
PY

TOTAL_STEPS="$(python - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["recipe"]["total_steps"])
PY
)"
SEGMENT_STEPS="$(python - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["recipe"]["segment_steps"])
PY
)"
SEGMENTS=$((TOTAL_STEPS / SEGMENT_STEPS))

LATEST_RESUME=""
for ((segment=1; segment<=SEGMENTS; segment++)); do
  TARGET=$((segment * SEGMENT_STEPS))
  SEG_DIR="$RUN_ROOT/segment-$(printf '%04d' "$TARGET")"
  mkdir -p "$SEG_DIR"

  echo "[PHASE-RLVR] segment $segment/$SEGMENTS target_step=$TARGET"
  set +e
  colab exec -s "$SESSION" -f "$CAMPAIGN/colab/remote_o10.py" --timeout "$SEGMENT_TIMEOUT"
  EXEC_RC=$?
  set -e

  # Always attempt evidence retrieval. Notebook transport status is not authoritative.
  colab download -s "$SESSION" /content/o10/export/export_receipt.json "$SEG_DIR/export_receipt.json" || true
  colab download -s "$SESSION" /content/o10/export/segment_latest.tar.gz "$SEG_DIR/segment_evidence.tar.gz" || true
  colab download -s "$SESSION" /content/o10/export/resume_latest.tar.gz "$SEG_DIR/resume_checkpoint.tar.gz" || true

  if [[ ! -s "$SEG_DIR/export_receipt.json" || ! -s "$SEG_DIR/segment_evidence.tar.gz" || ! -s "$SEG_DIR/resume_checkpoint.tar.gz" ]]; then
    echo "segment $TARGET failed evidence retrieval; exec_rc=$EXEC_RC" >&2
    exit 4
  fi

  PYTHONPATH="$PYTHONPATH_CAMPAIGN" python "$CAMPAIGN/scripts/check_o10_export.py" \
    --evidence "$SEG_DIR/segment_evidence.tar.gz" \
    --resume "$SEG_DIR/resume_checkpoint.tar.gz" \
    --export-receipt "$SEG_DIR/export_receipt.json" \
    --expected-iterant-head "$ITERANT_HEAD" \
    --expected-source-sha256 "$SOURCE_SHA256"

  LATEST_RESUME="$SEG_DIR/resume_checkpoint.tar.gz"
  cp "$LATEST_RESUME" "$RUN_ROOT/resume_latest.tar.gz"

  if [[ $EXEC_RC -ne 0 ]]; then
    echo "Colab transport returned rc=$EXEC_RC, but authoritative evidence was GREEN; continuing." >&2
  fi
done

colab log -s "$SESSION" -o "$RUN_ROOT/colab-execution.md" || true

python - "$RUN_ROOT" "$ITERANT_HEAD" "$SOURCE_SHA256" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
head=sys.argv[2]
source_sha=sys.argv[3]
segments=sorted(root.glob("segment-*/export_receipt.json"))
if not segments:
    raise SystemExit("no admitted O10-T4 segments")
receipt={
    "schema_version":1,
    "status":"GREEN_O10_T4_HOSTED",
    "iterant_head":head,
    "source_archive_sha256":source_sha,
    "segment_count":len(segments),
    "segments":[json.loads(p.read_text()) for p in segments],
}
(root/"host_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(receipt,sort_keys=True))
PY

echo "[PHASE-RLVR] O10-T4 hosted run GREEN: $RUN_ROOT"
