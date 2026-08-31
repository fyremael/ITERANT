from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from phase_rlvr.o10_t4 import validate_o10_t4_manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_member_json(tf: tarfile.TarFile, name: str) -> dict:
    member = tf.getmember(name)
    fileobj = tf.extractfile(member)
    if fileobj is None:
        raise ValueError(f"missing readable archive member {name}")
    return json.loads(fileobj.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Admit one downloaded O10-T4 Colab segment")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--expected-iterant-head", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args(argv)

    export = json.loads(args.export_receipt.read_text(encoding="utf-8"))
    evidence_sha = sha256_file(args.evidence)
    resume_sha = sha256_file(args.resume)
    if export.get("evidence_sha256") != evidence_sha:
        raise ValueError("evidence archive digest mismatch")
    if export.get("resume_sha256") != resume_sha:
        raise ValueError("resume archive digest mismatch")
    if export.get("segment_status") != "GREEN_OBSERVER":
        raise ValueError(f"segment is not admissible: {export.get('segment_status')}")

    with tarfile.open(args.evidence, "r:gz") as tf:
        names = set(tf.getnames())
        required = {"segment_receipt.json", "o10_manifest.json", "observer_summary.jsonl", "state.json"}
        missing = required - names
        if missing:
            raise ValueError(f"evidence archive missing required members: {sorted(missing)}")
        receipt = read_member_json(tf, "segment_receipt.json")
        manifest = read_member_json(tf, "o10_manifest.json")
        state = read_member_json(tf, "state.json")

    validate_o10_t4_manifest(manifest)

    if receipt.get("status") != "GREEN_OBSERVER":
        raise ValueError("embedded segment receipt is not GREEN_OBSERVER")
    if receipt.get("controller_mutation_enabled") is not False:
        raise ValueError("embedded receipt does not prove controller mutation disabled")
    if manifest.get("controller_mutation_enabled") is not False:
        raise ValueError("embedded manifest enables controller mutation")
    if int(receipt.get("target_step", -1)) != int(state.get("completed_target_step", -2)):
        raise ValueError("segment receipt/state checkpoint disagreement")
    if manifest.get("iterant_head") != args.expected_iterant_head:
        raise ValueError("embedded manifest ITERANT head mismatch")
    if receipt.get("iterant_head") != args.expected_iterant_head:
        raise ValueError("embedded receipt ITERANT head mismatch")
    if receipt.get("source_archive_sha256") != args.expected_source_sha256:
        raise ValueError("embedded source archive digest mismatch")
    if receipt.get("fixed_recipe_hash") != manifest.get("fixed_recipe_hash"):
        raise ValueError("embedded recipe hash disagreement")

    with tarfile.open(args.resume, "r:gz") as tf:
        resume_names = set(tf.getnames())
    if "state.json" not in resume_names:
        raise ValueError("resume archive missing state.json")
    if not any(name.startswith("checkpoints/") for name in resume_names):
        raise ValueError("resume archive contains no checkpoint files")

    print(
        "O10-T4 segment admitted:",
        json.dumps(
            {
                "target_step": receipt["target_step"],
                "evidence_sha256": evidence_sha,
                "resume_sha256": resume_sha,
            },
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
