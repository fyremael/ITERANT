from __future__ import annotations

import argparse
import json
from pathlib import Path

VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
REQUIRED = {
    "run_id",
    "iterant_head",
    "verl_commit",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "train_dataset",
    "train_dataset_revision",
    "validation_dataset",
    "validation_dataset_revision",
    "verifier_hash",
    "seed_training",
    "seed_sampling",
    "seed_data",
    "hardware",
    "cuda",
    "torch",
    "rollout_backend",
    "precision_policy",
    "fixed_recipe_hash",
    "validation_data_dir",
    "rollout_data_dir",
    "controller_mutation_enabled",
}


def validate_manifest(payload: dict) -> None:
    missing = REQUIRED - payload.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if payload["verl_commit"] != VERL_COMMIT:
        raise ValueError(f"verl_commit must equal governed pin {VERL_COMMIT}")
    if payload["controller_mutation_enabled"] is not False:
        raise ValueError("O10 requires controller_mutation_enabled=false")
    for key in REQUIRED - {"controller_mutation_enabled"}:
        value = payload[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"required field {key} cannot be empty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PHASE-RLVR WP01-O10 evidence manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(payload)
    print("O10 manifest contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
