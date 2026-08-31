from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_rlvr.o10_t4 import canonical_json_hash, validate_o10_t4_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the governed O10-T4 manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_o10_t4_manifest(payload)
    expected = canonical_json_hash(payload["recipe"])
    if payload.get("fixed_recipe_hash") != expected:
        raise ValueError(
            f"fixed_recipe_hash mismatch: declared {payload.get('fixed_recipe_hash')}, computed {expected}"
        )
    print(f"O10-T4 manifest contract: PASS {canonical_json_hash(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
