from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_rlvr.o10_t4 import default_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize the governed O10-T4 manifest")
    parser.add_argument("--iterant-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    payload = default_manifest(args.iterant_head)
    if args.run_id:
        payload["run_id"] = args.run_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
