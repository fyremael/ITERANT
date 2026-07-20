from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a labeled resource-adapted seed-sweep manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["name"] = f"{payload['name']}-resource-adapted-seed-{args.seed}"
    payload["runtime"]["seeds"] = [args.seed]
    payload["gcl_classification"] = "resource-adapted-seed-sweep"
    payload["gcl_source_manifest"] = source.name
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
