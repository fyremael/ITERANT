from __future__ import annotations

import argparse
import json

from .gate import evaluate_hard_gate
from .generator import generate_submission
from .matrix import expand_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="GCL One Layer Deeper campaign tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--profile", required=True)
    generate.add_argument("--template", required=True)
    generate.add_argument("--output", required=True)

    matrix = sub.add_parser("matrix")
    matrix.add_argument("--config", required=True)

    gate = sub.add_parser("gate")
    gate.add_argument("--policy", required=True)
    gate.add_argument("evidence", nargs="+")

    args = parser.parse_args()
    if args.command == "generate":
        print(generate_submission(args.profile, args.template, args.output))
    elif args.command == "matrix":
        print(json.dumps(expand_runs(args.config), indent=2, sort_keys=True))
    else:
        decision = evaluate_hard_gate(args.evidence, args.policy)
        print(json.dumps({
            "approved": decision.approved,
            "failures": decision.failures,
            "warnings": decision.warnings,
            "summary": decision.summary,
        }, indent=2, sort_keys=True))
        raise SystemExit(0 if decision.approved else 2)


if __name__ == "__main__":
    main()
