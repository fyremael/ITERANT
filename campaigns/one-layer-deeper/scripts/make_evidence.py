from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _score(profile: dict[str, Any]) -> float:
    return float(profile["result"]["score"]["mean_exact_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one governed candidate-vs-baseline evidence record")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--id-accuracy", type=float, required=True)
    parser.add_argument("--heldout-depth-accuracy", type=float, required=True)
    parser.add_argument("--state-norm-p99", type=float, required=True)
    parser.add_argument("--dynamics-finite", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = _load(args.candidate)
    baseline = _load(args.baseline)
    for key in ("official_upstream_commit", "tier", "dataset", "seed"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"candidate and baseline differ on {key}")
    payload = {
        "schema_version": 1,
        "official_upstream_commit": candidate["official_upstream_commit"],
        "tier": candidate["tier"],
        "dataset": candidate["dataset"],
        "seed": candidate["seed"],
        "candidate_profile": candidate["profile"],
        "baseline_profile": baseline["profile"],
        "candidate_exact_accuracy": _score(candidate),
        "baseline_exact_accuracy": _score(baseline),
        "candidate_examples_per_second": candidate["profiler"]["examples_per_second"],
        "baseline_examples_per_second": baseline["profiler"]["examples_per_second"],
        "peak_memory_gib": candidate["profiler"]["peak_memory_gib"],
        "id_accuracy": args.id_accuracy,
        "heldout_depth_accuracy": args.heldout_depth_accuracy,
        "dynamics": {
            "finite": args.dynamics_finite,
            "state_norm_p99": args.state_norm_p99,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
