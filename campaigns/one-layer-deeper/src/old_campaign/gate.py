from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    summary: dict[str, float] = field(default_factory=dict)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_hard_gate(
    evidence_paths: Iterable[str | Path],
    policy_path: str | Path,
) -> GateDecision:
    policy = _read_json(policy_path)
    records = [_read_json(path) for path in evidence_paths]
    failures: list[str] = []
    warnings: list[str] = []

    required = set(policy["required_datasets"])
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dataset = record.get("dataset")
        if dataset:
            by_dataset.setdefault(str(dataset), []).append(record)

    missing = sorted(required - set(by_dataset))
    if missing:
        failures.append(f"missing required datasets: {', '.join(missing)}")

    candidate_scores: list[float] = []
    ood_gaps: list[float] = []
    throughput_ratios: list[float] = []
    peak_memory: list[float] = []

    for dataset in sorted(required & set(by_dataset)):
        rows = by_dataset[dataset]
        seeds = {row.get("seed") for row in rows}
        if len(seeds) < int(policy["minimum_seeds_per_dataset"]):
            failures.append(
                f"{dataset}: {len(seeds)} seeds < {policy['minimum_seeds_per_dataset']}"
            )
        for row in rows:
            invalid = False
            for key in (
                "candidate_exact_accuracy",
                "baseline_exact_accuracy",
                "candidate_examples_per_second",
                "baseline_examples_per_second",
                "peak_memory_gib",
                "heldout_depth_accuracy",
                "id_accuracy",
            ):
                if not _finite(row.get(key)):
                    failures.append(f"{dataset}: missing/non-finite {key}")
                    invalid = True
            if invalid:
                continue
            candidate = float(row["candidate_exact_accuracy"])
            baseline = float(row["baseline_exact_accuracy"])
            ratio = float(row["candidate_examples_per_second"]) / max(
                float(row["baseline_examples_per_second"]), 1e-12
            )
            gap = float(row["id_accuracy"]) - float(row["heldout_depth_accuracy"])
            candidate_scores.append(candidate)
            throughput_ratios.append(ratio)
            ood_gaps.append(gap)
            peak_memory.append(float(row["peak_memory_gib"]))
            if candidate < baseline + float(policy["minimum_absolute_gain"]):
                failures.append(
                    f"{dataset}: candidate gain {candidate - baseline:.6f} below minimum"
                )
            if ratio < float(policy["minimum_throughput_ratio"]):
                failures.append(f"{dataset}: throughput ratio {ratio:.3f} below minimum")
            if gap > float(policy["maximum_heldout_depth_gap"]):
                failures.append(f"{dataset}: held-out depth gap {gap:.3f} exceeds maximum")
            if float(row["peak_memory_gib"]) > float(policy["maximum_peak_memory_gib"]):
                failures.append(f"{dataset}: peak memory exceeds policy")
            dynamics = row.get("dynamics", {})
            if not dynamics.get("finite", False):
                failures.append(f"{dataset}: dynamics are not certified finite")
            if float(dynamics.get("state_norm_p99", math.inf)) > float(
                policy["maximum_state_norm_p99"]
            ):
                failures.append(f"{dataset}: state norm p99 exceeds policy")

    if records and any(
        row.get("official_upstream_commit") != policy["upstream_commit"]
        for row in records
    ):
        failures.append("evidence was not produced against the governed upstream commit")

    if not records:
        failures.append("no evidence records supplied")

    summary = {
        "mean_candidate_exact_accuracy": (
            sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0
        ),
        "mean_throughput_ratio": (
            sum(throughput_ratios) / len(throughput_ratios) if throughput_ratios else 0.0
        ),
        "maximum_heldout_depth_gap": max(ood_gaps, default=0.0),
        "maximum_peak_memory_gib": max(peak_memory, default=0.0),
    }
    if len(records) < len(required) * int(policy["minimum_seeds_per_dataset"]):
        warnings.append("evidence count is below the nominal full matrix")
    return GateDecision(not failures, tuple(failures), tuple(warnings), summary)
