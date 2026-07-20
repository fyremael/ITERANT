from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_matrix(path: str | Path) -> dict[str, Any]:
    matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    phases = matrix.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("run matrix must define phases")
    ids: set[str] = set()
    for phase in phases:
        phase_id = str(phase.get("id", ""))
        if not phase_id or phase_id in ids:
            raise ValueError("phase ids must be non-empty and unique")
        ids.add(phase_id)
        if not phase.get("runs"):
            raise ValueError(f"phase {phase_id} has no runs")
    return matrix


def expand_runs(path: str | Path) -> list[dict[str, Any]]:
    matrix = load_matrix(path)
    runs: list[dict[str, Any]] = []
    for phase in matrix["phases"]:
        for run in phase["runs"]:
            record = {"phase": phase["id"], **run}
            record["run_id"] = "-".join(
                str(record[key])
                for key in ("phase", "profile", "tier", "dataset", "seed")
            )
            runs.append(record)
    return runs
