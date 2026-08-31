from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .metrics import pass_at_k_unbiased


@dataclass(frozen=True)
class ValidationSample:
    step: int
    prompt: str
    output: str
    score: float

    @property
    def prompt_id(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ValidationSummary:
    step: int
    prompts: int
    samples: int
    samples_per_prompt_min: int
    samples_per_prompt_max: int
    pass1: float
    passk: float
    k: int
    all_fail_rate: float
    correct_output_uniqueness: float
    mean_correct_outputs_per_solved_prompt: float


def canonicalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def read_validation_jsonl(path: str | Path) -> list[ValidationSample]:
    samples: list[ValidationSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"input", "output", "score", "step"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {sorted(missing)}")
            samples.append(
                ValidationSample(
                    step=int(row["step"]),
                    prompt=str(row["input"]),
                    output=str(row["output"]),
                    score=float(row["score"]),
                )
            )
    return samples


def summarize_step(
    samples: Iterable[ValidationSample],
    *,
    k: int,
    correct_threshold: float = 0.0,
    canonicalizer: Callable[[str], str] = canonicalize_text,
) -> ValidationSummary:
    rows = list(samples)
    if not rows:
        raise ValueError("cannot summarize an empty validation step")
    steps = {row.step for row in rows}
    if len(steps) != 1:
        raise ValueError(f"expected one step, got {sorted(steps)}")
    if k < 1:
        raise ValueError("k must be >= 1")

    grouped: dict[str, list[ValidationSample]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt_id].append(row)

    p1_values: list[float] = []
    pk_values: list[float] = []
    solved_uniqueness: list[float] = []
    correct_counts: list[int] = []
    n_values: list[int] = []
    all_fail = 0

    for prompt_rows in grouped.values():
        n = len(prompt_rows)
        if n < k:
            raise ValueError(f"prompt has only n={n} samples but k={k}")
        correct_rows = [row for row in prompt_rows if row.score > correct_threshold]
        c = len(correct_rows)
        n_values.append(n)
        p1_values.append(c / n)
        pk_values.append(pass_at_k_unbiased(n, c, k))
        if c == 0:
            all_fail += 1
        else:
            canon = {canonicalizer(row.output) for row in correct_rows}
            solved_uniqueness.append(len(canon) / c)
            correct_counts.append(c)

    prompts = len(grouped)
    return ValidationSummary(
        step=next(iter(steps)),
        prompts=prompts,
        samples=len(rows),
        samples_per_prompt_min=min(n_values),
        samples_per_prompt_max=max(n_values),
        pass1=sum(p1_values) / prompts,
        passk=sum(pk_values) / prompts,
        k=k,
        all_fail_rate=all_fail / prompts,
        correct_output_uniqueness=(
            sum(solved_uniqueness) / len(solved_uniqueness) if solved_uniqueness else 0.0
        ),
        mean_correct_outputs_per_solved_prompt=(
            sum(correct_counts) / len(correct_counts) if correct_counts else 0.0
        ),
    )


def summarize_directory(
    validation_dir: str | Path,
    *,
    k: int,
    correct_threshold: float = 0.0,
) -> list[ValidationSummary]:
    root = Path(validation_dir)
    if not root.is_dir():
        raise ValueError(f"validation_dir is not a directory: {root}")
    summaries: list[ValidationSummary] = []
    for path in sorted(root.glob("*.jsonl"), key=lambda p: int(p.stem)):
        summaries.append(
            summarize_step(
                read_validation_jsonl(path),
                k=k,
                correct_threshold=correct_threshold,
            )
        )
    if not summaries:
        raise ValueError(f"no *.jsonl validation dumps found in {root}")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize read-only veRL validation dumps for PHASE-RLVR")
    parser.add_argument("validation_dir")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--correct-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    rows = summarize_directory(
        args.validation_dir,
        k=args.k,
        correct_threshold=args.correct_threshold,
    )
    payload = "\n".join(json.dumps(asdict(row), sort_keys=True) for row in rows) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
