from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REQUIRED = {
    "architecture",
    "optimizer",
    "d_model",
    "num_heads",
    "train_loops_min",
    "train_loops_max",
    "eval_loops",
    "batch_size",
    "max_steps",
    "learning_rate",
    "weight_decay",
    "warmup_fraction",
    "stability_gates",
    "normalize_state",
    "loop_consistency_weight",
    "muon_ns_steps",
}


def load_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = sorted(_REQUIRED - set(profile))
    if missing:
        raise ValueError(f"profile missing required keys: {missing}")
    if profile["architecture"] not in {"baseline", "tied_transformer", "neural_tape"}:
        raise ValueError("unsupported architecture")
    if profile["optimizer"] not in {"adamw", "hybrid_muon", "groupwise_adamw"}:
        raise ValueError("unsupported optimizer")
    if not 1 <= int(profile["train_loops_min"]) <= int(profile["train_loops_max"]):
        raise ValueError("invalid train loop range")
    if int(profile["eval_loops"]) < int(profile["train_loops_min"]):
        raise ValueError("eval_loops must cover the minimum training depth")
    return profile


def render_submission(template: str, profile: dict[str, Any]) -> str:
    replacements = {
        "__ARCHITECTURE__": repr(profile["architecture"]),
        "__OPTIMIZER__": repr(profile["optimizer"]),
        "__D_MODEL__": str(int(profile["d_model"])),
        "__NUM_HEADS__": str(int(profile["num_heads"])),
        "__TRAIN_LOOPS_MIN__": str(int(profile["train_loops_min"])),
        "__TRAIN_LOOPS_MAX__": str(int(profile["train_loops_max"])),
        "__EVAL_LOOPS__": str(int(profile["eval_loops"])),
        "__BATCH_SIZE__": str(int(profile["batch_size"])),
        "__MAX_STEPS__": str(int(profile["max_steps"])),
        "__LEARNING_RATE__": repr(float(profile["learning_rate"])),
        "__WEIGHT_DECAY__": repr(float(profile["weight_decay"])),
        "__WARMUP_FRACTION__": repr(float(profile["warmup_fraction"])),
        "__STABILITY_GATES__": repr(bool(profile["stability_gates"])),
        "__NORMALIZE_STATE__": repr(bool(profile["normalize_state"])),
        "__LOOP_CONSISTENCY_WEIGHT__": repr(float(profile["loop_consistency_weight"])),
        "__MUON_NS_STEPS__": str(int(profile["muon_ns_steps"])),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", template)))
    if unresolved:
        raise ValueError(f"unresolved template markers: {unresolved[:5]}")
    if len(template.encode("utf-8")) > 256 * 1024:
        raise ValueError("rendered submission exceeds 256 KiB")
    compile(template, "submission.py", "exec")
    return template


def generate_submission(
    profile_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    profile = load_profile(profile_path)
    template = Path(template_path).read_text(encoding="utf-8")
    rendered = render_submission(template, profile)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output
