from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    DISCOVERY = "discovery"
    SHARPENING = "sharpening"
    STALL = "stall"
    COLLAPSE = "collapse"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Telemetry:
    pass1: float
    passk: float
    pass1_slope: float
    passk_slope: float
    all_fail_rate: float
    entropy: float
    entropy_flow: float
    correct_diversity: float
    correct_diversity_slope: float
    token_efficiency: float
    token_efficiency_slope: float
    kl: float
    train_infer_gap: float
    verifier_invariance: float

    def validate(self) -> None:
        for name in ("pass1", "passk", "all_fail_rate", "verifier_invariance"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if self.passk + 1e-12 < self.pass1:
            raise ValueError("passk must be >= pass1 for the same evaluation distribution")
        if self.kl < 0.0:
            raise ValueError("kl must be non-negative")
        if self.train_infer_gap < 0.0:
            raise ValueError("train_infer_gap must be non-negative")
        if self.correct_diversity < 0.0:
            raise ValueError("correct_diversity must be non-negative")
