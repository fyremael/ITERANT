from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .state import Phase, Telemetry


@dataclass(frozen=True)
class Thresholds:
    discovery_passk_slope: float = 0.002
    sharpening_pass1_slope: float = 0.002
    flat_passk_abs: float = 0.001
    stall_pass1_abs: float = 0.001
    stall_passk_abs: float = 0.001
    stall_efficiency_abs: float = 0.001
    entropy_flow_floor: float = -0.02
    diversity_slope_floor: float = -0.01
    kl_max: float = 0.15
    train_infer_gap_max: float = 0.05
    verifier_invariance_min: float = 0.95
    verifier_stable_min: float = 0.97
    all_fail_rebalance: float = 0.85


@dataclass(frozen=True)
class Recipe:
    name: str
    alpha: float
    rollout_n: int
    passk_reward_k: int
    context_multiplier: float
    lr_multiplier: float
    kl_loss_coef_multiplier: float
    curriculum: str
    require_engine_resync: bool = False


RECIPES = {
    "discovery": Recipe(
        name="discovery",
        alpha=0.25,
        rollout_n=16,
        passk_reward_k=4,
        context_multiplier=1.25,
        lr_multiplier=1.0,
        kl_loss_coef_multiplier=1.0,
        curriculum="frontier",
    ),
    "sharpening": Recipe(
        name="sharpening",
        alpha=1.0,
        rollout_n=8,
        passk_reward_k=1,
        context_multiplier=0.8,
        lr_multiplier=0.9,
        kl_loss_coef_multiplier=1.0,
        curriculum="consolidate",
    ),
    "recovery": Recipe(
        name="recovery",
        alpha=1.0,
        rollout_n=8,
        passk_reward_k=1,
        context_multiplier=1.0,
        lr_multiplier=0.5,
        kl_loss_coef_multiplier=2.0,
        curriculum="hold",
        require_engine_resync=True,
    ),
    "rebalance": Recipe(
        name="rebalance",
        alpha=0.5,
        rollout_n=16,
        passk_reward_k=4,
        context_multiplier=1.1,
        lr_multiplier=0.75,
        kl_loss_coef_multiplier=1.25,
        curriculum="boundary_rebalance",
    ),
    "hold": Recipe(
        name="hold",
        alpha=1.0,
        rollout_n=8,
        passk_reward_k=1,
        context_multiplier=1.0,
        lr_multiplier=0.0,
        kl_loss_coef_multiplier=1.0,
        curriculum="hold",
    ),
}


def safety_violations(x: Telemetry, t: Thresholds) -> tuple[str, ...]:
    violations: list[str] = []
    if x.entropy_flow < t.entropy_flow_floor:
        violations.append("entropy_flow")
    if x.correct_diversity_slope < t.diversity_slope_floor:
        violations.append("correct_diversity")
    if x.kl > t.kl_max:
        violations.append("kl")
    if x.train_infer_gap > t.train_infer_gap_max:
        violations.append("train_infer_gap")
    if x.verifier_invariance < t.verifier_invariance_min:
        violations.append("verifier_invariance")
    return tuple(violations)


def classify(x: Telemetry, t: Thresholds = Thresholds()) -> Phase:
    x.validate()
    if safety_violations(x, t):
        return Phase.COLLAPSE

    if (
        x.passk_slope > t.discovery_passk_slope
        and x.verifier_invariance >= t.verifier_stable_min
    ):
        return Phase.DISCOVERY

    if (
        abs(x.passk_slope) <= t.flat_passk_abs
        and x.pass1_slope > t.sharpening_pass1_slope
        and x.correct_diversity_slope >= t.diversity_slope_floor
    ):
        return Phase.SHARPENING

    if (
        abs(x.pass1_slope) <= t.stall_pass1_abs
        and abs(x.passk_slope) <= t.stall_passk_abs
        and abs(x.token_efficiency_slope) <= t.stall_efficiency_abs
    ):
        return Phase.STALL

    return Phase.UNCERTAIN


def recipe_for(phase: Phase, x: Telemetry, t: Thresholds = Thresholds()) -> Recipe:
    if phase is Phase.COLLAPSE:
        return RECIPES["recovery"]
    if phase is Phase.DISCOVERY:
        return RECIPES["discovery"]
    if phase is Phase.SHARPENING:
        return RECIPES["sharpening"]
    if phase is Phase.STALL:
        if x.all_fail_rate >= t.all_fail_rebalance:
            return RECIPES["rebalance"]
        return RECIPES["rebalance"]
    return RECIPES["hold"]


@dataclass
class HystereticEstimator:
    thresholds: Thresholds = Thresholds()
    confirmations: int = 3
    min_dwell: int = 2
    current: Phase = Phase.UNCERTAIN
    _candidate: Phase = Phase.UNCERTAIN
    _candidate_count: int = 0
    _dwell_count: int = 0

    def update(self, x: Telemetry) -> Phase:
        raw = classify(x, self.thresholds)

        # Safety overrides are fail-closed and immediate.
        if raw is Phase.COLLAPSE:
            self.current = raw
            self._candidate = raw
            self._candidate_count = self.confirmations
            self._dwell_count = 0
            return self.current

        self._dwell_count += 1
        if raw == self.current:
            self._candidate = raw
            self._candidate_count = 0
            return self.current

        if raw != self._candidate:
            self._candidate = raw
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        if (
            self._candidate_count >= self.confirmations
            and self._dwell_count >= self.min_dwell
        ):
            self.current = raw
            self._dwell_count = 0
            self._candidate_count = 0

        return self.current


def replay(estimator: HystereticEstimator, windows: Iterable[Telemetry]) -> list[Phase]:
    return [estimator.update(window) for window in windows]
