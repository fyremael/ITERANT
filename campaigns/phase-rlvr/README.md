# PHASE-RLVR

**Governed regime detection and constrained scheduling for reinforcement learning with verifiable rewards.**

PHASE-RLVR tests a narrow claim: an RLVR learner can obtain better validated reasoning utility per unit compute when a supervisory controller distinguishes operational training regimes and selects among a small, audited set of training recipes, compared with a fixed recipe or a predeclared stage schedule.

The campaign does **not** assume that RLVR has universal metaphysical "phases". `discovery`, `sharpening`, `stall`, `collapse`, and `uncertain` are operational labels inferred from finite-window telemetry. They may coexist across difficulty buckets.

## Core thesis

For difficulty bucket `b`, let the telemetry window be

\[
x_t^{(b)}=(p_1,p_K,\dot p_1,\dot p_K,q_{fail},H,\Phi_H,D_c,\dot D_c,
\eta_{tok},\dot\eta_{tok},D_{KL},\Delta_{TI},V_{iso}).
\]

A guarded estimator produces a latent operational state

\[
\hat z_t^{(b)}\in\{\text{discovery},\text{sharpening},\text{stall},
\text{collapse},\text{uncertain}\},
\]

and a constrained controller selects an audited recipe. Safety violations override apparent progress and route immediately to recovery.

## Why this campaign exists

Recent RLVR evidence motivates, but does not establish, controllable phases:

- Ring-Zero (arXiv:2607.12395) reports a discovery-then-sharpening pattern at 1T scale and highlights loss aggregation, training/inference correction, and mixed-precision stability.
- OPEFO (arXiv:2605.11491) identifies token-level entropy-flow imbalance as a mechanism of entropy collapse.
- UCPO (arXiv:2605.00365) shows that correctness-only RLVR can improve Pass@1 while collapsing diversity and high-K coverage.
- Pass@k Training (arXiv:2508.10751) derives an efficient advantage construction for Pass@k-aware optimization.
- Pass@k inversion work (arXiv:2607.20543) shows that ordinary RLVR can erase rare correct boundary trajectories.
- ICLR 2026 work on CoT-Pass@K shows that high-K analysis can detect reasoning-boundary changes rather than merely one-shot reliability.

PHASE-RLVR asks whether these observations can be converted into a **safe online scheduling problem**.

## Work packages

### WP00 — Controller contract

No LLM training. Synthetic telemetry trajectories must exactly exercise all regime transitions, safety overrides, hysteresis, recipe routing, Pass@K estimation, and the continuous length-normalization contract. This work package is implemented in `src/phase_rlvr` and `tests`.

### WP01 — Observer-only replay

Attach telemetry collection to a fixed RLVR run without allowing PHASE-RLVR to change training. Reconstruct regime labels offline and test stability under bootstrap resampling, altered window lengths, and held-out seeds.

### WP02 — Static recipe frontier

Measure fixed `discovery`, `sharpening`, and matched fixed-stage schedules under identical token and wall-clock accounting. The adaptive controller is not permitted to compete until its action set has independently measured behavior.

### WP03 — Rule controller

Enable phase-dependent recipe switching at checkpoint boundaries. Compare against the strongest fixed and fixed-stage baselines under matched training-token and evaluation-token budgets.

### WP04 — Bucket-local control

Estimate regimes separately for base-difficulty buckets and permit curriculum reweighting without changing optimizer state within a control interval.

### WP05 — No-regret routing

Only after WP03/WP04 are positive, replace deterministic recipe selection with a constrained contextual-bandit/no-regret router over the same audited action set. No new action is introduced by the learned controller.

### WP06 — Scale confirmation

Replicate promoted findings at a larger model scale and on a second reasoning domain. No universality claim is permitted from a single model family or benchmark.

## Default execution tiers

- **T0:** CPU synthetic contract tests.
- **T1:** systems smoke test with a sub-1B model and small grouped rollouts.
- **T2:** scientific pilot at roughly 1.5B–4B with frontier-stratified mathematical tasks.
- **T3:** confirmatory 7B-scale run only after T2 promotion.

The project prefers the smallest tier capable of falsifying the next claim.

## Repository map

```text
campaigns/phase-rlvr/
  README.md
  SPEC.md
  RUN_MATRIX.md
  METRICS.md
  STATUS.md
  configs/recipes.json
  governance/ACCEPTANCE.md
  src/phase_rlvr/
  tests/
```

## Validation

```bash
make check
```

Passing unit tests proves only the controller contract. It is not RLVR evidence.
