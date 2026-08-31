# PHASE-RLVR Acceptance Contract

## Evidence classes

- `CONTRACT`: deterministic/unit evidence only.
- `OBSERVATIONAL`: telemetry from real training with controller disabled.
- `INTERVENTIONAL`: controlled adaptive-vs-baseline experiments.
- `CONFIRMATORY`: recurrence across model scale or domain.

WP00 may produce only `CONTRACT` evidence.

## Gate G0 — Controller contract

Required:

- all unit tests pass from a clean Python environment;
- invalid telemetry fails loudly;
- safety events override progress signals immediately;
- ordinary transitions respect confirmation and dwell rules;
- recipe definitions are versioned and immutable within a run.

## Gate G1 — Measurement adequacy

Required before any adaptive training:

- missing-data rate below declared limit;
- train/inference discrepancy baseline established;
- Pass@K uncertainty reported;
- verifier-invariance suite frozen;
- regime labels stable under predeclared bootstrap/window perturbations;
- telemetry collection overhead measured.

## Gate G2 — Action identification

Required before WP03:

- every adaptive action has been run as a fixed policy;
- dominated or unsafe actions are removed;
- thresholds are frozen using non-test runs;
- fixed-stage baseline is declared before adaptive results are inspected.

## Gate G3 — Adaptive pilot

Promotion requires:

- controller beats the strongest fixed or fixed-stage baseline on compute-normalized primary utility on at least 2/3 seeds;
- no statistically material degradation in high-K coverage, correct-solution diversity, verifier invariance, or unsafe-window rate;
- the gain persists when controller evaluation tokens are charged to the adaptive method;
- no threshold is changed after inspecting test-seed outcomes.

## Gate G4 — Mechanistic support

At least one must hold prospectively:

- phase label predicts which fixed recipe will perform better over the next control interval;
- switching according to the label improves outcome relative to counterfactual delayed or permuted switching;
- bucket-local labels improve response prediction beyond global training step and current Pass@1 alone.

Without G4, an adaptive gain may still be an engineering result but must not be presented as evidence for phase-specific control.

## Gate G5 — Generality

Required for claims beyond the original setting:

- second model scale;
- second evaluation family;
- recurrence of both utility gain and the relevant regime-response interaction.

## Governance boundary

No positive scientific claim is promoted merely because the software is merged or CI is green. Code correctness, observational evidence, intervention evidence, and generality are separate dispositions.
