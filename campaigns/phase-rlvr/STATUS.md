# PHASE-RLVR Campaign Status

Status: **WP00_IMPLEMENTED__NOT_YET_RLVR_EVIDENCED**

## Implemented

- falsifiable programme claim and non-claims;
- five-state operational regime vocabulary;
- safety-first classifier;
- three-window hysteretic estimator with immediate collapse override;
- four audited active recipes plus fail-safe hold;
- continuous `alpha` length-normalization reference contract;
- finite-sample Pass@K reference estimator;
- synthetic unit tests for discovery, sharpening, stall, collapse, verifier failure, hysteresis, and metric contracts;
- staged run matrix and explicit stop conditions.

## Not yet evidenced

- telemetry quality on a real RLVR run;
- entropy-flow implementation against real token updates;
- correct-solution structural clusterer;
- verifier isomorphism suite for the selected task corpus;
- train/inference discrepancy calibration;
- static recipe frontier;
- any adaptive training gain;
- any cross-seed, cross-model, or cross-domain recurrence.

## Immediate next executable gate

**WP01-O10: observer-only veRL integration.**

The controller must be unable to modify training. Its only permitted output is a timestamped telemetry/event log. WP01 is complete only after offline reconstruction demonstrates that candidate regime labels have useful temporal stability and predictive association with subsequent fixed-recipe response.

No claim of controllable RLVR phases is admitted before WP03 recurrence.
