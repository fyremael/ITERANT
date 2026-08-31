# PHASE-RLVR Campaign Status

Status: **WP00_IMPLEMENTED__O10_T4_RUNNER_READY__NOT_YET_RLVR_EVIDENCED**

## Implemented

- falsifiable programme claim and non-claims;
- five-state operational regime vocabulary;
- safety-first classifier;
- three-window hysteretic estimator with immediate collapse override;
- four audited active recipes plus fail-safe hold;
- continuous `alpha` length-normalization reference contract;
- finite-sample Pass@K reference estimator;
- synthetic unit tests for discovery, sharpening, stall, collapse, verifier failure, hysteresis, and metric contracts;
- staged run matrix and explicit stop conditions;
- read-only veRL v0.8.0 validation-dump observer and exact upstream tag pin;
- offline Pass@1/Pass@K/all-fail/correct-output-uniqueness summaries;
- O10-T4 Colab CLI adapter with exact model/data/veRL pins;
- T4/FP16/LoRA bounded execution contract;
- four-segment checkpoint/download/resume protocol;
- host-side evidence archive and digest admission;
- generated rollout/validation token accounting;
- fail-closed prohibition on controller mutation.

## O10-T4 execution disposition

`COLAB_CLI_ELIGIBLE_FOR_PHASE_RLVR_O10__ADAPTER_IMPLEMENTED`

The declared T4 lane uses fixed GRPO on the pinned 0.5B model. It exists to validate telemetry, evidence production, resume behavior, and offline regime reconstruction. It is not a capacity proxy for T2.

A100/H100 resources remain available on a needs basis for larger-capacity experiments after the relevant gate justifies them.

## Not yet evidenced

- telemetry quality on a real RLVR run;
- successful O10-T4 hosted execution;
- entropy-flow implementation against real token updates;
- correct-solution structural clusterer;
- verifier isomorphism suite for the selected task corpus;
- train/inference discrepancy calibration;
- static recipe frontier;
- any adaptive training gain;
- any cross-seed, cross-model, or cross-domain recurrence.

## Immediate next executable gate

**WP01-O10-T4: fixed-GRPO observer-only hosted run.**

The controller has no training-control handle. The remote command is generated only from the governed fixed recipe. Every 10-step segment must be returned to the host with a `GREEN_OBSERVER` receipt, an admitted evidence archive, and a resume archive before the next segment is accepted.

WP01 is complete only after O11/O12 show useful regime-label stability under seed, window, and bootstrap perturbation and prospective association with subsequent fixed-recipe response.

No claim of controllable RLVR phases is admitted before WP03 recurrence.
