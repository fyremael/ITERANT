# Phase 1 — Baseline Reproduction and Measurement

## Objective

Establish the official AdamW baseline on E1, M1, and M5 under the pinned evaluator. Record exact accuracy, loss, completed updates, examples per second, wall-clock consumption, optimizer-state elements where observable, software/hardware identity where observable, manifest identity, submission hash, and participant-facing evaluator metrics.

The authoritative baseline artifact is the byte-identical upstream `submissions/baseline_adamw/submission.py` extracted from the governed evaluator commit. The generated `baseline_adamw` profile is an implementation-equivalence candidate, not the baseline identity.

## Evidence classes

Two evidence classes are kept separate.

### Official-faithful

These runs use the unmodified public manifests, evaluator-controlled seed 74, and the byte-identical upstream baseline artifact. They establish the competition-faithful baseline and are admissible for candidate comparisons and the Hard gate.

```bash
./scripts/run_phase1.sh
```

Hosted competition submissions are also official-faithful when the materialized upstream artifact is submitted unchanged and the returned status and metric JSONL are archived.

### Resource-adapted seed sweep

These runs copy the public manifests and change only `runtime.seeds` to 11, 22, and 33. They estimate initialization and sampling sensitivity but are not official competition evidence and may not satisfy the Hard gate.

```bash
./scripts/run_phase1_seed_sweep.sh
```

The distinction is necessary because repeated official runs at the same fixed seed measure hosted runtime/nondeterminism variance, not an independent seed distribution.

## Acceptance checks

1. The evaluator checkout equals the governed commit.
2. Python is 3.13.5 and Torch is 2.12.1 for local official runs.
3. Local H100 runs expose exactly one NVIDIA H100.
4. Each evidence record's declared seed equals both the manifest seed and result seed.
5. Manifest and submission identities are recorded.
6. The official baseline is byte-identical to the governed upstream Git blob.
7. No failed or partial run is promoted as a baseline.
8. Official and resource-adapted records remain visibly classified.
9. Hosted comparisons report completed updates as well as score because fixed-budget throughput varies across accepted runs.

## Initial hosted findings

Three official E1 repetitions at fixed seed 74 completed 284, 232, and 237 updates in the same 60-second allowance, with scores 1.33%, 0.67%, and 0.33%. Their matched-step learning traces were nearly identical. A generated implementation-equivalence candidate completed 243 updates with the same matched-step loss trajectory and a 0.00% score.

The current evidence does not support either a semantic mismatch or a systematic throughput disadvantage for the generated artifact: its update count lies inside the observed official-host range. The upstream file remains authoritative because baseline reproduction requires source identity, while exact-score comparisons require replication and completed-step normalization.

The first official Medium observations are M1 score 0.03% at 48,190 completed updates and M5 score 0.07% at 64,503 completed updates, each using the full 600-second allowance. These are single-run floor estimates pending recurrent-candidate comparisons.

## Current execution status

Hosted official-faithful E1, M1, and M5 observations have been obtained. Detailed local H100 telemetry such as peak VRAM and independent seed sweeps still requires an accessible development H100. No hosted Hard attempt has been consumed.
