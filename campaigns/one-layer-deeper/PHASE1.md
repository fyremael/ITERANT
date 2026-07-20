# Phase 1 — Baseline Reproduction and Measurement

## Objective

Establish the official AdamW baseline on E1, M1, and M5 under the pinned evaluator and one visible H100. Record exact accuracy, loss, completed updates, examples per second, wall-clock consumption, optimizer-state elements, peak visible GPU memory, software/hardware identity, manifest hash, submission hash, and raw evaluator output.

## Evidence classes

Two evidence classes are kept separate.

### Official-faithful

These runs use the unmodified public manifests, including evaluator-controlled seed 74. They establish the competition-faithful baseline and are admissible for candidate comparisons and the Hard gate.

```bash
./scripts/run_phase1.sh
```

### Resource-adapted seed sweep

These runs copy the public manifests and change only `runtime.seeds` to 11, 22, and 33. They estimate initialization and sampling sensitivity but are not official competition evidence and may not satisfy the Hard gate.

```bash
./scripts/run_phase1_seed_sweep.sh
```

The distinction is necessary because repeated official runs at the same fixed seed measure runtime/nondeterminism variance, not an independent seed distribution.

## Acceptance checks

1. The evaluator checkout equals the governed commit.
2. Python is 3.13.5 and Torch is 2.12.1.
3. Exactly one visible NVIDIA H100 is present.
4. Each evidence record's declared seed equals both the manifest seed and result seed.
5. Manifest and generated submission SHA-256 hashes are recorded.
6. No failed or partial run is promoted as a baseline.
7. Official and resource-adapted records remain visibly classified.

## Current execution status

The active assistant execution environment has Python 3.13.5 but no visible NVIDIA device, no `nvidia-smi`, no One Layer Deeper API key, and no outbound Git clone access. It is therefore not an admissible environment for E1/M1/M5 H100 measurement. This is a recorded resource-boundary finding, not a benchmark result.
