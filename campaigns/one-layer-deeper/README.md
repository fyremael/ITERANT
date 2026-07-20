# GCL One Layer Deeper Campaign

A governed architecture-and-optimizer campaign for the [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper) competition.

The campaign tests whether compact weight-tied operator models can learn transferable function composition under a fixed H100 wall-clock budget. It provides:

- a pinned official evaluator contract;
- a generator for legal, self-contained `submission.py` files;
- official-style AdamW reproduction, tied Transformer, and recurrent neural-tape profiles;
- stochastic unroll and evaluation over-unrolling profiles;
- gated residual, normalized-state, and loop-consistency controls;
- AdamW, hybrid Muon, and groupwise adaptive-beta optimizer modes;
- an H100 wall-clock and peak-memory profiler;
- a phase-ordered public-tier run matrix;
- a fail-closed gate for hosted Hard submissions.

## Authority and boundaries

The governed upstream revision is recorded in `UPSTREAM_PIN.env`. Results produced against another evaluator revision are not interchangeable. The campaign does not inspect evaluator-owned data, implement a task-specific solver, control backward, alter manifests for official claims, or exploit the Hard metric.

Generated submissions use only the public `benchmark` API and pinned evaluator dependencies. The source generator rejects outputs above the 256 KiB competition limit.

## Bootstrap

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
./scripts/bootstrap_upstream.sh
python -m unittest discover -s tests -v
```

The official evaluator currently requires Python 3.13.5 and Torch 2.12.1. Campaign tooling supports Python 3.11+, but official H100 runs must use the upstream environment.

## Generate a submission

```bash
old-campaign generate \
  --profile profiles/tied_transformer_stable.json \
  --template templates/submission.py.tmpl \
  --output artifacts/submissions/tied_transformer_stable/submission.py

python .upstream/one-layer-deeper/client/cli.py validate \
  artifacts/submissions/tied_transformer_stable/submission.py
```

## Run an H100 profile

Expose exactly one H100 and install the campaign package inside the pinned evaluator environment. Then:

```bash
./scripts/run_profile.sh baseline_adamw easy e1 74
./scripts/run_profile.sh tied_transformer_adamw easy e1 74
```

Each profile record contains the evaluator `RESULT_JSON`, completed updates, measured examples per second, peak visible GPU memory, environment identity, stdout, and stderr.

## Phase 1 execution

See [`PHASE1.md`](PHASE1.md). Official-faithful E1/M1/M5 runs require one visible H100. Resource-adapted seed sweeps are labeled separately and are not admissible for the Hard gate.

## Campaign sequence

1. **Reproduction:** establish official AdamW throughput, memory, and score variance.
2. **Recurrence:** compare tied Transformer and neural tape at the same evaluator time allowance.
3. **Extrapolation:** sample training unroll counts and test evaluation over-unrolling.
4. **Stability:** add only gated residual scales, normalized recurrent state, and loop-consistency regularization.
5. **Optimizer:** compare AdamW, hybrid Muon, and groupwise adaptive betas on a fixed recurrent architecture.
6. **Hard governance:** require M1–M5 evidence, multi-seed replication, throughput, memory, held-out-depth, and dynamical checks before asking the Human Steward to authorize one hosted Hard attempt.

See `SPEC.md`, `RUN_MATRIX.md`, `METRICS.md`, and `governance/HARD_GATE.md`.

## Current status

The campaign scaffold and CPU contract tests are implemented. No H100 or hosted Hard result is claimed in this repository until corresponding evidence files are committed.
