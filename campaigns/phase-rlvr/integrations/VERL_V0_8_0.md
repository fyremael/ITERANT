# veRL v0.8.0 Observer Integration Contract

Upstream tag: `v0.8.0`

Exact tag commit: `7aed6b230776f963fa09509c10d9c3a767d1102c`

PHASE-RLVR WP01 initially integrates **without modifying veRL training behavior**.

## Existing upstream surfaces used

At v0.8.0, the synchronous PPO/GRPO trainer:

- repeats each validation sample according to `actor_rollout_ref.rollout.val_kwargs.n`;
- evaluates the repeated generations with the configured reward function;
- can dump validation generations as step-indexed JSONL using `trainer.validation_data_dir`;
- writes `input`, `output`, ground truth, score, and step to those dumps;
- logs actor entropy when old log probabilities are recomputed;
- computes rollout/training mismatch diagnostics when rollout log probabilities are present;
- performs validation according to `trainer.test_freq`.

The PHASE-RLVR observer consumes these outputs after they are produced. It has no handle to actor updates, learning rate, curriculum, rollout count, loss aggregation, or optimizer state in WP01.

## Required observer configuration

Conceptually:

```yaml
trainer:
  val_before_train: true
  test_freq: <governed interval>
  validation_data_dir: <immutable run directory>/validation
  rollout_data_dir: <immutable run directory>/rollouts
actor_rollout_ref:
  rollout:
    val_kwargs:
      do_sample: true
      n: <N >= routine K>
```

The exact full veRL config is pinned per run; this excerpt is not a runnable substitute.

## Offline command

```bash
PYTHONPATH=campaigns/phase-rlvr/src \
python -m phase_rlvr.observer <validation_data_dir> --k 4 \
  --output evidence/<run-id>/validation-summary.jsonl
```

The first observer derives only directly supported quantities: Pass@1, Pass@K, all-fail rate, and exact/canonical uniqueness among correct outputs. It does **not** invent entropy flow, structural diversity, verifier invariance, token efficiency, or train/inference discrepancy values when they are unavailable.

## G1 remaining instrumentation

Before adaptive control is allowed, WP01 must add governed collectors for:

1. signed entropy flow rather than entropy level alone;
2. a frozen structural correct-solution clusterer;
3. verifier-isomorphism transformations and invariance scoring;
4. explicit train/inference discrepancy quantiles;
5. training and evaluation token accounting;
6. robust windowed slope estimation with bootstrap uncertainty;
7. difficulty-bucket identifiers defined from frozen-base sampling.

Until these fields are available, observer summaries may support measurement design but may not instantiate a fully safe adaptive control state.
