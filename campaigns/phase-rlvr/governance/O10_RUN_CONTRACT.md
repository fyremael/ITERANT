# WP01-O10 Run Contract

O10 is the first real-system PHASE-RLVR observation. It is intentionally incapable of testing adaptive control.

## Fixed treatment

The training treatment is a single, predeclared GRPO recipe for the entire run. The exact veRL upstream is `v0.8.0` at commit `7aed6b230776f963fa09509c10d9c3a767d1102c`.

The run must set and retain:

- one model/tokenizer revision;
- one train and validation dataset revision;
- one verifier implementation hash;
- one loss aggregation mode;
- one KL configuration;
- one rollout group size for training;
- one sampled validation count `N >= routine K`;
- fixed learning-rate schedule and optimizer configuration;
- fixed validation cadence;
- `controller_mutation_enabled=false` in the evidence manifest.

PHASE-RLVR may read generated validation/rollout data after veRL emits it. It may not write into veRL configuration or trainer state.

## Minimal veRL observer surfaces

The governed configuration must enable sampled validation before training and at a fixed interval, and persist step-indexed validation JSONL. At v0.8.0 the relevant configuration surfaces include `trainer.val_before_train`, `trainer.test_freq`, `trainer.validation_data_dir`, and `actor_rollout_ref.rollout.val_kwargs.n`.

The exact command line is a run artifact, not part of this generic contract, because GPU count, model identity, and dataset paths are runtime-specific.

## Required manifest

Before a run is admitted as O10 evidence, create a JSON object containing every field required by `scripts/check_o10_manifest.py` and validate it:

```bash
python campaigns/phase-rlvr/scripts/check_o10_manifest.py evidence/<run-id>/manifest.json
```

The manifest pins the ITERANT head, upstream veRL commit, model/tokenizer revisions, data revisions, verifier hash, independent seeds, hardware/software identity, fixed recipe hash, raw data locations, and the negative capability flag `controller_mutation_enabled=false`.

## Required outputs

An admitted O10 evidence packet contains:

```text
evidence/<run-id>/
  manifest.json
  command.txt
  config-resolved.yaml
  validation/*.jsonl
  validation-summary.jsonl
  metrics.jsonl
  stdout.log
  environment.txt
  hashes.sha256
```

If rollout dumps are enabled, they are retained under `rollouts/` and included in `hashes.sha256`.

## Admission rule

O10 can establish only that the telemetry path works on a real RLVR run. It cannot establish discovery, sharpening, causal regime control, or adaptive advantage. Those require later gates.
