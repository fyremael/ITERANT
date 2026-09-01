# PHASE-RLVR Run Matrix

| ID | WP | Condition | Hardware / model tier | Controller | Purpose | Promotion evidence |
|---|---|---|---|---|---|---|
| C00 | WP00 | synthetic regime replay | T0 / CPU | rule | exact state contract | all tests pass |
| C01 | WP00 | safety override replay | T0 / CPU | rule | fail-closed behavior | immediate recovery on every invariant breach |
| C02 | WP00 | hysteresis/chatter replay | T0 / CPU | rule | temporal stability | no premature transition |
| O10-T4 | WP01 | fixed GRPO observer | T1 / T4 / 0.5B | absent | telemetry + hosted evidence plumbing | all segmented receipts admitted |
| O11 | WP01 | fixed GRPO observer, seed 2 | T1; T4 unless capacity requires escalation | absent | measurement recurrence | compatible regime reconstruction |
| O12 | WP01 | window/bootstrap sweep | T0/offline | absent | label robustness + prospective value | declared stability/information threshold met |
| S20 | WP02 | fixed discovery recipe | T2 / A100 or H100 as required / 1.5B–4B | fixed | action frontier | matched-compute metrics |
| S21 | WP02 | fixed sharpening recipe | T2 / A100 or H100 as required / 1.5B–4B | fixed | action frontier | matched-compute metrics |
| S22 | WP02 | predeclared two-stage | T2 / A100 or H100 as required / 1.5B–4B | fixed stage | Ring-Zero-style baseline | matched-compute metrics |
| S23 | WP02 | fixed GRPO/veRL baseline | T2 / A100 or H100 as required / 1.5B–4B | fixed | standard baseline | matched-compute metrics |
| A30 | WP03 | PHASE-RLVR rule, seed 1 | T2 / A100 or H100 as required | adaptive | main test | primary endpoint |
| A31 | WP03 | PHASE-RLVR rule, seed 2 | T2 / A100 or H100 as required | adaptive | recurrence | primary endpoint |
| A32 | WP03 | PHASE-RLVR rule, seed 3 | T2 / A100 or H100 as required | adaptive | recurrence | primary endpoint |
| B40 | WP04 | bucket-local phase control | T2 / capacity matched | adaptive | heterogeneity test | fixed eval composition; positive utility |
| N50 | WP05 | constrained no-regret router | T2 / capacity matched | learned selection | scheduler test | beat rule controller within safety bounds |
| X60 | WP06 | promoted controller, larger model | T3 / H100-class as required / >=7B | promoted | scale confirmation | recurrence |
| X61 | WP06 | second reasoning domain | T2/T3 | promoted | domain confirmation | recurrence |

## O10-T4 defaults

The first real-system experiment is a bounded Colab T4 run using `Qwen/Qwen2.5-0.5B-Instruct`, LoRA, fixed GRPO, rollout group 4, validation `n=8`, routine `K=4`, 256/256 prompt-response caps, and four 10-step segments. This is a measurement-system qualification, not a model-capability result.

The host must admit each segment's evidence and resume archive before the run can proceed. Controller mutation is prohibited by manifest and by construction of the veRL command.

## Capacity escalation

T4 is preferred only while it remains the smallest accelerator capable of falsifying the next claim. A100/H100 resources may be allocated on a needs basis when memory, context, rollout multiplicity, model scale, or wall-clock efficiency becomes part of the discriminating experiment.

No result may be promoted because it used a larger accelerator; scientific comparisons remain matched on declared token/evaluation budgets and report hardware separately.

## T2 defaults

T2 should use a roughly 1.5B–4B base model on a mathematical RLVR corpus with a separately frozen evaluation panel spanning solved, boundary, and currently-unsolved buckets. Difficulty buckets are defined from frozen-base sampling before RL begins and are not recomputed from the adaptive policy for the main comparison.

## Compute matching

For every adaptive/fixed comparison, report both:

1. equal generated **training-token** budget; and
2. equal total **train + controller-evaluation token** budget.

A method that wins only by spending materially more on phase measurement is not promoted as compute-efficient.
