# PHASE-RLVR Run Matrix

| ID | WP | Condition | Model tier | Controller | Purpose | Promotion evidence |
|---|---|---|---|---|---|---|
| C00 | WP00 | synthetic regime replay | T0 | rule | exact state contract | all tests pass |
| C01 | WP00 | safety override replay | T0 | rule | fail-closed behavior | immediate recovery on every invariant breach |
| C02 | WP00 | hysteresis/chatter replay | T0 | rule | temporal stability | no premature transition |
| O10 | WP01 | fixed GRPO observer | T1 | disabled | telemetry plumbing | complete pre-action logs |
| O11 | WP01 | fixed GRPO observer, seed 2 | T1 | disabled | measurement recurrence | compatible regime reconstruction |
| O12 | WP01 | window/bootstrap sweep | T0/offline | disabled | label robustness | declared stability threshold met |
| S20 | WP02 | fixed discovery recipe | T2 | fixed | action frontier | matched-compute metrics |
| S21 | WP02 | fixed sharpening recipe | T2 | fixed | action frontier | matched-compute metrics |
| S22 | WP02 | predeclared two-stage | T2 | fixed stage | Ring-Zero-style baseline | matched-compute metrics |
| S23 | WP02 | fixed GRPO/veRL baseline | T2 | fixed | standard baseline | matched-compute metrics |
| A30 | WP03 | PHASE-RLVR rule, seed 1 | T2 | adaptive | main test | primary endpoint |
| A31 | WP03 | PHASE-RLVR rule, seed 2 | T2 | adaptive | recurrence | primary endpoint |
| A32 | WP03 | PHASE-RLVR rule, seed 3 | T2 | adaptive | recurrence | primary endpoint |
| B40 | WP04 | bucket-local phase control | T2 | adaptive | heterogeneity test | fixed eval composition; positive utility |
| N50 | WP05 | constrained no-regret router | T2 | learned selection | scheduler test | beat rule controller within safety bounds |
| X60 | WP06 | promoted controller, larger model | T3 | promoted | scale confirmation | recurrence |
| X61 | WP06 | second reasoning domain | T2/T3 | promoted | domain confirmation | recurrence |

## Pilot defaults

The first real-system smoke test should use a model small enough that grouped rollouts are practical. veRL currently documents GRPO plumbing down to Qwen3-0.6B; this is suitable for T1 systems validation, not for a strong scientific claim.

T2 should use a roughly 1.5B–4B base model on a mathematical RLVR corpus with a separately frozen evaluation panel spanning solved, boundary, and currently-unsolved buckets. Difficulty buckets are defined from frozen-base sampling before RL begins and are not recomputed from the adaptive policy for the main comparison.

## Compute matching

For every adaptive/fixed comparison, report both:

1. equal generated **training-token** budget; and
2. equal total **train + controller-evaluation token** budget.

A method that wins only by spending materially more on phase measurement is not promoted as compute-efficient.
