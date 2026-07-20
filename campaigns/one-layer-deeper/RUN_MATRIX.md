# Public-Tier Run Matrix

The machine-readable source is `configs/run_matrix.json`. The initial matrix is intentionally narrow; replications are added after gross failures are eliminated.

| Phase | Profiles | Public probes | Decision question |
|---|---|---|---|
| P1 | `baseline_adamw` | E1, M1, M5 | What are the official score, update throughput, memory, and variance reference points? |
| P2 | `tied_transformer_adamw`, `neural_tape_adamw` | E1, M5 | Which recurrent primitive gives the better wall-clock frontier? |
| P3 | `tied_transformer_extrapolation` | M1, M2, M5 | Does stochastic unrolling improve held-out depth and useful over-unrolling? |
| P4 | `tied_transformer_stable` | M1, M2, M5 | Do inexpensive stability controls expand the stable computation interval? |
| P5 | stable AdamW, hybrid Muon, groupwise AdamW | M5, then M1–M5 | Which optimizer converts a fixed hour into the greatest generalizing accuracy? |
| P6 | promoted candidate | M1–M5, ≥3 seeds | Is evidence sufficient to justify one daily Hard attempt? |

## Matched-cost rule

Comparisons use the evaluator's fixed wall-clock allowance. Completed updates, batch size, model-state count, optimizer-state count, and evaluation time are reported rather than artificially equalized.

## Replication rule

A single run can reject a grossly defective candidate but cannot promote one to Hard eligibility. Promotion evidence requires at least three governed seeds per required Medium dataset.

## Dataset interpretation

- M1/M2 emphasize fixed-modulus depth variation.
- M3/M4 emphasize modulus variation at fixed public depth.
- M5 jointly varies modulus family and depth and is the first screening proxy.

No public task is treated as a faithful disclosure of the hidden recurrence.
