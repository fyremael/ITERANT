# Hard Submission Gate

A hosted Hard attempt is a scarce, externally visible experiment. It is not an exploratory run.

## Machine gate

`old-campaign gate` fails closed unless all conditions in `hard_gate_policy.json` are satisfied:

1. evidence exists for M1–M5;
2. each dataset has the minimum governed seed count;
3. every record uses the exact upstream evaluator commit;
4. the candidate exceeds its paired baseline by the declared minimum;
5. throughput remains above the declared fraction of baseline;
6. held-out-depth degradation remains bounded;
7. peak GPU memory preserves safety headroom;
8. recurrent dynamics are finite and state norms remain bounded;
9. all required numeric values are finite.

Thresholds may be revised before a new evidence campaign begins. They may not be relaxed after observing a candidate merely to admit it.

## Council obligations

- **Axiomatist:** confirms claims and assumptions.
- **Cartographer:** confirms run lineage and phase dependencies.
- **Verifier:** confirms evaluator pin, generated source, hashes, and repeatability.
- **Adversary:** searches for public-tier specialization, leakage, instability, and measurement confounds.
- **Steward:** confirms compute expenditure is proportionate.
- **Amanuensis:** confirms the artifact ledger and decision record are complete.
- **Referee:** applies the declared gate and records residual uncertainty.

## Human authorization

A passing machine gate produces an eligible candidate, not an automatic submission. The Human Steward reviews the candidate source, evidence summary, known limitations, and reversal plan, then explicitly authorizes or declines the hosted attempt.

## Post-attempt rule

The returned Hard score is recorded without retroactively rewriting the admission rationale. A failed attempt updates the evidence model and candidate frontier; it does not authorize metric exploitation or uncontrolled retries.
