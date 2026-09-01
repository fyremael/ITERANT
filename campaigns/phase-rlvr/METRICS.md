# PHASE-RLVR Metrics Contract

## Primary endpoints

1. **Compute-normalized Pass@1 AUC** over training tokens and wall-clock time.
2. **High-K coverage** on a frozen evaluation panel (`K=16/64` in the pilot; larger K only as an audit when affordable).
3. **Correct-solution diversity** conditioned on verifier-positive outputs.
4. **Tokens per validated correct answer**, including rollout and evaluation tokens.
5. **Verifier invariance** under a fixed semantics-preserving transformation suite.
6. **Unsafe-window count** and recovery duration.

No single metric is sufficient for promotion.

## Pass@K

Given `n` sampled responses of which `c` are correct, the finite-sample estimator is

\[
\widehat{pass@K}=1-\frac{\binom{n-c}{K}}{\binom nK}
\]

when `n-c >= K`, otherwise 1.0. Raw sample outcomes are retained so all K values can be recomputed.

## Reachability slopes

Slopes are estimated over a predeclared window using a robust estimator. Report estimate and bootstrap interval. Routine regime decisions use modest K; high-K audits are less frequent to prevent the controller from consuming its advantage in evaluation cost.

## Diversity among correct solutions

At least two views must be logged:

- exact/canonical trace uniqueness;
- structural clustering, such as equation/program/operator sequence signatures.

The clusterer is frozen before adaptive comparisons. Diversity calculated over all outputs is secondary because garbage diversity is not exploration utility.

## Entropy and entropy flow

Log mean token entropy `H`, but controller safety uses signed entropy-flow telemetry `Phi_H` when available. Entropy must be stratified by response position or semantic token class when diagnostics are run.

## Efficiency

Report separately:

- generated training tokens;
- generated evaluation tokens;
- verifier calls;
- optimizer updates;
- GPU-seconds;
- peak memory;
- correct-answer tokens;
- wall-clock to target Pass@1 thresholds.

## Numerical consistency

`Delta_TI` compares training-engine and rollout-engine log probabilities on identical frozen sequences. Report median, high quantiles, and worst finite value. A threshold breach is a safety event, not a phase signal.

## Statistical reporting

- minimum three training seeds for promoted T2 claims;
- paired prompt panels across methods;
- bootstrap confidence intervals over prompts and seeds;
- effect sizes with raw denominators;
- no promotion from a single best seed;
- adaptive-controller evaluation includes its extra telemetry/evaluation cost.
