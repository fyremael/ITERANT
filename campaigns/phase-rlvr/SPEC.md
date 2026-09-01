# Programme Specification: PHASE-RLVR

## 1. Falsifiable research claim

Under matched train-token, evaluation-token, verifier, model, data, and wall-clock accounting, an online controller that infers operational RLVR regimes from reachability, reliability, diversity, entropy flow, efficiency, verifier robustness, and numerical-stability telemetry can improve the validated utility frontier relative to both:

1. the strongest fixed RLVR recipe in the audited action set; and
2. the strongest predeclared fixed-stage schedule using the same recipes.

A positive result requires recurrence across seeds and at least two evaluation families. A single final Pass@1 gain is insufficient.

## 2. Non-claims

PHASE-RLVR does not initially claim:

- universal discrete phases of RLVR;
- causal identification from observational telemetry alone;
- equivalence between response length and reasoning depth;
- entropy as a sufficient exploration statistic;
- transfer of 1T-model dynamics to small models;
- that high-K success alone proves novel reasoning;
- that an adaptive controller may change arbitrary hyperparameters online.

## 3. State and observables

For difficulty bucket `b`, estimate

\[
x_t^{(b)}=
[\hat p_1,\hat p_K,\dot p_1,\dot p_K,q_{fail},H,\Phi_H,
D_c,\dot D_c,\eta_{tok},\dot\eta_{tok},D_{KL},\Delta_{TI},V_{iso}].
\]

Definitions:

- `p1`: one-sample verifier success on a frozen evaluation panel.
- `pK`: multi-sample success; routine control uses modest K and periodic audits use larger K.
- `q_fail`: fraction of prompt groups with no verifier-positive rollout.
- `H`: policy entropy on governed token positions.
- `Phi_H`: update-induced entropy flow, retaining sign information rather than only mean entropy.
- `D_c`: diversity among verifier-positive solutions, computed with a declared canonicalization/clusterer.
- `eta_tok`: validated utility improvement per generated training token.
- `D_KL`: policy drift to the declared reference.
- `Delta_TI`: train/inference log-probability discrepancy on identical sequences.
- `V_iso`: verifier consistency under declared semantics-preserving transformations.

All slopes are robust finite-window estimates with confidence intervals. A point estimate may not trigger promotion.

## 4. Operational regime classifier

The initial classifier is deliberately transparent.

### Discovery

\[
\dot p_K>\delta_K
\quad\land\quad
V_{iso}\ge \tau_{stable}
\quad\land\quad
\text{no safety violation}.
\]

### Sharpening

\[
|\dot p_K|\le\delta_{flat}
\quad\land\quad
\dot p_1>\delta_1
\quad\land\quad
\dot D_c\ge\delta_{D,min}
\quad\land\quad
\text{no safety violation}.
\]

### Stall

\[
|\dot p_1|\le\delta_s,
\quad |\dot p_K|\le\delta_s,
\quad |\dot\eta_{tok}|\le\delta_\eta.
\]

### Collapse / unsafe

Any declared invariant breach, including excessive negative entropy flow, correct-diversity loss, KL excursion, train/inference discrepancy, or verifier-invariance failure, is classified fail-closed as `collapse` regardless of apparent reward gains.

### Uncertain

All other windows are `uncertain`; the default action is hold rather than forced classification.

## 5. Hysteresis and control timing

- ordinary regime changes require at least three consecutive qualifying telemetry windows;
- safety violations bypass hysteresis and route immediately to recovery;
- a minimum dwell interval prevents rapid recipe chatter;
- controller decisions occur at checkpoint/control boundaries, never midway through an optimizer step;
- all controller inputs must be timestamped before the selected action is applied.

This separation is required to preserve causal auditability.

## 6. Audited action set

### A0 — Discovery

Purpose: preserve/expand reachable correct modes near the current boundary.

Default interventions:

- low/intermediate length-normalization exponent `alpha`;
- grouped rollouts and Pass@k-aware advantage option;
- frontier-weighted curriculum;
- larger but bounded context allowance;
- entropy-flow guard.

### A1 — Sharpening

Purpose: increase probability and token efficiency on already reachable solutions without diversity collapse.

Default interventions:

- sample-normalized loss (`alpha -> 1`);
- Pass@1-oriented advantage;
- smaller rollout group;
- reduced context allowance;
- consolidation-weighted curriculum.

### A2 — Recovery

Purpose: arrest instability.

Default interventions:

- reduced learning-rate multiplier;
- stronger KL control;
- frozen sensitive mixed-precision policy;
- rollout/training-engine resynchronization when required;
- no curriculum expansion.

### A3 — Rebalance

Purpose: correct prompt-distribution mismatch or reward starvation.

Default interventions:

- increase frontier/boundary prompt mass;
- moderate length normalization;
- retain grouped rollouts;
- conservative optimizer multiplier.

The first adaptive experiment may choose only among these actions. Learned controllers are forbidden from synthesizing new recipes.

## 7. Continuous length-normalization contract

For response `i` of length `T_i`,

\[
\mathcal L_\alpha =
-\frac{1}{B}\sum_i
\frac{A_i}{T_i^\alpha}
\sum_{t=1}^{T_i}\ell^{PG}_{i,t}
+\beta D_{KL},
\qquad \alpha\in[0,1].
\]

`alpha=0` reproduces token-summed sequence weighting; `alpha=1` gives sample-normalized weighting. Intermediate alpha values are experimental and must be ablated. Precision policy is not an ordinary phase-control dimension.

## 8. Controller utility

Controller ranking uses delayed validation utility, not training reward:

\[
U = w_1\Delta p_1+w_K\Delta p_K+w_D\Delta D_c
-c_TN_{train\ tok}-c_EN_{eval\ tok}-c_WT_{wall}
-\lambda_V(1-V_{iso})-\lambda_R R_{unsafe}.
\]

Primary reporting is Pareto-based; scalar `U` is secondary and may rank candidates only within predeclared constraints.

## 9. Experimental sequencing

### WP00 — Controller contract

Synthetic deterministic tests. Gate: 100% expected transition and invariant behavior.

### WP01 — Observer-only

No controller actions. Estimate measurement variance, lag, autocorrelation, and bucket heterogeneity. Gate: regime labels must be stable enough under bootstrap/window perturbations to make intervention scientifically interpretable.

### WP02 — Static recipes

Each action is run as a fixed policy. Gate: identify dominated recipes and calibrate safety thresholds before adaptation.

### WP03 — Adaptive rule controller

Matched compute comparison against fixed and predeclared-stage baselines. Gate: positive compute-normalized utility with no safety regression on at least 2/3 seeds.

### WP04 — Bucket-local controller

Permit different curriculum weights by base-difficulty bucket. Gate: improvement must not be attributable solely to shifting evaluation composition.

### WP05 — Constrained no-regret router

Contextual router over audited actions only. Gate: outperform deterministic routing without increased safety violations or materially worse variance.

### WP06 — Confirmation

Second model scale and second task family. Only here may the project discuss cross-scale generality.

## 10. Reproducibility contract

Each run records:

- repository exact head;
- upstream veRL exact commit/release;
- model and tokenizer immutable identifiers;
- dataset revision and prompt split hashes;
- verifier code hash and transformation-suite hash;
- action/threshold config hash;
- random seeds for sampling, training, and data order;
- hardware, driver, CUDA, PyTorch, rollout backend, and precision policy;
- every telemetry window before action selection;
- every selected action and its effective step interval;
- train/evaluation token counts and wall-clock time;
- raw sample-level verifier outputs sufficient to recompute Pass@K.

## 11. Stop conditions

Close or redesign the campaign if any of the following recur under adequate measurement power:

- inferred regimes are not stable under reasonable window/bootstrap perturbations;
- the strongest fixed recipe matches adaptive control within uncertainty;
- gains disappear under matched evaluation-token accounting;
- apparent gains are explained by verifier exploitation or diversity collapse;
- control switching itself induces instability exceeding any benefit;
- bucket-local phases provide no predictive value for subsequent recipe response.
