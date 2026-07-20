# Programme Specification: One Layer Deeper

## 1. Research claim

A compact model with a learned, weight-tied transition operator can achieve greater exact task accuracy per H100 second than an untied shallow baseline, while retaining useful accuracy when computational depth or problem family is held out.

The programme does not assume the hidden Hard recurrence is repeated modular squaring. Public tasks are treated as probes of reusable composition, not as permission to encode a modular-arithmetic solver.

## 2. Objective function

The primary external score is evaluator-defined mean exact sequence accuracy. Internal model selection is multi-objective:

\[
J = \operatorname{ExactAcc}_{\mathrm{OOD}}
- \lambda_t \log \frac{\mathrm{seconds}}{\mathrm{update}}
- \lambda_m \frac{\mathrm{peak\ memory}}{\mathrm{H100\ memory}}
- \lambda_g \max(0,\operatorname{IDAcc}-\operatorname{DepthAcc}).
\]

No scalarized internal score may override the Hard gate. It is used only for ranking candidates within an approved phase.

## 3. Architectural candidates

### 3.1 Official-style baseline

One Transformer block, learned token/position embeddings, tied output embedding, RMS normalization, AdamW. Its role is reproduction and measurement calibration.

### 3.2 Tied recurrent Transformer

One shared attention/MLP cell is applied repeatedly:

\[
h_{k+1}=h_k+\alpha_a A_\theta(h_k,c)+\alpha_m M_\theta(h_k,c).
\]

Training and evaluation unroll counts are explicit profile parameters. Stochastic unrolls are sampled once per forward pass; evaluation uses a fixed governed count.

### 3.3 Recurrent neural tape

A shared local convolution and gated channel mixer update short digit/work positions. This candidate tests whether local recurrent computation provides a superior throughput–accuracy frontier to attention.

## 4. Stability interventions

Stability mechanisms enter only after recurrent baselines are measured:

1. sigmoid-gated residual scales;
2. per-loop RMS state normalization;
3. a low-weight penalty on variance of loop update energy.

A mechanism is retained only when it improves held-out-depth accuracy or permits deeper evaluation without an unacceptable throughput loss.

## 5. Optimizer conditions

- **AdamW:** common baseline, betas `(0.9, 0.95)`.
- **Hybrid Muon:** Newton–Schulz orthogonalized updates for internal matrix parameters and AdamW for embeddings, norms, biases, gates, and readout boundary state.
- **Groupwise AdamW:** slower momentum decay for the recurrent core and faster boundary adaptation for encoder/readout parameters.

Optimizer comparisons hold architecture, profile depth, batch size, and time tier fixed.

## 6. Experimental phases

### P1 — Reproduction

Reproduce baseline E1, M1, and M5. Record exact accuracy, completed steps, examples/s, peak GPU memory, evaluator commit, Python/Torch/CUDA/driver identity, and full result payload. Repeat baseline seeds before using variance estimates in candidate judgments.

### P2 — Recurrence

Compare tied Transformer and neural tape first on E1, then M5. Match evaluator wall-clock allowance, not nominal parameter count or step count. Reject candidates that gain accuracy only by exceeding evaluation time or memory limits.

### P3 — Extrapolation

Train on a distribution of unroll counts and evaluate beyond the maximum common training depth. Produce accuracy-versus-unroll and state-norm-versus-unroll curves. A useful operator should exhibit a nontrivial stable interval rather than one privileged depth.

### P4 — Stability

Ablate each intervention individually before combining. The combined stable profile is promoted only if at least one of the following is observed on replicated runs:

- improved held-out-depth exact accuracy;
- a wider stable evaluation-unroll interval;
- reduced catastrophic state growth without material throughput loss.

### P5 — Optimizer co-design

Compare the three optimizer modes on the promoted architecture. Report optimizer-state size after the first step, updates/s, score, and run-to-run variance. Muon is rejected if Newton–Schulz cost erases its learning-speed advantage under the wall clock.

### P6 — Hard governance

The machine gate requires complete M1–M5 evidence, at least three seeds per dataset, minimum baseline gain, bounded throughput loss, bounded held-out-depth gap, peak-memory headroom, finite dynamics, and the exact evaluator pin. Passing the machine gate does not submit. The Human Steward authorizes the daily hosted attempt.

## 7. Reproducibility contract

Every run artifact must contain:

- immutable upstream commit;
- generated profile name and generated submission hash;
- command and manifest path;
- seed;
- hardware, driver, Python, and package identity;
- raw evaluator result and logs;
- profiler measurements;
- classification as `official-faithful` or `resource-adapted`.

Only official-faithful evidence may satisfy the Hard gate.

## 8. Acceptance and rejection

A candidate is promoted between phases only when it survives its declared discriminating test. Architectural novelty is not an acceptance criterion. The simplest candidate on the superior wall-clock frontier is preferred.
