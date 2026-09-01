# ITERANT

**Governed research on learned iterative computation, recurrent depth, and architecture–optimizer co-design.**

ITERANT develops compact systems that perform useful computation by repeatedly applying shared operators. Its central concern is not recurrence as an architectural ornament, but whether learned iteration can provide reliable algorithmic depth, controlled extrapolation, and superior utility per unit of compute.

## Research boundary

ITERANT owns executable campaigns, model implementations, optimizer experiments, profiling, evidence, and benchmark submissions. Institutional acceptance, review provenance, and irreversible submission authority remain governed through Grand Challenge Labs' INTELLECT system.

```text
ITERANT                                   INTELLECT
implementation and experiments     <->   charter and acceptance criteria
run artifacts and diagnostics      <->   review and decision records
candidate submissions              <->   Human Steward authorization
```

## Campaigns

### PHASE-RLVR

PHASE-RLVR studies whether RLVR training can be supervised by an operational regime detector that distinguishes discovery, sharpening, stall, collapse, and uncertainty from governed telemetry, then selects among a small audited set of training recipes. The campaign begins with a synthetic controller contract before any adaptive LLM training.

See [`campaigns/phase-rlvr/README.md`](campaigns/phase-rlvr/README.md).

### One Layer Deeper

The One Layer Deeper campaign targets learned recurrent computation for the Tilde Research / Core Automation architecture-and-optimizer setting. It includes the official-style AdamW baseline, tied recurrent Transformer and neural-tape cores, stochastic unroll curricula, low-cost RUNT/SPINDLE stability mechanisms, optimizer co-design, H100 profiling, and a fail-closed Hard-submission gate.

See [`campaigns/one-layer-deeper/README.md`](campaigns/one-layer-deeper/README.md).

## Repository map

```text
campaigns/phase-rlvr/         RLVR regime-control implementation and evidence contract
campaigns/one-layer-deeper/   recurrent-depth implementation and evidence contract
.github/workflows/             pinned evaluator and repository validation
GOVERNANCE.md                  boundary between research execution and institutional authority
```

## Validation

```bash
make check
```

Campaign contract tests establish source and control-logic correctness only. Performance, RLVR, or scientific claims require immutable run evidence under the relevant campaign acceptance contract.
