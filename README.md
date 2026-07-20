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

## Active campaigns

### One Layer Deeper

The first campaign targets the Tilde Research / Core Automation architecture-and-optimizer competition. It includes the official-style AdamW baseline, tied recurrent Transformer and neural-tape cores, stochastic unroll curricula, low-cost RUNT/SPINDLE stability mechanisms, optimizer co-design, H100 profiling, and a fail-closed Hard-submission gate.

See [`campaigns/one-layer-deeper/README.md`](campaigns/one-layer-deeper/README.md).

## Repository map

```text
campaigns/one-layer-deeper/   competition implementation and evidence contract
.github/workflows/             pinned evaluator and repository validation
GOVERNANCE.md                  boundary between research execution and institutional authority
```

## Validation

```bash
make check
```

Official H100 measurements require the pinned evaluator environment and exactly one visible H100. CPU validation establishes source and evaluator compatibility only; it is not performance evidence.
