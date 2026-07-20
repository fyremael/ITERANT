# Metrics and Telemetry

## External outcomes

- mean exact accuracy;
- split exact accuracy and loss;
- exact accuracy by requested recurrence depth where locally derivable without inspecting protected evaluator files;
- ID versus held-out-depth gap.

## Wall-clock efficiency

- completed optimizer updates;
- updates per training second;
- examples per training second;
- end-to-end wall seconds;
- evaluation seconds versus allowance;
- model import and construction overhead where observable.

## Memory

- model persistent-state elements;
- optimizer-state elements after the first step;
- peak visible GPU memory sampled by `nvidia-smi`;
- H100 headroom.

## Recurrent dynamics

Instrumented development runs may record non-persistent summaries from the recurrent state:

- per-loop RMS state norm;
- per-loop update RMS;
- cosine similarity of successive updates;
- prediction entropy by loop;
- exact accuracy by evaluation unroll;
- finite/non-finite flag;
- p50/p95/p99 state norm;
- approximate dominant Jacobian gain on a small probe batch.

Instrumentation that changes official submission runtime must be disabled in scoring profiles and retained only in diagnostic variants.

## Evidence schema

`profile_h100.py` records raw official results and resource measurements. `make_evidence.py` pairs a candidate and baseline under the same upstream revision, tier, dataset, and seed. The Hard gate consumes only paired evidence records.
