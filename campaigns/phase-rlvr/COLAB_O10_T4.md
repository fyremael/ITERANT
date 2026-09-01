# O10-T4 Colab CLI Runbook

## Disposition

`COLAB_CLI_ELIGIBLE_FOR_PHASE_RLVR_O10__ADAPTER_IMPLEMENTED`

This lane is a T1 systems/observer experiment. It is not the T2 adaptive-vs-static scientific comparison.

## Governed envelope

- accelerator: exactly one Tesla T4 with at least 14,000 MiB visible HBM;
- precision: FP16;
- model: `Qwen/Qwen2.5-0.5B-Instruct`;
- model revision: `7ae557604adf67be50417f59c2c2f167def9a775`;
- dataset: `openai/gsm8k`;
- dataset revision: `a8402b897b382048aed4a75738da3c83e908a7f9`;
- veRL: `v0.8.0`, exact commit `7aed6b230776f963fa09509c10d9c3a767d1102c`;
- algorithm: fixed GRPO;
- adaptation: LoRA rank 16;
- train rollout group: 4;
- validation samples per prompt: 8;
- routine observer K: 4;
- bounded data: 512 training prompts, 32 frozen validation prompts;
- context: 256 prompt tokens / 256 response tokens;
- schedule: 40 optimizer steps in four 10-step segments;
- controller mutation: physically absent from the veRL command and required `false` in the manifest.

These are O10 plumbing defaults, not promoted scientific hyperparameters.

## Why segmented execution

Colab is treated as disposable compute. The host is the durable experiment ledger.

Each 10-step segment produces:

1. a veRL checkpoint suitable for exact resume;
2. raw validation JSONL;
3. raw rollout JSONL when veRL emits it;
4. an offline observer summary;
5. generated-token counts;
6. a segment receipt;
7. an evidence archive;
8. a resume archive.

The host downloads and verifies both archives before authorizing the next segment. A runtime eviction therefore loses at most the active segment.

## Host command

From the ITERANT repository root:

```bash
./campaigns/phase-rlvr/scripts/colab_run_o10_t4.sh
```

The host must provide the authenticated `google-colab-cli` command as `colab`. The runner uses the existing explicit lifecycle:

```text
new -> status -> upload -> exec -> download/log -> stop
```

No notebook state is admitted as evidence until the corresponding host-side archive and digest are present.

## Resume after runtime loss

Point the next invocation at the latest admitted resume bundle:

```bash
PHASE_RLVR_RESUME_ARCHIVE=runs/hosted/<run-id>/resume_latest.tar.gz \
  ./campaigns/phase-rlvr/scripts/colab_run_o10_t4.sh
```

The remote worker re-downloads the exact pinned base model and restores only governed training state from the resume archive.

## Evidence statuses

`GREEN_BOOTSTRAP` means the runtime, T4 allocation, and exact veRL checkout were established.

`GREEN_OBSERVER` means one training segment completed and its observer/evidence bundle passed the remote contract.

`GREEN_O10_T4_HOSTED` means every declared segment was downloaded and admitted by the host-side verifier.

None of these statuses is evidence that RLVR has controllable phases.

## Escalation boundary

T4 is deliberately restricted to O10/O11 systems and measurement work. Capacity-limited failures are not interpreted as PHASE-RLVR failures.

A100/H100 resources may be allocated when the next discriminating experiment requires larger model scale, rollout multiplicity, context, or throughput. Hardware escalation does not alter the scientific acceptance criteria; it changes only the declared execution tier and hardware receipt.
