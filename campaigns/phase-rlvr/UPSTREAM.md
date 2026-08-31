# Upstream and Literature Pins

## veRL integration anchor

- Repository: `verl-project/verl`.
- Release: `v0.8.0`.
- Exact tag commit: `7aed6b230776f963fa09509c10d9c3a767d1102c`.

WP01 uses the v0.8.0 validation and rollout dump surfaces in read-only mode. The O10-T4 bootstrap checks out this exact commit and records the realized Torch, vLLM, Ray, Transformers, Datasets, PEFT, and veRL package versions in the hosted receipt.

## O10-T4 model pin

- Repository: `Qwen/Qwen2.5-0.5B-Instruct`.
- Exact revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Role: T1 systems/observer model only.

The base snapshot is downloaded by exact revision on every fresh worker. LoRA is used to bound the T4 training state.

## O10-T4 data pin

- Dataset: `openai/gsm8k`.
- Configuration: `main`.
- Exact revision: `a8402b897b382048aed4a75738da3c83e908a7f9`.
- Bounded O10 panel: 512 deterministically shuffled training examples and the first 32 test examples.

The prepared parquet files carry dataset revision and dataset fingerprints in `dataset_receipt.json`.

## Literature anchors

- Ring-Zero: `arXiv:2607.12395`.
- OPEFO: `arXiv:2605.11491`.
- UCPO: `arXiv:2605.00365`.
- Pass@k Training: `arXiv:2508.10751`.
- Pass@k inversion / Per-Problem Base Anchoring: `arXiv:2607.20543`.
- RLVR reasoning-boundary / CoT-Pass@K: ICLR 2026 paper “Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs”.

These are conceptual anchors, not reproduced evidence.
