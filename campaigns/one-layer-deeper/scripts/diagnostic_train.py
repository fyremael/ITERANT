from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("diagnostic_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-adapted recurrent dynamics probe")
    parser.add_argument("--evaluator-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--trace-loops", type=int, default=32)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()

    evaluator_root = Path(args.evaluator_root).resolve()
    sys.path.insert(0, str(evaluator_root))
    from benchmark.api import ModelSpec, OptimizerSpec
    from benchmark.batches import prepare_batch
    from benchmark.manifest import load_manifest
    from benchmark.runner import _loss_and_accuracy
    from data import infer_max_seq_len, infer_vocab_size, make_dataloaders

    manifest = load_manifest(args.manifest)
    device = torch.device(manifest.runtime.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic run requires exactly one visible GPU")
    dataloaders = make_dataloaders(manifest.data, device=device)
    module = _load_module(Path(args.submission).resolve())
    spec = ModelSpec(
        vocab_size=infer_vocab_size(manifest.data),
        max_seq_len=infer_max_seq_len(manifest.data),
        maximum_model_state_elements=manifest.model_state.maximum_elements,
    )
    model = module.SUBMISSION.build_model(spec).to(device=device, dtype=torch.float32)
    bundle = module.SUBMISSION.build_optimizer(
        model,
        OptimizerSpec(training_time_seconds=args.seconds, device_type=device.type),
    )
    model.train()
    iterator = iter(dataloaders["train"])
    started = time.monotonic()
    steps = 0
    final_loss = None
    while time.monotonic() - started < args.seconds:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloaders["train"])
            batch = next(iterator)
        bundle.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = _loss_and_accuracy(
            model,
            batch,
            manifest,
            device,
            training_loss=module.SUBMISSION.training_loss,
        )
        loss.backward()
        if manifest.runtime.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), manifest.runtime.grad_clip)
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
        steps += 1
        final_loss = float(loss.item())

    split = "ood" if "ood" in dataloaders else "test"
    batch = next(iter(dataloaders[split]))
    input_ids, _, attention_mask, _ = prepare_batch(batch, device)
    model.eval()
    limit = min(32, input_ids.shape[0])
    trace = model.trace_dynamics(
        input_ids[:limit],
        None if attention_mask is None else attention_mask[:limit],
        loops=args.trace_loops,
    )
    state_norms = [row["state_rms"] for row in trace]
    payload = {
        "schema_version": 1,
        "classification": "resource-adapted-diagnostic",
        "official_upstream_commit": args.upstream_commit,
        "manifest": manifest.name,
        "training_seconds": time.monotonic() - started,
        "completed_steps": steps,
        "final_loss": final_loss,
        "trace": trace,
        "dynamics": {
            "finite": all(row["finite"] for row in trace),
            "state_norm_max": max(state_norms, default=0.0),
            "state_norm_p99": (
                sorted(state_norms)[max(0, int(0.99 * len(state_norms)) - 1)]
                if state_norms else 0.0
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
