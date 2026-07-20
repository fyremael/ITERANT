from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from old_campaign.gate import evaluate_hard_gate
from old_campaign.generator import generate_submission
from old_campaign.matrix import expand_runs


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int
    max_seq_len: int
    maximum_model_state_elements: int


@dataclass(frozen=True)
class OptimizerSpec:
    training_time_seconds: float
    device_type: str


@dataclass(frozen=True)
class OptimizerBundle:
    optimizer: object
    scheduler: object | None = None


@dataclass(frozen=True)
class Submission:
    build_model: object
    build_optimizer: object
    training_loss: object | None = None
    batch_size: int | None = None
    max_steps: int | None = None


def assert_model_state(model, spec):
    count = sum(parameter.numel() for parameter in model.parameters())
    count += sum(buffer.numel() for buffer in model.buffers())
    if count > spec.maximum_model_state_elements:
        raise AssertionError("model exceeds state budget")
    return count


def load_generated(profile: str):
    benchmark = types.ModuleType("benchmark")
    benchmark.ModelSpec = ModelSpec
    benchmark.OptimizerSpec = OptimizerSpec
    benchmark.OptimizerBundle = OptimizerBundle
    benchmark.Submission = Submission
    benchmark.assert_model_state = assert_model_state
    sys.modules["benchmark"] = benchmark
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "submission.py"
    generate_submission(
        ROOT / "profiles" / f"{profile}.json",
        ROOT / "templates" / "submission.py.tmpl",
        path,
    )
    module_spec = importlib.util.spec_from_file_location(f"generated_{profile}", path)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.D_MODEL = 24
    module.NUM_HEADS = 4
    module.TRAIN_LOOPS_MIN = 1
    module.TRAIN_LOOPS_MAX = 2
    module.EVAL_LOOPS = 2
    return temporary, path, module


class GenerationTests(unittest.TestCase):
    def test_all_profiles_generate_below_contract_limit(self):
        for profile in sorted((ROOT / "profiles").glob("*.json")):
            with self.subTest(profile=profile.stem), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "submission.py"
                generate_submission(profile, ROOT / "templates" / "submission.py.tmpl", output)
                self.assertLess(output.stat().st_size, 256 * 1024)
                compile(output.read_text(encoding="utf-8"), str(output), "exec")

    def test_recurrent_models_forward_backward_and_optimizer_coverage(self):
        for profile in (
            "tied_transformer_adamw",
            "neural_tape_adamw",
            "tied_transformer_muon",
            "tied_transformer_groupwise",
        ):
            with self.subTest(profile=profile):
                temporary, _, module = load_generated(profile)
                try:
                    spec = ModelSpec(17, 6, 500_000_000)
                    model = module.SUBMISSION.build_model(spec)
                    model.train()
                    input_ids = torch.randint(0, 17, (2, 6))
                    mask = torch.ones(2, 6, dtype=torch.bool)
                    logits, auxiliary = model(input_ids, attention_mask=mask)
                    self.assertEqual(tuple(logits.shape), (2, 6, 17))
                    token_logits = logits[:, 0, :].float()
                    labels = torch.randint(0, 17, (2,))
                    if module.SUBMISSION.training_loss is None:
                        loss = torch.nn.functional.cross_entropy(token_logits, labels)
                    else:
                        loss = module.SUBMISSION.training_loss(token_logits, labels, auxiliary)
                    loss.backward()
                    bundle = module.SUBMISSION.build_optimizer(
                        model, OptimizerSpec(60.0, "cpu")
                    )
                    expected = {
                        id(parameter)
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    }
                    actual = [
                        id(parameter)
                        for group in bundle.optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    self.assertEqual(expected, set(actual))
                    self.assertEqual(len(expected), len(actual))
                    bundle.optimizer.step()
                    bundle.scheduler.step()
                finally:
                    temporary.cleanup()


class MatrixTests(unittest.TestCase):
    def test_matrix_expands_unique_run_ids(self):
        runs = expand_runs(ROOT / "configs" / "run_matrix.json")
        self.assertGreaterEqual(len(runs), 10)
        identifiers = [run["run_id"] for run in runs]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual({run["phase"] for run in runs}, {
            "p1-reproduction",
            "p2-recurrence",
            "p3-extrapolation",
            "p4-stability",
            "p5-optimizer",
        })


class GateTests(unittest.TestCase):
    def _record(self, dataset: str, seed: int) -> dict:
        return {
            "official_upstream_commit": "885fb4c95fc00f39b5038f07169e79bc3b285d72",
            "dataset": dataset,
            "seed": seed,
            "candidate_exact_accuracy": 0.10,
            "baseline_exact_accuracy": 0.05,
            "candidate_examples_per_second": 800.0,
            "baseline_examples_per_second": 1000.0,
            "peak_memory_gib": 45.0,
            "heldout_depth_accuracy": 0.08,
            "id_accuracy": 0.10,
            "dynamics": {"finite": True, "state_norm_p99": 4.0},
        }

    def test_gate_fails_closed_for_missing_evidence(self):
        decision = evaluate_hard_gate([], ROOT / "governance" / "hard_gate_policy.json")
        self.assertFalse(decision.approved)
        self.assertTrue(decision.failures)

    def test_gate_accepts_complete_good_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for dataset in ("m1", "m2", "m3", "m4", "m5"):
                for seed in (11, 22, 33):
                    path = Path(directory) / f"{dataset}-{seed}.json"
                    path.write_text(
                        json.dumps(self._record(dataset, seed)),
                        encoding="utf-8",
                    )
                    paths.append(path)
            decision = evaluate_hard_gate(
                paths,
                ROOT / "governance" / "hard_gate_policy.json",
            )
            self.assertTrue(decision.approved, decision.failures)


if __name__ == "__main__":
    unittest.main()
