import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_admission_module():
    path = ROOT / "scripts" / "check_o10_export.py"
    spec = importlib.util.spec_from_file_location("phase_rlvr_test_o10_admission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class O10FailureDiagnosticsTests(unittest.TestCase):
    def test_failed_evidence_surfaces_inner_training_log_tail(self):
        module = load_admission_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            receipt = tmp / "segment_receipt.json"
            receipt.write_text(
                json.dumps({"status": "FAILED", "fatal_error": "RuntimeError: trainer exited 1"}),
                encoding="utf-8",
            )
            train_log = tmp / "train-to-10.log"
            train_log.write_text(
                "preamble\nINNER_VERL_ROOT_CAUSE\nfinal-line\n",
                encoding="utf-8",
            )
            evidence = tmp / "segment_evidence.tar.gz"
            with tarfile.open(evidence, "w:gz") as tf:
                tf.add(receipt, arcname="segment_receipt.json")
                tf.add(train_log, arcname="logs/train-to-10.log")

            diagnostics = module.failure_diagnostics(evidence, tail_lines=2)
            self.assertIn("remote fatal_error: RuntimeError: trainer exited 1", diagnostics)
            self.assertIn("INNER_VERL_ROOT_CAUSE", diagnostics)
            self.assertIn("final-line", diagnostics)
            self.assertNotIn("preamble", diagnostics)


if __name__ == "__main__":
    unittest.main()
