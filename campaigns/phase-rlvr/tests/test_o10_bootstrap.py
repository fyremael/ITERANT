import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "colab" / "bootstrap_verl.py"


def load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("phase_rlvr_test_bootstrap_verl", BOOTSTRAP)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {BOOTSTRAP}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class O10BootstrapContractTests(unittest.TestCase):
    def test_sync_trainer_transferqueue_dependency_is_governed(self):
        module = load_bootstrap_module()
        self.assertEqual(module.TRANSFERQUEUE_VERSION, "0.1.7")

        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn('f"TransferQueue=={TRANSFERQUEUE_VERSION}"', source)
        self.assertIn('package_version("TransferQueue")', source)
        self.assertIn('"transferqueue_governed_version": TRANSFERQUEUE_VERSION', source)


if __name__ == "__main__":
    unittest.main()
