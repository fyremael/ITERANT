import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from phase_rlvr.o10_t4 import (
    DATASET_REVISION,
    MODEL_REVISION,
    VERL_COMMIT,
    build_verl_command,
    canonical_json_hash,
    default_manifest,
    segment_targets,
    validate_o10_t4_manifest,
)


class O10T4ContractTests(unittest.TestCase):
    def valid(self):
        return default_manifest("a" * 40)

    def test_default_manifest_is_governed_and_segmented(self):
        payload = self.valid()
        self.assertEqual(payload["verl_commit"], VERL_COMMIT)
        self.assertEqual(payload["model"]["revision"], MODEL_REVISION)
        self.assertEqual(payload["dataset"]["revision"], DATASET_REVISION)
        self.assertFalse(payload["controller_mutation_enabled"])
        self.assertEqual(segment_targets(payload), [10, 20, 30, 40])
        self.assertEqual(payload["fixed_recipe_hash"], canonical_json_hash(payload["recipe"]))

    def test_t4_rejects_bf16(self):
        payload = self.valid()
        payload["hardware"]["precision_policy"] = "bf16"
        with self.assertRaises(ValueError):
            validate_o10_t4_manifest(payload)

    def test_t4_rejects_non_lora(self):
        payload = self.valid()
        payload["recipe"]["adaptation"] = "full"
        with self.assertRaises(ValueError):
            validate_o10_t4_manifest(payload)

    def test_t4_rejects_controller_mutation(self):
        payload = self.valid()
        payload["controller_mutation_enabled"] = True
        with self.assertRaises(ValueError):
            validate_o10_t4_manifest(payload)

    def test_t4_rejects_unpinned_model(self):
        payload = self.valid()
        payload["model"]["revision"] = "main"
        with self.assertRaises(ValueError):
            validate_o10_t4_manifest(payload)

    def test_command_is_fixed_grpo_and_has_no_controller_path(self):
        payload = self.valid()
        cmd = build_verl_command(
            payload,
            target_step=10,
            model_path="/models/qwen",
            train_path="/data/train.parquet",
            validation_path="/data/validation.parquet",
            work_dir="/run/o10",
        )
        joined = "\n".join(cmd)
        self.assertIn("algorithm.adv_estimator=grpo", joined)
        self.assertIn("actor_rollout_ref.model.lora_rank=16", joined)
        self.assertIn("actor_rollout_ref.rollout.dtype=float16", joined)
        self.assertIn("trainer.total_training_steps=10", joined)
        self.assertIn("trainer.val_before_train=true", joined)
        self.assertIn("trainer.validation_data_dir=/run/o10/validation", joined)
        self.assertNotIn("phase_rlvr.controller", joined)
        self.assertNotIn("controller_mutation", joined)

    def test_resumed_segment_does_not_revalidate_starting_checkpoint(self):
        payload = self.valid()
        cmd = build_verl_command(
            payload,
            target_step=20,
            model_path="/models/qwen",
            train_path="/data/train.parquet",
            validation_path="/data/validation.parquet",
            work_dir="/run/o10",
        )
        self.assertIn("trainer.val_before_train=false", cmd)

    def test_command_rejects_non_segment_target(self):
        payload = self.valid()
        with self.assertRaises(ValueError):
            build_verl_command(
                payload,
                target_step=11,
                model_path="/models/qwen",
                train_path="/data/train.parquet",
                validation_path="/data/validation.parquet",
                work_dir="/run/o10",
            )

    def test_shell_runner_parses(self):
        script = ROOT / "scripts" / "colab_run_o10_t4.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_shell_runner_session_stamp_is_ascii(self):
        script = ROOT / "scripts" / "colab_run_o10_t4.sh"
        raw = script.read_bytes()
        text = raw.decode("ascii")
        self.assertIn('STAMP="$(LC_ALL=C date -u +%Y%m%dT%H%M%SZ)"', text)
        self.assertIn('SESSION="gcl-phase-rlvr-o10-${STAMP,,}-$$"', text)

    def test_hosted_evidence_directory_is_ignored(self):
        gitignore = ROOT.parents[1] / ".gitignore"
        entries = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
        self.assertIn("runs/hosted/", entries)

    def test_export_admission_accepts_green_segment(self):
        import hashlib
        import tarfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            evidence = tmp / "evidence.tar.gz"
            resume = tmp / "resume.tar.gz"
            export = tmp / "export.json"

            payloads = {
                "segment_receipt.json": {
                    "status": "GREEN_OBSERVER",
                    "controller_mutation_enabled": False,
                    "target_step": 10,
                    "iterant_head": "a" * 40,
                    "source_archive_sha256": "b" * 64,
                    "fixed_recipe_hash": self.valid()["fixed_recipe_hash"],
                },
                "o10_manifest.json": self.valid(),
                "state.json": {"completed_target_step": 10},
            }
            with tarfile.open(evidence, "w:gz") as tf:
                for name, payload in payloads.items():
                    p = tmp / name
                    p.write_text(json.dumps(payload), encoding="utf-8")
                    tf.add(p, arcname=name)
                summary = tmp / "observer_summary.jsonl"
                summary.write_text('{"step":10}\n', encoding="utf-8")
                tf.add(summary, arcname="observer_summary.jsonl")
            with tarfile.open(resume, "w:gz") as tf:
                state = tmp / "state.json"
                tf.add(state, arcname="state.json")
                ckpt = tmp / "checkpoint.bin"
                ckpt.write_bytes(b"checkpoint")
                tf.add(ckpt, arcname="checkpoints/global_step_10/actor/checkpoint.bin")

            def digest(p):
                return hashlib.sha256(p.read_bytes()).hexdigest()
            export.write_text(
                json.dumps(
                    {
                        "segment_status": "GREEN_OBSERVER",
                        "evidence_sha256": digest(evidence),
                        "resume_sha256": digest(resume),
                    }
                ),
                encoding="utf-8",
            )
            script = ROOT / "scripts" / "check_o10_export.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--evidence",
                    str(evidence),
                    "--resume",
                    str(resume),
                    "--export-receipt",
                    str(export),
                    "--expected-iterant-head",
                    "a" * 40,
                    "--expected-source-sha256",
                    "b" * 64,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
