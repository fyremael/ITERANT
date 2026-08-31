import json
import tempfile
import unittest
from pathlib import Path

from phase_rlvr.observer import read_validation_jsonl, summarize_directory, summarize_step


class ObserverTests(unittest.TestCase):
    def test_step_summary(self):
        rows = []
        for output, score in [("a", 1), ("b", 0), ("c", 0), ("d", 0)]:
            rows.append({"input": "p1", "output": output, "score": score, "step": 10})
        for output, score in [("e", 0), ("f", 0), ("g", 0), ("h", 0)]:
            rows.append({"input": "p2", "output": output, "score": score, "step": 10})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "10.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary = summarize_step(read_validation_jsonl(path), k=2)

        self.assertEqual(summary.prompts, 2)
        self.assertEqual(summary.samples, 8)
        self.assertAlmostEqual(summary.pass1, 0.125)
        self.assertAlmostEqual(summary.passk, 0.25)
        self.assertAlmostEqual(summary.all_fail_rate, 0.5)
        self.assertAlmostEqual(summary.correct_output_uniqueness, 1.0)

    def test_canonical_uniqueness_is_conditioned_on_correct(self):
        rows = [
            {"input": "p", "output": "x  =  1", "score": 1, "step": 2},
            {"input": "p", "output": "x = 1", "score": 1, "step": 2},
            {"input": "p", "output": "wrong", "score": 0, "step": 2},
            {"input": "p", "output": "other wrong", "score": 0, "step": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary = summarize_step(read_validation_jsonl(path), k=2)
        self.assertAlmostEqual(summary.correct_output_uniqueness, 0.5)

    def test_directory_orders_numeric_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (10, 2):
                rows = [
                    {"input": "p", "output": "ok", "score": 1, "step": step},
                    {"input": "p", "output": "no", "score": 0, "step": step},
                ]
                (root / f"{step}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            summaries = summarize_directory(root, k=1)
        self.assertEqual([row.step for row in summaries], [2, 10])

    def test_rejects_insufficient_samples_for_k(self):
        rows = [
            {"input": "p", "output": "ok", "score": 1, "step": 1},
            {"input": "p", "output": "no", "score": 0, "step": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize_step(read_validation_jsonl(path), k=4)


if __name__ == "__main__":
    unittest.main()
