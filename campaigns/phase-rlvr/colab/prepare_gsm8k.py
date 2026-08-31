from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from phase_rlvr.o10_t4 import DATASET_ID, DATASET_REVISION


def extract_solution(solution_str: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if match is None:
        raise ValueError("GSM8K answer does not contain governed #### final answer")
    return match.group(1).replace(",", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the pinned bounded GSM8K panel for O10-T4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--validation-limit", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    import datasets

    dataset = datasets.load_dataset(DATASET_ID, "main", revision=DATASET_REVISION)
    train = dataset["train"].shuffle(seed=args.seed).select(range(min(args.train_limit, len(dataset["train"]))))
    test = dataset["test"].select(range(min(args.validation_limit, len(dataset["test"]))))

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def convert(split: str):
        def fn(example, idx):
            question_raw = example["question"]
            answer_raw = example["answer"]
            return {
                "data_source": DATASET_ID,
                "prompt": [{"role": "user", "content": question_raw + " " + instruction}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(answer_raw)},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "dataset_revision": DATASET_REVISION,
                },
            }
        return fn

    train = train.map(convert("train"), with_indices=True, remove_columns=train.column_names)
    test = test.map(convert("test"), with_indices=True, remove_columns=test.column_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    train.to_parquet(train_path)
    test.to_parquet(validation_path)

    receipt = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "train_limit": len(train),
        "validation_limit": len(test),
        "data_seed": args.seed,
        "train_fingerprint": train._fingerprint,
        "validation_fingerprint": test._fingerprint,
    }
    (args.output_dir / "dataset_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
