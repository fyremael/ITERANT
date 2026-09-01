from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Iterable

ROOT_DEFAULT = Path("/content/o10")
SOURCE_ARCHIVE_DEFAULT = Path("/content/phase_rlvr_source.tar.gz")
MANIFEST_DEFAULT = Path("/content/o10_manifest.json")
RESUME_ARCHIVE_DEFAULT = Path("/content/o10_resume.tar.gz")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str | None:
    if not root.exists():
        return None
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(path)))
    return h.hexdigest()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with return code {proc.returncode}: {cmd}")


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        base = dest.resolve()
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if base not in target.parents and target != base:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(dest)


def load_phase_package(source_root: Path) -> None:
    package_root = source_root / "campaigns" / "phase-rlvr" / "src"
    if not package_root.is_dir():
        raise RuntimeError(f"PHASE-RLVR package missing from source payload: {package_root}")
    sys.path.insert(0, str(package_root))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_generation_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("*.jsonl"), key=lambda p: int(p.stem))


def count_generated_tokens(root: Path, tokenizer) -> tuple[int, int]:
    samples = 0
    tokens = 0
    if not root.is_dir():
        return samples, tokens
    for path in iter_generation_files(root):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                output = str(row.get("output", ""))
                samples += 1
                tokens += len(tokenizer.encode(output, add_special_tokens=False))
    return samples, tokens


def package_tar(output: Path, root: Path, members: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as tf:
        for path in members:
            if path.exists():
                tf.add(path, arcname=path.relative_to(root))


def _parse_runtime_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    if argv is not None:
        return parser.parse_args(argv)
    args, unknown = parser.parse_known_args()
    if not unknown:
        return args
    if len(unknown) == 2 and unknown[0] == "-f":
        kernel_file = Path(unknown[1])
        if kernel_file.name.startswith("kernel-") and kernel_file.suffix == ".json":
            return args
    parser.error(f"unrecognized arguments: {' '.join(unknown)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute exactly one governed O10-T4 segment")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--source-archive", type=Path, default=SOURCE_ARCHIVE_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--resume-archive", type=Path, default=RESUME_ARCHIVE_DEFAULT)
    return _parse_runtime_args(parser, argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root = args.root
    source_root = root / "source"
    export_dir = root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    receipt: dict = {
        "status": "FAILED",
        "experiment_class": "observer_only",
        "controller_mutation_enabled": False,
        "fatal_error": None,
    }

    try:
        if not args.source_archive.is_file():
            raise RuntimeError(f"source archive missing: {args.source_archive}")
        if source_root.exists():
            shutil.rmtree(source_root)
        safe_extract(args.source_archive, source_root)
        load_phase_package(source_root)

        from phase_rlvr.o10_t4 import (
            MODEL_ID,
            MODEL_REVISION,
            build_verl_command,
            canonical_json_hash,
            segment_targets,
            validate_o10_t4_manifest,
        )
        from phase_rlvr.observer import summarize_directory

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate_o10_t4_manifest(manifest)

        if args.resume_archive.is_file() and not (root / "state.json").exists():
            safe_extract(args.resume_archive, root)

        state_path = root / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        completed = int(state.get("completed_target_step", 0))
        targets = segment_targets(manifest)
        pending = [target for target in targets if target > completed]
        if not pending:
            raise RuntimeError(f"O10-T4 already complete at step {completed}")
        target_step = pending[0]

        verl_dir = root / "verl"
        observed_verl = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=verl_dir, text=True
        ).strip()
        if observed_verl != manifest["verl_commit"]:
            raise RuntimeError(f"verl checkout drift before segment: expected {manifest['verl_commit']}, observed {observed_verl}")

        data_dir = root / "data"
        if not (data_dir / "dataset_receipt.json").exists():
            prep = source_root / "campaigns" / "phase-rlvr" / "colab" / "prepare_gsm8k.py"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(source_root / "campaigns" / "phase-rlvr" / "src")
            run(
                [
                    sys.executable,
                    str(prep),
                    "--output-dir",
                    str(data_dir),
                    "--train-limit",
                    str(manifest["dataset"]["train_limit"]),
                    "--validation-limit",
                    str(manifest["dataset"]["validation_limit"]),
                    "--seed",
                    str(manifest["seeds"]["data"]),
                ],
                env=env,
            )

        from huggingface_hub import snapshot_download

        model_path = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(root / "hf-cache"),
            )
        )

        cmd = build_verl_command(
            manifest,
            target_step=target_step,
            model_path=str(model_path),
            train_path=str(data_dir / "train.parquet"),
            validation_path=str(data_dir / "validation.parquet"),
            work_dir=str(root),
        )

        train_log = root / "logs" / f"train-to-{target_step}.log"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(source_root / "campaigns" / "phase-rlvr" / "src"),
                str(verl_dir),
                env.get("PYTHONPATH", ""),
            ]
        )
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["VLLM_USE_V1"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        run(cmd, cwd=verl_dir, env=env, log=train_log)

        observer_started = time.time()
        summaries = summarize_directory(
            root / "validation",
            k=int(manifest["recipe"]["observer_k"]),
        )
        observer_seconds = time.time() - observer_started
        observer_path = root / "observer_summary.jsonl"
        observer_path.write_text(
            "".join(json.dumps(summary.__dict__, sort_keys=True) + "\n" for summary in summaries),
            encoding="utf-8",
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        validation_samples, validation_tokens = count_generated_tokens(root / "validation", tokenizer)
        rollout_samples, rollout_tokens = count_generated_tokens(root / "rollouts", tokenizer)

        state = {
            "completed_target_step": target_step,
            "segment_index": targets.index(target_step),
            "total_steps": int(manifest["recipe"]["total_steps"]),
        }
        write_json(state_path, state)

        latest_checkpoint = root / "checkpoints"
        receipt.update(
            {
                "status": "GREEN_OBSERVER",
                "run_id": manifest["run_id"],
                "target_step": target_step,
                "segment_index": state["segment_index"],
                "iterant_head": manifest["iterant_head"],
                "verl_commit": observed_verl,
                "manifest_sha256": canonical_json_hash(manifest),
                "source_archive_sha256": sha256_file(args.source_archive),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset_id": manifest["dataset"]["id"],
                "dataset_revision": manifest["dataset"]["revision"],
                "fixed_recipe_hash": manifest["fixed_recipe_hash"],
                "validation_samples": validation_samples,
                "validation_generated_tokens": validation_tokens,
                "rollout_samples": rollout_samples,
                "rollout_generated_tokens": rollout_tokens,
                "observer_seconds": observer_seconds,
                "checkpoint_tree_sha256": sha256_tree(latest_checkpoint),
                "elapsed_seconds": time.time() - started,
            }
        )
    except Exception as exc:
        receipt["fatal_error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        receipt["elapsed_seconds"] = time.time() - started

    receipt_path = root / "segment_receipt.json"
    write_json(receipt_path, receipt)

    evidence_members = [
        args.manifest if args.manifest.is_relative_to(root) else root / "missing-manifest",
        root / "bootstrap_receipt.json",
        root / "dataset_receipt.json",
        root / "data" / "dataset_receipt.json",
        root / "state.json",
        root / "segment_receipt.json",
        root / "observer_summary.jsonl",
        root / "validation",
        root / "rollouts",
        root / "logs",
    ]
    # The manifest is uploaded outside root; copy it into evidence first.
    manifest_copy = root / "o10_manifest.json"
    if args.manifest.is_file():
        shutil.copy2(args.manifest, manifest_copy)
    evidence_members[0] = manifest_copy

    evidence_tar = export_dir / "segment_latest.tar.gz"
    package_tar(evidence_tar, root, evidence_members)

    resume_tar = export_dir / "resume_latest.tar.gz"
    package_tar(
        resume_tar,
        root,
        [root / "state.json", root / "checkpoints"],
    )

    write_json(
        export_dir / "export_receipt.json",
        {
            "segment_status": receipt["status"],
            "evidence_sha256": sha256_file(evidence_tar),
            "resume_sha256": sha256_file(resume_tar),
        },
    )

    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "GREEN_OBSERVER" else 1


if __name__ == "__main__":
    raise SystemExit(main())
