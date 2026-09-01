from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

VERL_REPOSITORY = "https://github.com/verl-project/verl.git"
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
VLLM_VERSION = "0.11.0"


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def gpu_info() -> dict:
    query = run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"O10-T4 requires exactly one visible GPU; observed {len(lines)}")
    name, memory_mib, driver = [part.strip() for part in lines[0].split(",", maxsplit=2)]
    if "T4" not in name:
        raise RuntimeError(f"O10-T4 requires a Tesla T4; observed {name!r}")
    if int(memory_mib) < 14000:
        raise RuntimeError(f"O10-T4 requires >= 14000 MiB; observed {memory_mib}")
    return {"name": name, "memory_total_mib": int(memory_mib), "driver": driver}


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


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
    parser = argparse.ArgumentParser(description="Bootstrap governed veRL v0.8.0 on the O10-T4 Colab worker")
    parser.add_argument("--root", type=Path, default=Path("/content/o10"))
    parser.add_argument("--skip-install", action="store_true", help="Validate checkout/runtime without pip installation")
    return _parse_runtime_args(parser, argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if sys.version_info < (3, 10):
        raise RuntimeError(f"veRL v0.8.0 requires Python >=3.10; observed {platform.python_version()}")

    args.root.mkdir(parents=True, exist_ok=True)
    info = gpu_info()
    verl_dir = args.root / "verl"

    if verl_dir.exists() and not (verl_dir / ".git").exists():
        raise RuntimeError(f"{verl_dir} exists but is not a git checkout")
    if not verl_dir.exists():
        run(["git", "clone", "--filter=blob:none", VERL_REPOSITORY, str(verl_dir)])
    run(["git", "fetch", "--depth=1", "origin", VERL_COMMIT], cwd=verl_dir)
    run(["git", "checkout", "--detach", VERL_COMMIT], cwd=verl_dir)
    observed = run(["git", "rev-parse", "HEAD"], cwd=verl_dir)
    if observed != VERL_COMMIT:
        raise RuntimeError(f"veRL checkout drift: expected {VERL_COMMIT}, observed {observed}")

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
        # v0.8.0 upstream installation pins vLLM 0.11.0. Install it explicitly,
        # but omit upstream FlashAttention because Turing/T4 is outside FA2 support.
        run([sys.executable, "-m", "pip", "install", f"vllm=={VLLM_VERSION}", "--extra-index-url", "https://download.pytorch.org/whl/cu128"])
        run([sys.executable, "-m", "pip", "install", "-e", ".[math]"], cwd=verl_dir)

    receipt = {
        "status": "GREEN_BOOTSTRAP",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": info,
        "verl_repository": VERL_REPOSITORY,
        "verl_commit": observed,
        "vllm_governed_version": VLLM_VERSION,
        "packages": {
            name: package_version(name)
            for name in ("torch", "vllm", "ray", "transformers", "datasets", "peft", "verl")
        },
    }
    receipt_path = args.root / "bootstrap_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
