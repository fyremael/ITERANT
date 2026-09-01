from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VERL_REPOSITORY = "https://github.com/verl-project/verl.git"
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "a8402b897b382048aed4a75738da3c83e908a7f9"

EXPERIMENT_CLASS = "observer_only"
HARDWARE_CLASS = "T4"
ROLLOUT_BACKEND = "vllm"
PRECISION_POLICY = "fp16"

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "experiment_class",
    "iterant_head",
    "verl_commit",
    "model",
    "dataset",
    "seeds",
    "hardware",
    "recipe",
    "controller_mutation_enabled",
    "fixed_recipe_hash",
}


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha(value: Any, *, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc


def validate_o10_t4_manifest(payload: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL - payload.keys()
    if missing:
        raise ValueError(f"missing required O10-T4 fields: {sorted(missing)}")

    if payload["schema_version"] != 1:
        raise ValueError("O10-T4 schema_version must equal 1")
    if payload["experiment_class"] != EXPERIMENT_CLASS:
        raise ValueError(f"experiment_class must equal {EXPERIMENT_CLASS!r}")
    if payload["controller_mutation_enabled"] is not False:
        raise ValueError("O10-T4 forbids any controller mutation path")
    _require_sha(payload["iterant_head"], length=40, field="iterant_head")
    if payload["verl_commit"] != VERL_COMMIT:
        raise ValueError(f"verl_commit must equal governed pin {VERL_COMMIT}")
    expected_recipe_hash = canonical_json_hash(payload["recipe"])
    if payload["fixed_recipe_hash"] != expected_recipe_hash:
        raise ValueError(
            f"fixed_recipe_hash mismatch: declared {payload['fixed_recipe_hash']}, computed {expected_recipe_hash}"
        )

    model = payload["model"]
    if model.get("id") != MODEL_ID:
        raise ValueError(f"O10-T4 model.id must equal {MODEL_ID}")
    if model.get("revision") != MODEL_REVISION:
        raise ValueError(f"O10-T4 model.revision must equal {MODEL_REVISION}")
    if model.get("tokenizer_id") != MODEL_ID:
        raise ValueError("O10-T4 tokenizer must be pinned to the model repository")
    if model.get("tokenizer_revision") != MODEL_REVISION:
        raise ValueError("O10-T4 tokenizer revision must equal the model revision")

    dataset = payload["dataset"]
    if dataset.get("id") != DATASET_ID:
        raise ValueError(f"O10-T4 dataset.id must equal {DATASET_ID}")
    if dataset.get("revision") != DATASET_REVISION:
        raise ValueError(f"O10-T4 dataset.revision must equal {DATASET_REVISION}")
    if int(dataset.get("train_limit", 0)) < 64:
        raise ValueError("O10-T4 train_limit must be >= 64")
    if not 8 <= int(dataset.get("validation_limit", 0)) <= 128:
        raise ValueError("O10-T4 validation_limit must be in [8, 128]")

    seeds = payload["seeds"]
    for name in ("training", "sampling", "data"):
        if not isinstance(seeds.get(name), int) or seeds[name] < 0:
            raise ValueError(f"seeds.{name} must be a non-negative integer")

    hardware = payload["hardware"]
    if hardware.get("class") != HARDWARE_CLASS:
        raise ValueError("O10-T4 hardware.class must equal 'T4'")
    if int(hardware.get("min_gpu_memory_mib", 0)) < 14000:
        raise ValueError("O10-T4 requires at least 14,000 MiB declared GPU memory")
    if hardware.get("precision_policy") != PRECISION_POLICY:
        raise ValueError("O10-T4 precision_policy must be fp16")
    if hardware.get("rollout_backend") != ROLLOUT_BACKEND:
        raise ValueError("O10-T4 rollout_backend must be vllm")

    recipe = payload["recipe"]
    if recipe.get("algorithm") != "grpo":
        raise ValueError("O10-T4 recipe.algorithm must be grpo")
    if recipe.get("adaptation") != "lora":
        raise ValueError("O10-T4 requires LoRA adaptation to bound memory")
    if not 1 <= int(recipe.get("lora_rank", 0)) <= 64:
        raise ValueError("O10-T4 lora_rank must be in [1, 64]")
    if int(recipe.get("rollout_n", 0)) != 4:
        raise ValueError("O10-T4 rollout_n is governed at 4")
    if int(recipe.get("validation_n", 0)) != 8:
        raise ValueError("O10-T4 validation_n is governed at 8")
    if int(recipe.get("observer_k", 0)) > int(recipe["validation_n"]):
        raise ValueError("observer_k cannot exceed validation_n")
    if not 1 <= int(recipe.get("train_batch_size", 0)) <= 16:
        raise ValueError("O10-T4 train_batch_size must be in [1, 16]")
    if not 64 <= int(recipe.get("max_prompt_length", 0)) <= 512:
        raise ValueError("O10-T4 max_prompt_length must be in [64, 512]")
    if not 64 <= int(recipe.get("max_response_length", 0)) <= 512:
        raise ValueError("O10-T4 max_response_length must be in [64, 512]")

    segment_steps = int(recipe.get("segment_steps", 0))
    total_steps = int(recipe.get("total_steps", 0))
    if segment_steps < 1 or total_steps < segment_steps:
        raise ValueError("O10-T4 requires positive segment_steps <= total_steps")
    if total_steps % segment_steps != 0:
        raise ValueError("O10-T4 total_steps must be divisible by segment_steps")
    if int(recipe.get("test_freq", 0)) != segment_steps:
        raise ValueError("O10-T4 test_freq must equal segment_steps")
    if int(recipe.get("save_freq", 0)) != segment_steps:
        raise ValueError("O10-T4 save_freq must equal segment_steps")

    if recipe.get("use_kl_loss") is not False:
        raise ValueError("O10-T4 fixed GRPO baseline has actor KL loss disabled")
    if recipe.get("norm_adv_by_std_in_grpo") is not True:
        raise ValueError("O10-T4 fixed GRPO baseline requires standard GRPO advantage normalization")


def segment_targets(payload: dict[str, Any]) -> list[int]:
    validate_o10_t4_manifest(payload)
    recipe = payload["recipe"]
    step = int(recipe["segment_steps"])
    total = int(recipe["total_steps"])
    return list(range(step, total + 1, step))


def build_verl_command(
    payload: dict[str, Any],
    *,
    target_step: int,
    model_path: str,
    train_path: str,
    validation_path: str,
    work_dir: str,
) -> list[str]:
    """Build the fixed veRL command. No PHASE-RLVR controller values enter this command."""
    validate_o10_t4_manifest(payload)
    targets = segment_targets(payload)
    if target_step not in targets:
        raise ValueError(f"target_step must be one of governed segment targets {targets}")

    recipe = payload["recipe"]
    seeds = payload["seeds"]
    root = Path(work_dir)
    checkpoints = root / "checkpoints"
    validation = root / "validation"
    rollouts = root / "rollouts"

    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo_sync",
        f"data.train_files={train_path}",
        f"data.val_files={validation_path}",
        f"data.train_batch_size={int(recipe['train_batch_size'])}",
        f"data.max_prompt_length={int(recipe['max_prompt_length'])}",
        f"data.max_response_length={int(recipe['max_response_length'])}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.model.lora_rank={int(recipe['lora_rank'])}",
        f"actor_rollout_ref.model.lora_alpha={int(recipe['lora_alpha'])}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.dtype=float16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(recipe['train_batch_size']) * int(recipe['rollout_n'])}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048",
        f"actor_rollout_ref.actor.optim.lr={recipe['learning_rate']}",
        f"actor_rollout_ref.actor.use_kl_loss={str(bool(recipe['use_kl_loss'])).lower()}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={int(recipe['rollout_n'])}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.dtype=float16",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={recipe['gpu_memory_utilization']}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048",
        "actor_rollout_ref.rollout.max_num_seqs=64",
        "actor_rollout_ref.rollout.enforce_eager=True",
        f"actor_rollout_ref.rollout.val_kwargs.n={int(recipe['validation_n'])}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={recipe['validation_temperature']}",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "algorithm.adv_estimator=grpo",
        f"algorithm.norm_adv_by_std_in_grpo={str(bool(recipe['norm_adv_by_std_in_grpo'])).lower()}",
        "algorithm.use_kl_in_reward=False",
        "trainer.logger=[console]",
        "trainer.project_name=phase-rlvr",
        f"trainer.experiment_name={payload['run_id']}",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"trainer.val_before_train={str(target_step == targets[0]).lower()}",
        f"trainer.test_freq={int(recipe['test_freq'])}",
        f"trainer.save_freq={int(recipe['save_freq'])}",
        f"trainer.total_training_steps={target_step}",
        f"trainer.default_local_dir={checkpoints}",
        "trainer.resume_mode=auto",
        f"trainer.validation_data_dir={validation}",
        f"trainer.rollout_data_dir={rollouts}",
        "trainer.max_actor_ckpt_to_keep=2",
        f"actor_rollout_ref.actor.data_loader_seed={int(seeds['training'])}",
        f"actor_rollout_ref.actor.fsdp_config.seed={int(seeds['training'])}",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.seed={int(seeds['sampling'])}",
    ]
    return cmd


def default_manifest(iterant_head: str) -> dict[str, Any]:
    _require_sha(iterant_head, length=40, field="iterant_head")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "PHASE-RLVR-WP01-O10-T4",
        "experiment_class": EXPERIMENT_CLASS,
        "iterant_head": iterant_head,
        "verl_commit": VERL_COMMIT,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "subset": "main",
            "train_limit": 512,
            "validation_limit": 32,
        },
        "seeds": {"training": 17, "sampling": 23, "data": 31},
        "hardware": {
            "class": HARDWARE_CLASS,
            "min_gpu_memory_mib": 14000,
            "precision_policy": PRECISION_POLICY,
            "rollout_backend": ROLLOUT_BACKEND,
        },
        "recipe": {
            "algorithm": "grpo",
            "adaptation": "lora",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 1e-6,
            "train_batch_size": 8,
            "rollout_n": 4,
            "validation_n": 8,
            "observer_k": 4,
            "validation_temperature": 0.7,
            "max_prompt_length": 256,
            "max_response_length": 256,
            "gpu_memory_utilization": 0.35,
            "segment_steps": 10,
            "total_steps": 40,
            "test_freq": 10,
            "save_freq": 10,
            "use_kl_loss": False,
            "norm_adv_by_std_in_grpo": True,
        },
        "controller_mutation_enabled": False,
    }
    payload["fixed_recipe_hash"] = canonical_json_hash(payload["recipe"])
    validate_o10_t4_manifest(payload)
    return payload
