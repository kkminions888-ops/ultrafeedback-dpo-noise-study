from __future__ import annotations

import io
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Mapping

from src.configs import load_experiment_config
from src.evaluate import build_metrics_record, build_sample_preview
from src.experiment_store import append_experiment_row, append_metrics_row, build_experiment_id, write_run_artifacts
from src.load_data import load_json_records


def resolve_train_dataset_path(spec: Mapping[str, Any], data_root: Path) -> Path:
    noise_type = str(spec["noise_type"])
    noise_rate = float(spec.get("noise_rate", 0.0))
    if noise_type == "clean" or noise_rate == 0.0:
        return data_root / "noisy" / "clean.jsonl"
    rate_suffix = int(round(noise_rate * 100))
    return data_root / "noisy" / f"{noise_type}_{rate_suffix:02d}.jsonl"


def resolve_eval_dataset_path(data_root: Path) -> Path:
    return data_root / "processed" / "eval_clean.jsonl"


def build_dpo_config_kwargs(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    training = config["training"]
    compute = config.get("compute", {})
    return {
        "output_dir": output_dir.as_posix(),
        "per_device_train_batch_size": training["batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "max_steps": training["max_steps"],
        "max_length": training["max_length"],
        "max_prompt_length": training.get("max_prompt_length"),
        "seed": training["seed"],
        "report_to": "none",
        "do_train": True,
        "do_eval": True,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": training["max_steps"],
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "use_cpu": bool(compute.get("local_cpu_only", False)),
        "bf16": False,
        "fp16": False,
    }


def _select_precision(torch: Any) -> dict[str, bool]:
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16_supported = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    return {"bf16": bf16_supported, "fp16": not bf16_supported}


def _import_training_stack():
    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        from trl import DPOConfig, DPOTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Phase 5 training requires torch, transformers, datasets, accelerate, and trl."
        ) from exc
    return torch, Dataset, AutoTokenizer, DPOConfig, DPOTrainer


def _load_preference_dataset(dataset_path: Path):
    _, Dataset, _, _, _ = _import_training_stack()
    records = load_json_records(dataset_path)
    return Dataset.from_list(records)


def _collect_training_metrics(log_history: list[Mapping[str, Any]]) -> dict[str, Any]:
    train_loss = "NA"
    eval_loss = "NA"
    reward_margin = "NA"
    chosen_logprob = "NA"
    rejected_logprob = "NA"
    win_rate = "NA"
    for entry in reversed(log_history):
        if train_loss == "NA":
            train_loss = next(
                (
                    entry.get(key)
                    for key in ("train_loss", "loss")
                    if entry.get(key) is not None
                ),
                "NA",
            )
        if eval_loss == "NA":
            eval_loss = next(
                (
                    entry.get(key)
                    for key in ("eval_loss", "loss")
                    if entry.get(key) is not None
                ),
                "NA",
            )
        if reward_margin == "NA":
            reward_margin = next(
                (
                    entry.get(key)
                    for key in ("reward_margin", "rewards/margins", "eval_reward_margin", "eval_rewards/margins")
                    if entry.get(key) is not None
                ),
                "NA",
            )
        if chosen_logprob == "NA":
            chosen_logprob = next(
                (
                    entry.get(key)
                    for key in ("chosen_logprob", "logps/chosen", "eval_logps/chosen", "rewards/chosen", "eval_rewards/chosen")
                    if entry.get(key) is not None
                ),
                "NA",
            )
        if rejected_logprob == "NA":
            rejected_logprob = next(
                (
                    entry.get(key)
                    for key in ("rejected_logprob", "logps/rejected", "eval_logps/rejected", "rewards/rejected", "eval_rewards/rejected")
                    if entry.get(key) is not None
                ),
                "NA",
            )
        if win_rate == "NA":
            win_rate = next(
                (
                    entry.get(key)
                    for key in ("win_rate", "rewards/accuracies", "eval_rewards/accuracies", "accuracy")
                    if entry.get(key) is not None
                ),
                "NA",
            )
    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "reward_margin": reward_margin,
        "chosen_logprob": chosen_logprob,
        "rejected_logprob": rejected_logprob,
        "win_rate": win_rate,
    }


def run_experiment(
    spec: Mapping[str, Any],
    *,
    results_root: Path,
    data_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(spec["config_path"])
    config = load_experiment_config(config_path)
    experiment_id = build_experiment_id(spec)
    run_dir = results_root / "runs" / experiment_id

    if resume and (run_dir / "metrics.json").exists():
        return {"experiment_id": experiment_id, "status": "skipped", **spec}

    torch, Dataset, AutoTokenizer, DPOConfig, DPOTrainer = _import_training_stack()
    precision = _select_precision(torch)
    train_path = resolve_train_dataset_path(spec, data_root)
    eval_path = resolve_eval_dataset_path(data_root)
    train_dataset = _load_preference_dataset(train_path)
    eval_dataset = _load_preference_dataset(eval_path)

    model_name = config["model"]["primary"]
    fallback_name = config["model"].get("fallback")
    logs = io.StringIO()
    start = time.perf_counter()
    trainer = None
    status = "success"
    note = "trained with primary model"
    train_metrics: dict[str, Any] = {"train_loss": "NA"}
    eval_metrics: dict[str, Any] = {}
    samples = build_sample_preview(load_json_records(train_path), limit=5)
    dpo_kwargs = build_dpo_config_kwargs(config, output_dir=run_dir)
    dpo_kwargs.update(precision)
    dpo_args = DPOConfig(**dpo_kwargs)

    try:
        with redirect_stdout(logs), redirect_stderr(logs):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            trainer = DPOTrainer(
                model=model_name,
                args=dpo_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
            trainer.train()
            train_metrics = _collect_training_metrics(trainer.state.log_history)
            eval_metrics = trainer.evaluate()
    except Exception as exc:
        if fallback_name and fallback_name != model_name:
            logs.write(f"\nPrimary model failed; retrying with fallback {fallback_name}\n")
            try:
                with redirect_stdout(logs), redirect_stderr(logs):
                    tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                    tokenizer.padding_side = "left"
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    trainer = DPOTrainer(
                        model=fallback_name,
                        args=dpo_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        processing_class=tokenizer,
                    )
                    trainer.train()
                    train_metrics = _collect_training_metrics(trainer.state.log_history)
                    eval_metrics = trainer.evaluate()
                    model_name = fallback_name
                    note = f"trained with fallback model after primary failure: {exc}"
            except Exception as fallback_exc:
                status = "crash"
                note = f"primary failure: {exc}; fallback failure: {fallback_exc}"
        else:
            status = "crash"
            note = f"training failed: {exc}"

    runtime_minutes = round((time.perf_counter() - start) / 60.0, 4)
    metrics = build_metrics_record(
        experiment_id=experiment_id,
        config_name=str(spec["config_name"]),
        noise_type=str(spec["noise_type"]),
        noise_rate=float(spec["noise_rate"]),
        seed=int(spec["seed"]),
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        runtime_minutes=runtime_minutes,
        status=status,
    )
    summary = {
        "experiment_id": experiment_id,
        "config_name": spec["config_name"],
        "version": spec["version"],
        "noise_type": spec["noise_type"],
        "noise_rate": spec["noise_rate"],
        "seed": spec["seed"],
        "model_name": model_name,
        "dataset_name": spec["dataset_name"],
        "train_size": spec["train_size"],
        "eval_size": spec["eval_size"],
        "max_steps": spec["max_steps"],
        "status": status,
        "primary_metric": metrics.get("reward_margin", "NA"),
        "notes": note,
    }
    write_run_artifacts(
        run_dir,
        config=config,
        metrics=metrics,
        samples=samples,
        train_log=logs.getvalue(),
    )
    append_experiment_row(results_root / "experiments.tsv", summary)
    append_metrics_row(results_root / "metrics.csv", metrics)
    return summary
