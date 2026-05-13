from __future__ import annotations

import io
import inspect
import json
import time
import threading
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import sys

from src.configs import load_experiment_config
from src.evaluate import build_metrics_record, build_sample_preview
from src.experiment_store import (
    append_experiment_row,
    append_history_rows,
    append_metrics_row,
    build_experiment_id,
    export_named_experiment_result,
    export_named_history_result,
    export_named_metrics_result,
    write_run_artifacts,
)
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


def _build_prompt_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["training"]
    max_prompt_length = training.get("max_prompt_length")
    if max_prompt_length is None:
        return {}
    return {"max_prompt_length": max_prompt_length}


def _filter_supported_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    supported: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in signature.parameters and value is not None:
            supported[key] = value
    return supported


class _TeeStream:
    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, text: str) -> int:
        primary_written = self._primary.write(text)
        self._secondary.write(text)
        return primary_written

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._primary, "isatty", None)
        return bool(isatty()) if callable(isatty) else False


@dataclass
class _RunHeartbeat:
    run_dir: Path

    def update(self, *, status: str, step: int | None = None, message: str | None = None) -> None:
        payload = {
            "status": status,
            "step": step,
            "message": message,
            "timestamp": time.time(),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "heartbeat.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
        from transformers import AutoTokenizer, TrainerCallback  # type: ignore
        from trl import DPOConfig, DPOTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Phase 5 training requires torch, transformers, datasets, accelerate, and trl."
        ) from exc
    return torch, Dataset, AutoTokenizer, TrainerCallback, DPOConfig, DPOTrainer


def _load_preference_dataset(dataset_path: Path):
    _, Dataset, _, _, _, _ = _import_training_stack()
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


def _build_history_rows(
    *,
    experiment_id: str,
    config_name: str,
    log_history: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        is_eval = any(str(key).startswith("eval_") for key in entry.keys())
        row = {
            "experiment_id": experiment_id,
            "config_name": config_name,
            "step": step,
            "epoch": entry.get("epoch", "NA"),
            "phase": "eval" if is_eval else "train",
            "loss": next(
                (
                    entry.get(key)
                    for key in ("eval_loss", "loss", "train_loss")
                    if entry.get(key) is not None
                ),
                "NA",
            ),
            "reward_margin": next(
                (
                    entry.get(key)
                    for key in ("eval_rewards/margins", "rewards/margins", "eval_reward_margin", "reward_margin")
                    if entry.get(key) is not None
                ),
                "NA",
            ),
            "win_rate": next(
                (
                    entry.get(key)
                    for key in ("eval_rewards/accuracies", "rewards/accuracies", "win_rate", "accuracy")
                    if entry.get(key) is not None
                ),
                "NA",
            ),
            "chosen_logprob": next(
                (
                    entry.get(key)
                    for key in ("eval_logps/chosen", "logps/chosen", "chosen_logprob", "eval_rewards/chosen", "rewards/chosen")
                    if entry.get(key) is not None
                ),
                "NA",
            ),
            "rejected_logprob": next(
                (
                    entry.get(key)
                    for key in ("eval_logps/rejected", "logps/rejected", "rejected_logprob", "eval_rewards/rejected", "rewards/rejected")
                    if entry.get(key) is not None
                ),
                "NA",
            ),
            "learning_rate": entry.get("learning_rate", "NA"),
        }
        rows.append(row)
    return rows


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

    torch, Dataset, AutoTokenizer, TrainerCallback, DPOConfig, DPOTrainer = _import_training_stack()
    precision = _select_precision(torch)
    train_path = resolve_train_dataset_path(spec, data_root)
    eval_path = resolve_eval_dataset_path(data_root)
    train_dataset = _load_preference_dataset(train_path)
    eval_dataset = _load_preference_dataset(eval_path)

    model_name = config["model"]["primary"]
    fallback_name = config["model"].get("fallback")
    logs = io.StringIO()
    heartbeat = _RunHeartbeat(run_dir)
    start = time.perf_counter()
    trainer = None
    status = "success"
    note = "trained with primary model"
    train_metrics: dict[str, Any] = {"train_loss": "NA"}
    eval_metrics: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    samples = build_sample_preview(load_json_records(train_path), limit=5)
    dpo_kwargs = build_dpo_config_kwargs(config, output_dir=run_dir)
    dpo_kwargs.update(precision)
    dpo_args = DPOConfig(**dpo_kwargs)

    class _HeartbeatCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
            heartbeat.update(status="training", step=0, message="training started")
            print("[heartbeat] training started", flush=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            heartbeat.update(status="training", step=int(getattr(state, "global_step", 0)), message="step update")
            if getattr(state, "global_step", 0) % 10 == 0:
                print(f"[heartbeat] step={int(state.global_step)}", flush=True)
            return control

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
            heartbeat.update(status="finished", step=int(getattr(state, "global_step", 0)), message="training ended")
            print("[heartbeat] training finished", flush=True)
            return control

    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(_TeeStream(sys.stdout, logs)))
            stack.enter_context(redirect_stderr(_TeeStream(sys.stderr, logs)))
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
                callbacks=[_HeartbeatCallback()],
                **_filter_supported_kwargs(DPOTrainer, _build_prompt_kwargs(config)),
            )
            trainer.train()
            train_metrics = _collect_training_metrics(trainer.state.log_history)
            history_rows = _build_history_rows(
                experiment_id=experiment_id,
                config_name=str(spec["config_name"]),
                log_history=trainer.state.log_history,
            )
            eval_metrics = trainer.evaluate()
            history_rows = _build_history_rows(
                experiment_id=experiment_id,
                config_name=str(spec["config_name"]),
                log_history=trainer.state.log_history,
            )
    except Exception as exc:
        if fallback_name and fallback_name != model_name:
            logs.write(f"\nPrimary model failed; retrying with fallback {fallback_name}\n")
            try:
                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(_TeeStream(sys.stdout, logs)))
                    stack.enter_context(redirect_stderr(_TeeStream(sys.stderr, logs)))
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
                        callbacks=[_HeartbeatCallback()],
                        **_filter_supported_kwargs(DPOTrainer, _build_prompt_kwargs(config)),
                    )
                    trainer.train()
                    train_metrics = _collect_training_metrics(trainer.state.log_history)
                    history_rows = _build_history_rows(
                        experiment_id=experiment_id,
                        config_name=str(spec["config_name"]),
                        log_history=trainer.state.log_history,
                    )
                    eval_metrics = trainer.evaluate()
                    history_rows = _build_history_rows(
                        experiment_id=experiment_id,
                        config_name=str(spec["config_name"]),
                        log_history=trainer.state.log_history,
                    )
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
        history_rows=history_rows,
        metrics=metrics,
        samples=samples,
        train_log=logs.getvalue(),
    )
    append_experiment_row(results_root / "experiments.tsv", summary)
    append_metrics_row(results_root / "metrics.csv", metrics)
    append_history_rows(results_root / "history.csv", history_rows)
    export_named_experiment_result(results_root, summary)
    export_named_metrics_result(results_root, metrics)
    export_named_history_result(results_root, str(spec["config_name"]), history_rows)
    return summary
