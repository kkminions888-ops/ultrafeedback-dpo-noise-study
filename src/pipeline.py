from __future__ import annotations

from pathlib import Path
from typing import Any

from src.configs import list_experiment_configs, load_experiment_config
from src.experiment_store import build_experiment_id


def build_experiment_spec(config_path: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    experiment = config["experiment"]
    return {
        "config_name": config_path.stem,
        "version": experiment["version"],
        "noise_type": experiment["noise_type"],
        "noise_rate": experiment["noise_rate"],
        "seed": experiment["seed"],
        "model_name": config["model"]["primary"],
        "dataset_name": config["dataset"]["name"],
        "train_size": config["dataset"]["train_size"],
        "eval_size": config["dataset"]["eval_size"],
        "max_steps": config["training"]["max_steps"],
        "config_path": config_path.as_posix(),
    }


def should_skip_experiment(run_dir: Path, *, resume: bool) -> bool:
    return resume and (run_dir / "metrics.json").exists()


def resolve_experiment_plan(plan_path: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for config_path in list_experiment_configs(plan_path):
        plan.append(build_experiment_spec(config_path))
    return plan


def run_batch(
    plan_path: Path,
    *,
    resume: bool,
    results_root: Path,
    data_root: Path | None = None,
    execute: bool = False,
    experiment_runner: Any | None = None,
) -> list[dict[str, Any]]:
    scheduled: list[dict[str, Any]] = []
    runner = experiment_runner
    if execute and runner is None:
        from src.train_dpo import run_experiment as runner  # local import to keep the entry thin

    for spec in resolve_experiment_plan(plan_path):
        experiment_id = build_experiment_id(spec)
        run_dir = results_root / "runs" / experiment_id
        if should_skip_experiment(run_dir, resume=resume):
            scheduled.append({"experiment_id": experiment_id, "status": "skipped", **spec})
            continue
        if execute and runner is not None:
            try:
                result = runner(
                    spec,
                    results_root=results_root,
                    data_root=data_root or Path("data"),
                    resume=resume,
                )
            except Exception as exc:  # pragma: no cover - defensive batch guard
                result = {**spec, "experiment_id": experiment_id, "status": "crash", "notes": str(exc)}
            scheduled.append(result)
        else:
            scheduled.append({"experiment_id": experiment_id, "status": "planned", **spec})
    return scheduled
