from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.experiment_store import METRICS_FIELDS


def _metric_or_na(metrics: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return "NA"


def build_metrics_record(
    *,
    experiment_id: str,
    config_name: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
    train_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    runtime_minutes: float,
    status: str,
) -> dict[str, Any]:
    merged = {
        "experiment_id": experiment_id,
        "config_name": config_name,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "seed": seed,
        "train_loss": _metric_or_na(train_metrics, "train_loss", "loss"),
        "eval_loss": _metric_or_na(eval_metrics, "eval_loss", "loss"),
        "reward_margin": _metric_or_na(
            eval_metrics,
            "reward_margin",
            "rewards/margins",
            "eval_reward_margin",
            "eval_rewards/margins",
        ),
        "chosen_logprob": _metric_or_na(
            train_metrics,
            "chosen_logprob",
            "logps/chosen",
            "eval_logps/chosen",
            "rewards/chosen",
            "eval_rewards/chosen",
        ),
        "rejected_logprob": _metric_or_na(
            train_metrics,
            "rejected_logprob",
            "logps/rejected",
            "eval_logps/rejected",
            "rewards/rejected",
            "eval_rewards/rejected",
        ),
        "win_rate": _metric_or_na(
            eval_metrics,
            "win_rate",
            "rewards/accuracies",
            "eval_rewards/accuracies",
            "accuracy",
        ),
        "runtime_minutes": runtime_minutes,
        "status": status,
    }
    return {field: merged.get(field, "NA") for field in METRICS_FIELDS}


def build_sample_preview(records: Sequence[Mapping[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for record in records[:limit]:
        preview.append(
            {
                "prompt": record.get("prompt", "NA"),
                "chosen": record.get("chosen", "NA"),
                "rejected": record.get("rejected", "NA"),
            }
        )
    return preview


def dump_training_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
