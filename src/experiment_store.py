from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


EXPERIMENT_FIELDS = (
    "experiment_id",
    "config_name",
    "version",
    "noise_type",
    "noise_rate",
    "seed",
    "model_name",
    "dataset_name",
    "train_size",
    "eval_size",
    "max_steps",
    "status",
    "primary_metric",
    "notes",
)

METRICS_FIELDS = (
    "experiment_id",
    "config_name",
    "noise_type",
    "noise_rate",
    "seed",
    "train_loss",
    "eval_loss",
    "reward_margin",
    "chosen_logprob",
    "rejected_logprob",
    "win_rate",
    "runtime_minutes",
    "status",
)

HISTORY_FIELDS = (
    "experiment_id",
    "config_name",
    "step",
    "epoch",
    "phase",
    "loss",
    "reward_margin",
    "win_rate",
    "chosen_logprob",
    "rejected_logprob",
    "learning_rate",
)


def build_experiment_id(spec: Mapping[str, Any]) -> str:
    payload = {
        key: spec.get(key)
        for key in (
            "config_name",
            "version",
            "noise_type",
            "noise_rate",
            "seed",
            "model_name",
            "dataset_name",
            "train_size",
            "eval_size",
            "max_steps",
        )
        if spec.get(key) is not None
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{payload['config_name']}_{digest}"


def write_run_artifacts(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    train_log: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(train_log, encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, delimiter=",")
        writer.writeheader()
        for row in history_rows:
            writer.writerow({name: row.get(name, "NA") for name in HISTORY_FIELDS})
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(dict(sample), ensure_ascii=False) + "\n")


def _append_row(path: Path, fieldnames: Sequence[str], row: Mapping[str, Any], *, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "NA") for name in fieldnames})


def append_experiment_row(path: Path, row: Mapping[str, Any]) -> None:
    _append_row(path, EXPERIMENT_FIELDS, row, delimiter="\t")


def append_metrics_row(path: Path, row: Mapping[str, Any]) -> None:
    _append_row(path, METRICS_FIELDS, row, delimiter=",")


def append_history_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        _append_row(path, HISTORY_FIELDS, row, delimiter=",")


def export_named_experiment_result(results_root: Path, row: Mapping[str, Any]) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    config_name = str(row.get("config_name", "unknown"))
    path = results_root / f"experiments_{config_name}.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow({name: row.get(name, "NA") for name in EXPERIMENT_FIELDS})
    return path


def export_named_metrics_result(results_root: Path, row: Mapping[str, Any]) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    config_name = str(row.get("config_name", "unknown"))
    path = results_root / f"metrics_{config_name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS, delimiter=",")
        writer.writeheader()
        writer.writerow({name: row.get(name, "NA") for name in METRICS_FIELDS})
    return path


def export_named_history_result(
    results_root: Path,
    config_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / f"history_{config_name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, delimiter=",")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "NA") for name in HISTORY_FIELDS})
    return path
