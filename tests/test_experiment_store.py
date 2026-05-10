from __future__ import annotations

import csv
from pathlib import Path

from src.experiment_store import (
    append_experiment_row,
    append_metrics_row,
    build_experiment_id,
    write_run_artifacts,
)


def test_build_experiment_id_is_stable_for_same_spec():
    spec = {
        "config_name": "label_flip_20",
        "version": "A",
        "noise_type": "label_flip",
        "noise_rate": 0.2,
        "seed": 42,
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_name": "trl-lib/ultrafeedback_binarized",
    }
    assert build_experiment_id(spec) == build_experiment_id(dict(reversed(list(spec.items()))))


def test_write_run_artifacts_creates_expected_files(tmp_path: Path):
    run_dir = tmp_path / "results" / "runs" / "label_flip_20_42"
    write_run_artifacts(
        run_dir,
        config={"experiment": {"version": "A"}},
        metrics={"status": "success", "train_loss": 1.0},
        samples=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        train_log="hello",
    )

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "samples.jsonl").exists()


def test_append_experiment_row_creates_header_and_appends_rows(tmp_path: Path):
    path = tmp_path / "results" / "experiments.tsv"

    append_experiment_row(
        path,
        {
            "experiment_id": "exp-1",
            "config_name": "clean",
            "version": "A",
            "noise_type": "clean",
            "noise_rate": 0.0,
            "seed": 42,
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset_name": "trl-lib/ultrafeedback_binarized",
            "train_size": 1000,
            "eval_size": 200,
            "max_steps": 200,
            "status": "planned",
            "primary_metric": "NA",
            "notes": "queued",
        },
    )

    text = path.read_text(encoding="utf-8")
    assert text.startswith(
        "experiment_id\tconfig_name\tversion\tnoise_type\tnoise_rate\tseed\tmodel_name\tdataset_name\ttrain_size\teval_size\tmax_steps\tstatus\tprimary_metric\tnotes"
    )
    assert "exp-1" in text


def test_append_metrics_row_creates_header_and_appends_rows(tmp_path: Path):
    path = tmp_path / "results" / "metrics.csv"

    append_metrics_row(
        path,
        {
            "experiment_id": "exp-1",
            "config_name": "clean",
            "noise_type": "clean",
            "noise_rate": 0.0,
            "seed": 42,
            "train_loss": "NA",
            "eval_loss": "NA",
            "reward_margin": "NA",
            "chosen_logprob": "NA",
            "rejected_logprob": "NA",
            "win_rate": "NA",
            "runtime_minutes": "NA",
            "status": "planned",
        },
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["experiment_id"] == "exp-1"
    assert rows[0]["status"] == "planned"
