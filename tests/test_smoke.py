from __future__ import annotations

import csv
from pathlib import Path

from src.smoke import build_cpu_smoke_report, write_smoke_metrics_schema


def test_build_cpu_smoke_report_mentions_required_steps():
    report = build_cpu_smoke_report(
        {
            "subset_path": "data/processed/train_clean.jsonl",
            "clean_path": "data/noisy/clean.jsonl",
            "label_flip_path": "data/noisy/label_flip_20.jsonl",
            "ambiguous_path": "data/noisy/ambiguous_20.jsonl",
            "weak_quality_path": "data/noisy/weak_quality_20.jsonl",
            "preview_report_path": "reports/noise_preview.md",
            "schema_path": "results/smoke_metrics_schema.csv",
            "pytest_status": "success",
            "training_status": "not_run",
        }
    )

    assert "CPU Smoke Test" in report
    assert "train_clean.jsonl" in report
    assert "label_flip_20.jsonl" in report
    assert "ambiguous_20.jsonl" in report
    assert "weak_quality_20.jsonl" in report
    assert "pytest" in report
    assert "placeholder metrics schema" in report
    assert "CPU-only" in report
    assert "Colab free GPU" in report


def test_write_smoke_metrics_schema_writes_expected_columns(tmp_path: Path):
    output_path = tmp_path / "smoke_metrics_schema.csv"

    write_smoke_metrics_schema(output_path)

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["metric_name"] == "train_loss"
    assert rows[0]["placeholder_value"] == "NA"
    assert {row["metric_name"] for row in rows} >= {
        "train_loss",
        "eval_loss",
        "reward_margin",
        "chosen_logprob",
        "rejected_logprob",
        "win_rate",
        "runtime_minutes",
        "status",
    }
