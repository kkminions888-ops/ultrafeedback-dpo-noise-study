from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.inject_noise import prepare_noise_artifacts
from src.load_data import load_json_records


SMOKE_METRICS_SCHEMA: Sequence[dict[str, str]] = (
    {
        "metric_name": "train_loss",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "eval_loss",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "reward_margin",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "chosen_logprob",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "rejected_logprob",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "win_rate",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "runtime_minutes",
        "data_type": "float",
        "placeholder_value": "NA",
        "notes": "Placeholder only; real training is not run in Phase 4.",
    },
    {
        "metric_name": "status",
        "data_type": "string",
        "placeholder_value": "NA",
        "notes": "Run status placeholder for later Colab experiments.",
    },
)


def write_smoke_metrics_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric_name", "data_type", "placeholder_value", "notes"),
        )
        writer.writeheader()
        writer.writerows(SMOKE_METRICS_SCHEMA)


def build_cpu_smoke_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CPU Smoke Test Report",
        "",
        "## Scope",
        "",
        "- Local CPU-only smoke validation for the data and noise pipeline.",
        "- No model downloads.",
        "- No formal DPO training.",
        "- Colab free GPU remains reserved for the later training phases.",
        "",
        "## Inputs and Outputs",
        "",
        f"- Loaded subset: `{summary.get('subset_path', 'NA')}`",
        f"- Loaded records: {summary.get('subset_count', 'NA')}",
        f"- Clean output: `{summary.get('clean_path', 'NA')}`",
        f"- Label-flip 20% output: `{summary.get('label_flip_path', 'NA')}`",
        f"- Ambiguous 20% output: `{summary.get('ambiguous_path', 'NA')}`",
        f"- Weak-quality 20% output: `{summary.get('weak_quality_path', 'NA')}`",
        f"- Preview report: `{summary.get('preview_report_path', 'NA')}`",
        f"- placeholder metrics schema: `{summary.get('schema_path', 'NA')}`",
        "",
        "## Checks",
        "",
        f"- pytest target: `{summary.get('pytest_target', 'NA')}`",
        f"- pytest status: `{summary.get('pytest_status', 'NA')}`",
        f"- Training status: `{summary.get('training_status', 'NA')}`",
        f"- Noise artifacts generated: `{summary.get('noise_artifacts_generated', 'NA')}`",
        "",
        "## Notes",
        "",
        "- This report documents a CPU smoke path only.",
        "- It is not a final experiment result and does not contain DPO metrics.",
        "- Later DPO training will happen in Colab on free GPU resources.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _run_pytest() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests"],
        capture_output=True,
        text=True,
    )


def run_cpu_smoke(
    *,
    input_path: Path,
    output_dir: Path,
    preview_report_path: Path,
    cpu_report_path: Path,
    schema_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    records = load_json_records(input_path)
    noise_summary = prepare_noise_artifacts(
        input_path=input_path,
        output_dir=output_dir,
        report_path=preview_report_path,
        noise_types=("clean", "label_flip", "ambiguous", "weak_quality"),
        noise_rates=(0.0, 0.2),
        seed=seed,
    )
    pytest_result = _run_pytest()
    write_smoke_metrics_schema(schema_path)

    summary: dict[str, Any] = {
        "subset_path": str(input_path),
        "subset_count": len(records),
        "clean_path": str(output_dir / "clean.jsonl"),
        "label_flip_path": str(output_dir / "label_flip_20.jsonl"),
        "ambiguous_path": str(output_dir / "ambiguous_20.jsonl"),
        "weak_quality_path": str(output_dir / "weak_quality_20.jsonl"),
        "preview_report_path": str(preview_report_path),
        "schema_path": str(schema_path),
        "noise_artifacts_generated": noise_summary.get("generated_files", 0),
        "pytest_target": "tests/",
        "pytest_status": "success" if pytest_result.returncode == 0 else "failed",
        "training_status": "not_run",
    }
    cpu_report_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_report_path.write_text(build_cpu_smoke_report(summary), encoding="utf-8")

    if pytest_result.returncode != 0:
        summary["pytest_stdout"] = pytest_result.stdout
        summary["pytest_stderr"] = pytest_result.stderr
        return summary

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local CPU smoke workflow.")
    parser.add_argument("--input", default="data/processed/train_clean.jsonl")
    parser.add_argument("--output-dir", default="data/noisy")
    parser.add_argument("--preview-report", default="reports/noise_preview.md")
    parser.add_argument("--cpu-report", default="reports/cpu_smoke_report.md")
    parser.add_argument("--schema-path", default="results/smoke_metrics_schema.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_cpu_smoke(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        preview_report_path=Path(args.preview_report),
        cpu_report_path=Path(args.cpu_report),
        schema_path=Path(args.schema_path),
        seed=args.seed,
    )
    print(summary)
    return 0 if summary.get("pytest_status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
