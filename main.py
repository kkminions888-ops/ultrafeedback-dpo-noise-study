from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.configs import load_experiment_config
from src.pipeline import build_experiment_spec, resolve_experiment_plan, run_batch
from src.smoke import run_cpu_smoke
from src.train_dpo import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="run the local CPU smoke workflow")

    train = subparsers.add_parser("train", help="run one experiment config")
    train.add_argument("--config", required=True)
    train.add_argument("--results-root", default="results")
    train.add_argument("--data-root", default="data")
    train.add_argument("--resume", action="store_true")

    batch = subparsers.add_parser("batch", help="schedule a batch from a plan")
    batch.add_argument("--plan", required=True)
    batch.add_argument("--results-root", default="results")
    batch.add_argument("--data-root", default="data")

    resume = subparsers.add_parser("resume", help="resume a batch from a plan")
    resume.add_argument("--plan", required=True)
    resume.add_argument("--results-root", default="results")
    resume.add_argument("--data-root", default="data")

    inspect = subparsers.add_parser("inspect", help="print the expanded experiment plan")
    inspect.add_argument("--plan", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "smoke":
        summary = run_cpu_smoke(
            input_path=Path("data/processed/train_clean.jsonl"),
            output_dir=Path("data/noisy"),
            preview_report_path=Path("reports/noise_preview.md"),
            cpu_report_path=Path("reports/cpu_smoke_report.md"),
            schema_path=Path("results/smoke_metrics_schema.csv"),
            seed=42,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "train":
        config_path = Path(args.config)
        config = load_experiment_config(config_path)
        spec = build_experiment_spec(config_path)
        summary = run_experiment(
            spec,
            results_root=Path(args.results_root),
            data_root=Path(args.data_root),
            resume=bool(args.resume),
        )
        print(json.dumps({"config": config, "summary": summary}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "inspect":
        plan = resolve_experiment_plan(Path(args.plan))
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if args.command == "batch":
        planned = run_batch(
            Path(args.plan),
            resume=False,
            results_root=Path(args.results_root),
            data_root=Path(args.data_root),
            execute=True,
            experiment_runner=run_experiment,
        )
        print(json.dumps(planned, indent=2, ensure_ascii=False))
        return 0

    if args.command == "resume":
        planned = run_batch(
            Path(args.plan),
            resume=True,
            results_root=Path(args.results_root),
            data_root=Path(args.data_root),
            execute=True,
            experiment_runner=run_experiment,
        )
        print(json.dumps(planned, indent=2, ensure_ascii=False))
        return 0

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
