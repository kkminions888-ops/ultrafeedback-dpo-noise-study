#!/usr/bin/env bash
set -euo pipefail

python -m src.smoke \
  --input data/processed/train_clean.jsonl \
  --output-dir data/noisy \
  --preview-report reports/noise_preview.md \
  --cpu-report reports/cpu_smoke_report.md \
  --schema-path results/smoke_metrics_schema.csv \
  --seed 42
