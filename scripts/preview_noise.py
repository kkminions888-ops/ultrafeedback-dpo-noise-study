from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inject_noise import prepare_noise_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate noisy dataset previews.")
    parser.add_argument("--input", default="data/processed/train_clean.jsonl")
    parser.add_argument("--output-dir", default="data/noisy")
    parser.add_argument("--report", default="reports/noise_preview.md")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_noise_artifacts(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
