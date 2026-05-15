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
    parser = argparse.ArgumentParser(description="Prepare fixed noisy training datasets for DPO runs.")
    parser.add_argument("--input", default="data/processed/train_clean.jsonl")
    parser.add_argument("--output-dir", default="data/noisy")
    parser.add_argument("--report", default="reports/noise_preview.md")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _changed_text_count(clean: list[dict], noisy: list[dict]) -> int:
    return sum(
        any(before.get(field) != after.get(field) for field in ("prompt", "chosen", "rejected"))
        for before, after in zip(clean, noisy)
    )


def _selected_count(rows: list[dict]) -> int:
    return sum(bool(row.get("metadata", {}).get("noise", {}).get("selected")) for row in rows)


def _print_validation(output_dir: Path) -> None:
    clean = _load_jsonl(output_dir / "clean.jsonl")
    for name in ["ambiguous_10", "ambiguous_20", "weak_quality_10", "weak_quality_20"]:
        rows = _load_jsonl(output_dir / f"{name}.jsonl")
        print(f"{name} text_changed={_changed_text_count(clean, rows)} selected={_selected_count(rows)}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary = prepare_noise_artifacts(
        input_path=Path(args.input),
        output_dir=output_dir,
        report_path=Path(args.report),
        noise_types=("clean", "label_flip", "ambiguous", "weak_quality"),
        noise_rates=(0.0, 0.1, 0.2),
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _print_validation(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
