from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _record(index: int, chosen_score: float, rejected_score: float) -> dict:
    return {
        "prompt": f"prompt-{index}",
        "chosen": f"chosen-{index}",
        "rejected": f"rejected-{index}",
        "source_index": index,
        "metadata": {
            "score_chosen": chosen_score,
            "score_rejected": rejected_score,
        },
    }


def test_prepare_noise_script_generates_and_validates_fixed_noise(tmp_path: Path):
    input_path = tmp_path / "train_clean.jsonl"
    records = [_record(i, 1000.0 - i, 0.0) for i in range(1000)]
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_noise.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "noisy"),
            "--report",
            str(tmp_path / "noise_preview.md"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ambiguous_10 text_changed=100 selected=100" in result.stdout
    assert "ambiguous_20 text_changed=200 selected=200" in result.stdout
    assert "weak_quality_10 text_changed=100 selected=100" in result.stdout
    assert "weak_quality_20 text_changed=200 selected=200" in result.stdout
    assert (tmp_path / "noisy" / "ambiguous_10.jsonl").exists()
    assert (tmp_path / "noisy" / "weak_quality_20.jsonl").exists()
