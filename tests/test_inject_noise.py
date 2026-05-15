from __future__ import annotations

import json
from pathlib import Path

from src.inject_noise import (
    build_noise_preview_report,
    inject_ambiguous_noise,
    inject_label_flip_noise,
    inject_weak_quality_noise,
    prepare_noise_artifacts,
)


def _make_record(i: int, chosen_score: float, rejected_score: float) -> dict:
    return {
        "prompt": f"prompt-{i}",
        "chosen": f"chosen-{i}",
        "rejected": f"rejected-{i}",
        "source_index": i,
        "metadata": {
            "score_chosen": chosen_score,
            "score_rejected": rejected_score,
        },
    }


def test_label_flip_noise_swaps_pairs_deterministically():
    records = [_make_record(i, 10.0 - i, 1.0 + i) for i in range(10)]

    noisy_a, summary_a = inject_label_flip_noise(records, noise_rate=0.2, seed=7)
    noisy_b, summary_b = inject_label_flip_noise(records, noise_rate=0.2, seed=7)

    assert summary_a == summary_b
    assert summary_a["selected_count"] == 2
    assert noisy_a == noisy_b

    changed = [
        index
        for index, (before, after) in enumerate(zip(records, noisy_a))
        if before["chosen"] != after["chosen"] or before["rejected"] != after["rejected"]
    ]
    assert len(changed) == 2

    for index in changed:
        assert noisy_a[index]["chosen"] == records[index]["rejected"]
        assert noisy_a[index]["rejected"] == records[index]["chosen"]
        assert noisy_a[index]["metadata"]["noise"]["type"] == "label_flip"
        assert noisy_a[index]["metadata"]["noise"]["selected"] is True
        assert noisy_a[index]["metadata"]["score_chosen"] == records[index]["metadata"]["score_rejected"]
        assert noisy_a[index]["metadata"]["score_rejected"] == records[index]["metadata"]["score_chosen"]

    for index in set(range(10)) - set(changed):
        assert noisy_a[index]["chosen"] == records[index]["chosen"]
        assert noisy_a[index]["rejected"] == records[index]["rejected"]
        assert noisy_a[index]["metadata"]["noise"]["selected"] is False


def test_ambiguous_noise_replaces_training_positions_with_smallest_gap_pairs():
    records = [
        _make_record(0, 9.0, 8.9),
        _make_record(1, 9.0, 7.5),
        _make_record(2, 8.5, 5.0),
        _make_record(3, 8.0, 2.0),
        _make_record(4, 7.5, 1.0),
    ]

    noisy, summary = inject_ambiguous_noise(records, noise_rate=0.4, seed=7)

    assert summary["selected_count"] == 2
    selected = [item for item in noisy if item["metadata"]["noise"]["selected"]]
    changed = [
        item
        for before, item in zip(records, noisy)
        if before["prompt"] != item["prompt"]
        or before["chosen"] != item["chosen"]
        or before["rejected"] != item["rejected"]
    ]
    assert len(changed) == 2
    assert len(selected) == 2
    assert sorted(item["source_index"] for item in selected) == [0, 1]
    for item in selected:
        assert item["chosen"] == records[item["source_index"]]["chosen"]
        assert item["rejected"] == records[item["source_index"]]["rejected"]
        assert item["metadata"]["noise"]["type"] == "ambiguous"
        assert item["metadata"]["noise"]["selection_strategy"] == "smallest_score_gap"
        assert item["metadata"]["noise"]["operation"] == "replace_with_small_gap_pair"
        assert "original_source_index" in item["metadata"]["noise"]


def test_weak_quality_noise_replaces_training_positions_with_low_chosen_score_pairs():
    records = [
        _make_record(0, 9.0, 8.0),
        _make_record(1, 6.0, 5.5),
        _make_record(2, 4.0, 3.0),
        _make_record(3, 8.5, 7.0),
        _make_record(4, 5.0, 4.8),
    ]

    noisy, summary = inject_weak_quality_noise(records, noise_rate=0.4, seed=7)

    assert summary["selected_count"] == 2
    selected = [item for item in noisy if item["metadata"]["noise"]["selected"]]
    changed = [
        item
        for before, item in zip(records, noisy)
        if before["prompt"] != item["prompt"]
        or before["chosen"] != item["chosen"]
        or before["rejected"] != item["rejected"]
    ]
    assert len(changed) == 2
    assert len(selected) == 2
    assert sorted(item["source_index"] for item in selected) == [2, 4]
    for item in selected:
        assert item["chosen"] == records[item["source_index"]]["chosen"]
        assert item["rejected"] == records[item["source_index"]]["rejected"]
        assert item["metadata"]["noise"]["type"] == "weak_quality"
        assert item["metadata"]["noise"]["selection_strategy"] == "lowest_chosen_score"
        assert item["metadata"]["noise"]["operation"] == "replace_with_low_quality_pair"
        assert "original_source_index" in item["metadata"]["noise"]


def test_prepare_noise_artifacts_writes_outputs(tmp_path: Path):
    records = [_make_record(i, 10.0 - i, 1.0 + i) for i in range(10)]
    input_path = tmp_path / "train_clean.jsonl"
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    summary = prepare_noise_artifacts(
        input_path=input_path,
        output_dir=tmp_path / "noisy",
        report_path=tmp_path / "noise_preview.md",
        noise_types=("clean", "label_flip", "ambiguous", "weak_quality"),
        noise_rates=(0.0, 0.2),
        seed=123,
        preview_limit=5,
    )

    assert (tmp_path / "noise_preview.md").exists()
    assert (tmp_path / "noisy" / "clean.jsonl").exists()
    assert (tmp_path / "noisy" / "label_flip_20.jsonl").exists()
    assert (tmp_path / "noisy" / "ambiguous_20.jsonl").exists()
    assert (tmp_path / "noisy" / "weak_quality_20.jsonl").exists()
    assert summary["generated_files"] >= 4


def test_build_noise_preview_report_mentions_examples():
    report = build_noise_preview_report(
        {
            "label_flip_20": [
                {
                    "before": {"prompt": "p", "chosen": "c", "rejected": "r"},
                    "after": {"prompt": "p", "chosen": "r", "rejected": "c"},
                    "noise": {"type": "label_flip", "selected": True},
                }
            ]
        }
    )

    assert "label_flip_20" in report
    assert "Before" in report
    assert "After" in report
