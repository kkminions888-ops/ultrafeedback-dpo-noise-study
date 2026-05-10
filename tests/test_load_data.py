from __future__ import annotations

import json
from pathlib import Path

from src.load_data import (
    PreferenceExample,
    build_dataset_report,
    coerce_preference_example,
    deterministic_split,
    prepare_dataset,
    write_jsonl,
)


def test_coerce_preference_example_accepts_prompt_chosen_rejected():
    raw = {
        "prompt": "What is DPO?",
        "chosen": "A preference-optimization method.",
        "rejected": "A vision model.",
        "score": 0.82,
    }

    example = coerce_preference_example(raw, source_index=7)

    assert isinstance(example, PreferenceExample)
    assert example.prompt == "What is DPO?"
    assert example.chosen == "A preference-optimization method."
    assert example.rejected == "A vision model."
    assert example.source_index == 7
    assert example.metadata["score"] == 0.82


def test_coerce_preference_example_handles_chat_message_lists():
    raw = {
        "chosen": [
            {"role": "user", "content": "Write a snake game."},
            {"role": "assistant", "content": "Here is a pygame version."},
        ],
        "rejected": [
            {"role": "user", "content": "Write a snake game."},
            {"role": "assistant", "content": "Use random text."},
        ],
        "score_chosen": 6.0,
        "score_rejected": 4.0,
    }

    example = coerce_preference_example(raw, source_index=3)

    assert example.prompt == "Write a snake game."
    assert example.chosen == "Here is a pygame version."
    assert example.rejected == "Use random text."
    assert example.metadata["score_chosen"] == 6.0
    assert example.metadata["score_rejected"] == 4.0


def test_deterministic_split_is_seeded_and_size_bound():
    records = [
        {"prompt": f"prompt-{i}", "chosen": f"chosen-{i}", "rejected": f"rejected-{i}"}
        for i in range(8)
    ]

    train_a, eval_a = deterministic_split(records, train_size=5, eval_size=2, seed=42)
    train_b, eval_b = deterministic_split(records, train_size=5, eval_size=2, seed=42)
    train_c, eval_c = deterministic_split(records, train_size=5, eval_size=2, seed=99)

    assert [item["prompt"] for item in train_a] == [item["prompt"] for item in train_b]
    assert [item["prompt"] for item in eval_a] == [item["prompt"] for item in eval_b]
    assert [item["prompt"] for item in train_a] != [item["prompt"] for item in train_c]
    assert [item["prompt"] for item in eval_a] != [item["prompt"] for item in eval_c]
    assert len(train_a) == 5
    assert len(eval_a) == 2
    assert set(item["prompt"] for item in train_a).isdisjoint(
        set(item["prompt"] for item in eval_a)
    )


def test_prepare_dataset_writes_outputs(tmp_path: Path):
    records = [
        {"prompt": f"prompt-{i}", "chosen": f"chosen-{i}", "rejected": f"rejected-{i}"}
        for i in range(10)
    ]

    summary = prepare_dataset(
        records,
        output_dir=tmp_path,
        train_size=6,
        eval_size=3,
        seed=123,
        source_name="trl-lib/ultrafeedback_binarized",
    )

    train_path = tmp_path / "train_clean.jsonl"
    eval_path = tmp_path / "eval_clean.jsonl"
    report_path = tmp_path / "data_report.md"

    assert train_path.exists()
    assert eval_path.exists()
    assert report_path.exists()
    assert summary["train_size"] == 6
    assert summary["eval_size"] == 3
    assert summary["retained_records"] == 10

    with train_path.open("r", encoding="utf-8") as handle:
        train_lines = [json.loads(line) for line in handle if line.strip()]
    assert len(train_lines) == 6
    assert train_lines[0]["prompt"].startswith("prompt-")

    report_text = report_path.read_text(encoding="utf-8")
    assert "UltraFeedback-derived preference data" in report_text
    assert "train_clean.jsonl" in report_text


def test_write_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    examples = [
        PreferenceExample(
            prompt="p",
            chosen="c",
            rejected="r",
            source_index=0,
            metadata={"note": "ok"},
        )
    ]

    write_jsonl(examples, path)

    text = path.read_text(encoding="utf-8").strip()
    assert json.loads(text) == {
        "prompt": "p",
        "chosen": "c",
        "rejected": "r",
        "source_index": 0,
        "metadata": {"note": "ok"},
    }


def test_build_dataset_report_mentions_counts():
    report = build_dataset_report(
        source_name="trl-lib/ultrafeedback_binarized",
        total_records=10,
        retained_records=10,
        train_size=6,
        eval_size=3,
        seed=123,
        dropped_records=0,
    )

    assert "10 raw records" in report
    assert "10 retained records" in report
    assert "train_clean.jsonl" in report
