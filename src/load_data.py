from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROMPT_KEYS = ("prompt", "instruction", "question", "input", "query")
CHOSEN_KEYS = ("chosen", "response_a", "accepted", "preferred", "winner", "answer_a")
REJECTED_KEYS = ("rejected", "response_b", "dispreferred", "lost", "answer_b")


@dataclass(slots=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    source_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _extract_message_text(value: Any, *, prefer_role: str | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        return None if content is None else str(content)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        messages = [item for item in value if isinstance(item, Mapping)]
        if not messages:
            return None
        if prefer_role is not None:
            for message in messages:
                if message.get("role") == prefer_role and message.get("content") is not None:
                    return str(message["content"])
        for message in messages:
            content = message.get("content")
            if content is not None:
                return str(content)
    return None


def _derive_prompt(record: Mapping[str, Any]) -> str | None:
    prompt = _first_present(record, PROMPT_KEYS)
    if prompt is not None:
        return _extract_message_text(prompt) or str(prompt)

    for key in CHOSEN_KEYS + REJECTED_KEYS:
        candidate = record.get(key)
        derived = _extract_message_text(candidate, prefer_role="user")
        if derived is not None:
            return derived
    return None


def _derive_completion(value: Any) -> str | None:
    derived = _extract_message_text(value, prefer_role="assistant")
    if derived is not None:
        return derived
    return _extract_message_text(value)


def coerce_preference_example(
    record: Mapping[str, Any], source_index: int
) -> PreferenceExample:
    prompt = _derive_prompt(record)
    chosen = _derive_completion(_first_present(record, CHOSEN_KEYS))
    rejected = _derive_completion(_first_present(record, REJECTED_KEYS))

    if prompt is None or chosen is None or rejected is None:
        missing = [
            name
            for name, value in (
                ("prompt", prompt),
                ("chosen", chosen),
                ("rejected", rejected),
            )
            if value is None
        ]
        raise ValueError(f"record {source_index} missing required fields: {', '.join(missing)}")

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {*PROMPT_KEYS, *CHOSEN_KEYS, *REJECTED_KEYS}
    }
    return PreferenceExample(
        prompt=str(prompt),
        chosen=str(chosen),
        rejected=str(rejected),
        source_index=source_index,
        metadata=metadata,
    )


def deterministic_split(
    records: Sequence[Mapping[str, Any]],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if train_size < 0 or eval_size < 0:
        raise ValueError("train_size and eval_size must be non-negative")

    selected_total = train_size + eval_size
    if selected_total > len(records):
        raise ValueError(
            f"requested {selected_total} records but only {len(records)} are available"
        )

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    selected = indices[:selected_total]
    train_indices = selected[:train_size]
    eval_indices = selected[train_size:selected_total]
    return [records[i] for i in train_indices], [records[i] for i in eval_indices]


def write_jsonl(records: Iterable[Mapping[str, Any] | PreferenceExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if isinstance(record, PreferenceExample) else dict(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_dataset_report(
    *,
    source_name: str,
    total_records: int,
    retained_records: int,
    train_size: int,
    eval_size: int,
    seed: int,
    dropped_records: int,
) -> str:
    return "\n".join(
        [
            "# Data Report",
            "",
            "## Dataset",
            "",
            f"- Source: {source_name}",
            "- Dataset family: UltraFeedback-derived preference data",
            f"- {total_records} raw records",
            f"- {retained_records} retained records",
            f"- {dropped_records} dropped records",
            f"- Train size: {train_size}",
            f"- Eval size: {eval_size}",
            f"- Seed: {seed}",
            "",
            "## Outputs",
            "",
            "- `train_clean.jsonl`",
            "- `eval_clean.jsonl`",
            "",
            "## Notes",
            "",
            "- Records are filtered to prompt/chosen/rejected triples.",
            "- Splits are deterministic for a fixed seed.",
        ]
    )


def _normalize_records(records: Sequence[Mapping[str, Any]]) -> list[PreferenceExample]:
    normalized: list[PreferenceExample] = []
    for source_index, record in enumerate(records):
        normalized.append(coerce_preference_example(record, source_index=source_index))
    return normalized


def prepare_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    train_size: int,
    eval_size: int,
    seed: int,
    source_name: str,
) -> dict[str, Any]:
    normalized = _normalize_records(records)
    train_examples, eval_examples = deterministic_split(
        [example.to_dict() for example in normalized],
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_clean.jsonl"
    eval_path = output_dir / "eval_clean.jsonl"
    report_path = output_dir / "data_report.md"

    write_jsonl(train_examples, train_path)
    write_jsonl(eval_examples, eval_path)
    report_text = build_dataset_report(
        source_name=source_name,
        total_records=len(records),
        retained_records=len(normalized),
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
        dropped_records=max(len(records) - len(normalized), 0),
    )
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "source_name": source_name,
        "total_records": len(records),
        "retained_records": len(normalized),
        "train_size": train_size,
        "eval_size": eval_size,
        "seed": seed,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "report_path": str(report_path),
    }


def load_json_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data]
        raise ValueError("JSON input must contain a list of records")
    raise ValueError(f"unsupported file format: {path.suffix}")


def load_hf_dataset_records(dataset_name: str, split: str = "train") -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "datasets is required to load Hugging Face datasets. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def prepare_from_source(
    source: str | Path,
    *,
    output_dir: Path,
    train_size: int,
    eval_size: int,
    seed: int,
    split: str = "train",
    source_name: str = "trl-lib/ultrafeedback_binarized",
) -> dict[str, Any]:
    source_path = Path(source)
    if source_path.exists():
        records = load_json_records(source_path)
        source_label = str(source_path)
    else:
        records = load_hf_dataset_records(str(source), split=split)
        source_label = source_name

    return prepare_dataset(
        records,
        output_dir=output_dir,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
        source_name=source_label,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UltraFeedback-derived preference subsets.")
    parser.add_argument("source", help="Path to a local JSON/JSONL file or a Hugging Face dataset name")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for prepared outputs")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train", help="Dataset split to read when using Hugging Face")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_from_source(
        args.source,
        output_dir=Path(args.output_dir),
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        split=args.split,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
