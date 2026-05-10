from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.load_data import load_json_records, write_jsonl


@dataclass(slots=True)
class NoiseSelection:
    index: int
    score_key: str | None
    score_value: float
    score_gap: float | None = None


def _deepcopy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(record))


def _ensure_noise_metadata(
    record: dict[str, Any],
    *,
    noise_type: str,
    noise_rate: float,
    selected: bool,
    selection_strategy: str,
    rank: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = copy.deepcopy(record.get("metadata", {}))
    noise = dict(metadata.get("noise", {}))
    noise.update(
        {
            "type": noise_type,
            "rate": noise_rate,
            "selected": selected,
            "selection_strategy": selection_strategy,
        }
    )
    if rank is not None:
        noise["selected_rank"] = rank
    if extra:
        noise.update(dict(extra))
    metadata["noise"] = noise
    record["metadata"] = metadata
    return record


def _get_score(record: Mapping[str, Any], key: str, fallback: float | None = None) -> float | None:
    metadata = record.get("metadata", {})
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    value = record.get(key)
    if value is not None:
        return float(value)
    return fallback


def _score_gap(record: Mapping[str, Any]) -> float:
    chosen = _get_score(record, "score_chosen")
    rejected = _get_score(record, "score_rejected")
    if chosen is not None and rejected is not None:
        return float(chosen - rejected)
    chosen_text = str(record.get("chosen", ""))
    rejected_text = str(record.get("rejected", ""))
    return abs(len(chosen_text) - len(rejected_text))


def _selected_count(total: int, noise_rate: float) -> int:
    if total <= 0 or noise_rate <= 0:
        return 0
    return max(1, math.floor(total * noise_rate))


def _select_by_random_sample(total: int, count: int, seed: int) -> list[int]:
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return sorted(indices[:count])


def _select_by_metric(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    metric_key: str,
    ascending: bool = True,
) -> list[int]:
    scored: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        metadata = record.get("metadata", {})
        chosen = _get_score(record, "score_chosen")
        rejected = _get_score(record, "score_rejected")
        if chosen is None or rejected is None:
            continue
        if metric_key == "score_gap":
            metric_value = abs(float(chosen - rejected))
        elif metric_key == "chosen_score":
            metric_value = float(chosen)
        else:
            metric_value = float(chosen)
        scored.append((metric_value, index))

    if not scored:
        scored = [(float(_score_gap(record)), index) for index, record in enumerate(records)]

    scored.sort(key=lambda item: (item[0], item[1]), reverse=not ascending)
    return sorted(index for _, index in scored[:count])


def inject_label_flip_noise(
    records: Sequence[Mapping[str, Any]], *, noise_rate: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = _selected_count(len(records), noise_rate)
    selected_indices = set(_select_by_random_sample(len(records), count, seed))
    noisy: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        updated = _deepcopy_record(record)
        selected = index in selected_indices
        if selected:
            updated["chosen"], updated["rejected"] = updated["rejected"], updated["chosen"]
            metadata = copy.deepcopy(updated.get("metadata", {}))
            score_chosen = metadata.get("score_chosen")
            score_rejected = metadata.get("score_rejected")
            metadata["score_chosen"], metadata["score_rejected"] = score_rejected, score_chosen
            updated["metadata"] = metadata
        updated = _ensure_noise_metadata(
            updated,
            noise_type="label_flip",
            noise_rate=noise_rate,
            selected=selected,
            selection_strategy="random_sample",
            rank=sum(1 for i in selected_indices if i < index) if selected else None,
            extra={
                "operation": "swap_chosen_rejected" if selected else "none",
                "original_score_gap": _score_gap(record),
            },
        )
        noisy.append(updated)

    summary = {
        "noise_type": "label_flip",
        "noise_rate": noise_rate,
        "seed": seed,
        "selected_count": len(selected_indices),
        "selection_strategy": "random_sample",
    }
    return noisy, summary


def inject_ambiguous_noise(
    records: Sequence[Mapping[str, Any]], *, noise_rate: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = _selected_count(len(records), noise_rate)
    selected_indices = set(_select_by_metric(records, count, metric_key="score_gap", ascending=True))
    noisy: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        updated = _deepcopy_record(record)
        selected = index in selected_indices
        gap = _score_gap(record)
        updated = _ensure_noise_metadata(
            updated,
            noise_type="ambiguous",
            noise_rate=noise_rate,
            selected=selected,
            selection_strategy="smallest_score_gap",
            rank=sum(1 for i in selected_indices if i < index) if selected else None,
            extra={
                "score_gap": gap,
                "operation": "retain_pair_with_small_gap" if selected else "retain_pair",
            },
        )
        noisy.append(updated)

    summary = {
        "noise_type": "ambiguous",
        "noise_rate": noise_rate,
        "selected_count": len(selected_indices),
        "selection_strategy": "smallest_score_gap",
    }
    return noisy, summary


def inject_weak_quality_noise(
    records: Sequence[Mapping[str, Any]], *, noise_rate: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = _selected_count(len(records), noise_rate)

    eligible_indices = [
        index
        for index, record in enumerate(records)
        if (_get_score(record, "score_chosen") is not None)
        and (_get_score(record, "score_rejected") is not None)
        and _get_score(record, "score_chosen") > _get_score(record, "score_rejected")
    ]
    eligible_indices.sort(key=lambda index: (_get_score(records[index], "score_chosen"), _score_gap(records[index]), index))
    selected_indices = set(eligible_indices[:count]) if eligible_indices else set()
    noisy: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        updated = _deepcopy_record(record)
        selected = index in selected_indices
        chosen_score = _get_score(record, "score_chosen")
        rejected_score = _get_score(record, "score_rejected")
        updated = _ensure_noise_metadata(
            updated,
            noise_type="weak_quality",
            noise_rate=noise_rate,
            selected=selected,
            selection_strategy="lowest_chosen_score",
            rank=sum(1 for i in selected_indices if i < index) if selected else None,
            extra={
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "operation": "retain_direction_with_low_quality_chosen" if selected else "retain_pair",
            },
        )
        noisy.append(updated)

    summary = {
        "noise_type": "weak_quality",
        "noise_rate": noise_rate,
        "selected_count": len(selected_indices),
        "selection_strategy": "lowest_chosen_score",
    }
    return noisy, summary


def inject_noise(
    records: Sequence[Mapping[str, Any]], *, noise_type: str, noise_rate: float, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if noise_type == "clean":
        clean = [_ensure_noise_metadata(_deepcopy_record(record), noise_type="clean", noise_rate=0.0, selected=False, selection_strategy="none", extra={"operation": "none"}) for record in records]
        return clean, {"noise_type": "clean", "noise_rate": 0.0, "selected_count": 0, "selection_strategy": "none"}
    if noise_type == "label_flip":
        return inject_label_flip_noise(records, noise_rate=noise_rate, seed=seed)
    if noise_type == "ambiguous":
        return inject_ambiguous_noise(records, noise_rate=noise_rate)
    if noise_type == "weak_quality":
        return inject_weak_quality_noise(records, noise_rate=noise_rate)
    raise ValueError(f"unsupported noise_type: {noise_type}")


def build_noise_preview_report(
    previews: Mapping[str, Sequence[Mapping[str, Any]]]
) -> str:
    lines: list[str] = ["# Noise Preview", ""]
    for label, items in previews.items():
        lines.extend([f"## {label}", ""])
        for i, item in enumerate(items[:5], start=1):
            before = item["before"]
            after = item["after"]
            noise = item.get("noise", {})
            lines.extend(
                [
                    f"### Example {i}",
                    "",
                    f"- Source index: {item.get('source_index', 'NA')}",
                    f"- Noise type: {noise.get('type', 'NA')}",
                    f"- Selection strategy: {noise.get('selection_strategy', 'NA')}",
                    f"- Changed text: {item.get('changed', False)}",
                    "",
                    "Before:",
                    "",
                    f"- prompt: {before['prompt']}",
                    f"- chosen: {before['chosen']}",
                    f"- rejected: {before['rejected']}",
                    "",
                    "After:",
                    "",
                    f"- prompt: {after['prompt']}",
                    f"- chosen: {after['chosen']}",
                    f"- rejected: {after['rejected']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _make_preview_items(
    before_records: Sequence[Mapping[str, Any]],
    after_records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for before, after in zip(before_records, after_records):
        noise = after.get("metadata", {}).get("noise", {})
        if not noise.get("selected") and before == after:
            continue
        changed_text = any(before[field] != after[field] for field in ("prompt", "chosen", "rejected"))
        previews.append(
            {
                "source_index": after.get("source_index", before.get("source_index", "NA")),
                "changed": changed_text,
                "before": {
                    "prompt": before["prompt"],
                    "chosen": before["chosen"],
                    "rejected": before["rejected"],
                },
                "after": {
                    "prompt": after["prompt"],
                    "chosen": after["chosen"],
                    "rejected": after["rejected"],
                },
                "noise": after.get("metadata", {}).get("noise", {}),
            }
        )
        if len(previews) >= limit:
            break
    return previews


def prepare_noise_artifacts(
    *,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
    noise_types: Sequence[str] = ("clean", "label_flip", "ambiguous", "weak_quality"),
    noise_rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
    seed: int = 42,
    preview_limit: int = 5,
) -> dict[str, Any]:
    records = load_json_records(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    previews: dict[str, list[dict[str, Any]]] = {}
    generated_files: list[str] = []

    for noise_type in noise_types:
        for noise_rate in noise_rates:
            if noise_type == "clean" and noise_rate != 0.0:
                continue
            if noise_type != "clean" and noise_rate == 0.0:
                continue

            noisy_records, summary = inject_noise(
                records,
                noise_type=noise_type,
                noise_rate=noise_rate,
                seed=seed,
            )
            if noise_type == "clean":
                file_name = "clean.jsonl"
            else:
                file_name = f"{noise_type}_{int(round(noise_rate * 100)):02d}.jsonl"
            output_path = output_dir / file_name
            write_jsonl(noisy_records, output_path)
            generated_files.append(str(output_path))

            label = "clean" if noise_type == "clean" else f"{noise_type}_{int(round(noise_rate * 100)):02d}"
            previews[label] = _make_preview_items(records, noisy_records, limit=preview_limit)
            previews[label][:0] = []

    report_text = build_noise_preview_report(previews)
    report_path.write_text(report_text, encoding="utf-8")

    definition_path = report_path.parent / "noise_definition.md"
    if not definition_path.exists():
        definition_path.write_text(build_noise_definition_report(), encoding="utf-8")

    return {
        "generated_files": len(generated_files),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "definition_path": str(definition_path),
        "labels": sorted(previews.keys()),
    }


def build_noise_definition_report() -> str:
    return "\n".join(
        [
            "# Noise Definitions",
            "",
            "## Label-flip noise",
            "",
            "- Operation: randomly swap `chosen` and `rejected` for a seed-controlled fraction of pairs.",
            "- Purpose: simulate direct annotation reversal.",
            "- Source signal: pairwise preference corruption.",
            "",
            "## Ambiguous preference noise",
            "",
            "- Operation: keep the pair direction unchanged but select pairs with the smallest preference gap.",
            "- Primary proxy: `score_chosen` minus `score_rejected` when available.",
            "- Fallback proxy: response length gap or other documented quality proxy if scores are unavailable.",
            "- Purpose: simulate weak or uncertain preference signal.",
            "",
            "## Weak-quality preference noise",
            "",
            "- Operation: keep `chosen` better than `rejected`, but preferentially select low-quality chosen responses.",
            "- Primary proxy: low `score_chosen` while still exceeding `score_rejected`.",
            "- Fallback proxy: documented quality proxy that preserves pair direction.",
            "- Purpose: simulate correct direction with degraded supervision quality.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate noisy preference datasets and preview reports.")
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
