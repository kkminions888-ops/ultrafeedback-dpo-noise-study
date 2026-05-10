from __future__ import annotations

from pathlib import Path

from src.pipeline import resolve_experiment_plan, should_skip_experiment


def test_should_skip_experiment_when_metrics_exist(tmp_path: Path):
    run_dir = tmp_path / "results" / "runs" / "clean_42"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"status": "success"}', encoding="utf-8")
    assert should_skip_experiment(run_dir, resume=True) is True


def test_version_a_plan_is_sequential_and_config_driven():
    plan = resolve_experiment_plan(Path("configs/version_a.yaml"))
    assert [item["noise_type"] for item in plan] == ["clean", "label_flip", "ambiguous", "weak_quality"]
