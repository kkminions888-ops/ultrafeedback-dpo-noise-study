from __future__ import annotations

from pathlib import Path

from src.configs import list_experiment_configs, load_experiment_config


def test_version_a_plan_expands_to_four_experiments():
    plan = list_experiment_configs(Path("configs/version_a.yaml"))
    assert [item.name for item in plan] == [
        "clean.yaml",
        "label_flip_20.yaml",
        "ambiguous_20.yaml",
        "weak_quality_20.yaml",
    ]


def test_version_b_plan_expands_to_ten_experiments():
    plan = list_experiment_configs(Path("configs/version_b.yaml"))
    assert len(plan) == 10
    assert plan[0].name == "clean.yaml"
    assert plan[-1].name == "weak_quality_30.yaml"


def test_single_experiment_version_a_plan_expands_to_one_experiment():
    plan = list_experiment_configs(Path("configs/version_a_clean.yaml"))
    assert [item.name for item in plan] == ["clean.yaml"]


def test_smoke_label_flip_plan_expands_to_one_experiment():
    plan = list_experiment_configs(Path("configs/version_a_label_flip_20_smoke.yaml"))
    assert [item.name for item in plan] == ["label_flip_20_smoke.yaml"]


def test_nested_inherits_resolve_base_model_fields():
    config = load_experiment_config(Path("configs/label_flip_20_smoke.yaml"))

    assert config["model"]["primary"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["training"]["max_steps"] == 10
