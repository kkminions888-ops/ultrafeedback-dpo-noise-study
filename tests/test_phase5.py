from __future__ import annotations

from pathlib import Path

from src.evaluate import build_metrics_record
from src.pipeline import run_batch
from src.train_dpo import build_dpo_config_kwargs, resolve_train_dataset_path


def test_resolve_train_dataset_path_selects_clean_and_noisy_files():
    clean_spec = {
        "config_name": "clean",
        "noise_type": "clean",
        "noise_rate": 0.0,
    }
    noisy_spec = {
        "config_name": "label_flip_20",
        "noise_type": "label_flip",
        "noise_rate": 0.2,
    }

    assert resolve_train_dataset_path(clean_spec, Path("data")) == Path("data/noisy/clean.jsonl")
    assert resolve_train_dataset_path(noisy_spec, Path("data")) == Path("data/noisy/label_flip_20.jsonl")


def test_build_dpo_config_kwargs_uses_base_training_parameters():
    config = {
        "training": {
            "max_steps": 200,
            "learning_rate": 0.000005,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 512,
            "max_prompt_length": 256,
            "seed": 42,
        }
    }

    kwargs = build_dpo_config_kwargs(config, output_dir=Path("results/runs/test"))

    assert kwargs["output_dir"] == "results/runs/test"
    assert kwargs["per_device_train_batch_size"] == 1
    assert kwargs["max_steps"] == 200
    assert kwargs["learning_rate"] == 0.000005
    assert kwargs["gradient_accumulation_steps"] == 8
    assert kwargs["max_length"] == 512
    assert kwargs["seed"] == 42
    assert kwargs["report_to"] == "none"
    assert kwargs["do_train"] is True


def test_build_metrics_record_fills_missing_metrics_with_na():
    record = build_metrics_record(
        experiment_id="exp-1",
        config_name="clean",
        noise_type="clean",
        noise_rate=0.0,
        seed=42,
        train_metrics={"train_loss": 0.5},
        eval_metrics={},
        runtime_minutes=1.25,
        status="success",
    )

    assert record["experiment_id"] == "exp-1"
    assert record["train_loss"] == 0.5
    assert record["eval_loss"] == "NA"
    assert record["reward_margin"] == "NA"
    assert record["runtime_minutes"] == 1.25
    assert record["status"] == "success"


def test_build_metrics_record_accepts_trl_metric_names():
    record = build_metrics_record(
        experiment_id="exp-2",
        config_name="clean",
        noise_type="clean",
        noise_rate=0.0,
        seed=42,
        train_metrics={"loss": 0.4, "logps/chosen": -0.1, "logps/rejected": -0.3},
        eval_metrics={"eval_loss": 0.2, "rewards/margins": 0.7, "rewards/accuracies": 1.0},
        runtime_minutes=2.0,
        status="success",
    )

    assert record["train_loss"] == 0.4
    assert record["eval_loss"] == 0.2
    assert record["reward_margin"] == 0.7
    assert record["chosen_logprob"] == -0.1
    assert record["rejected_logprob"] == -0.3
    assert record["win_rate"] == 1.0


def test_run_batch_can_use_injected_executor(tmp_path: Path):
    seen: list[str] = []

    def fake_runner(spec, *, results_root, data_root, resume):
        seen.append(spec["config_name"])
        return {"experiment_id": spec["config_name"], "status": "success"}

    results = run_batch(
        Path("configs/version_a.yaml"),
        resume=False,
        results_root=tmp_path / "results",
        data_root=Path("data"),
        execute=True,
        experiment_runner=fake_runner,
    )

    assert seen == ["clean", "label_flip_20", "ambiguous_20", "weak_quality_20"]
    assert [item["status"] for item in results] == ["success", "success", "success", "success"]
