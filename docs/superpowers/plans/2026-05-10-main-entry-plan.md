# Main Entry Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one thin top-level entry point plus deterministic experiment recording so Colab training can be scheduled, resumed, and audited from a single command.

**Architecture:** `main.py` will only parse commands and dispatch work. `src/pipeline.py` will resolve configs, batch plans, and resume behavior. `src/experiment_store.py` will own append-only run directories and repository-level TSV/CSV bookkeeping. Existing smoke/data/noise modules stay focused and are reused rather than rewritten.

**Tech Stack:** Python 3.11, `argparse`, `pathlib`, `yaml` via `pyyaml`, `csv`, `json`, `pytest`.

---

### Task 1: Add config manifests for scheduled experiments

**Files:**
- Create: `configs/label_flip_20.yaml`
- Create: `configs/ambiguous_20.yaml`
- Create: `configs/weak_quality_20.yaml`
- Create: `configs/version_a.yaml`
- Create: `configs/version_b.yaml`
- Create: `src/configs.py`
- Test: `tests/test_configs.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.configs import load_yaml_config, list_experiment_configs


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_configs.py -v`
Expected: FAIL with missing `src.pipeline` or missing config files.

- [ ] **Step 3: Write minimal implementation**

```yaml
# src/configs.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def list_experiment_configs(plan_path: Path) -> list[Path]:
    plan = load_yaml_config(plan_path)
    return [Path(item) for item in plan["experiments"]]
```

```yaml
# configs/version_a.yaml
name: version_a
experiments:
  - configs/clean.yaml
  - configs/label_flip_20.yaml
  - configs/ambiguous_20.yaml
  - configs/weak_quality_20.yaml
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_configs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/label_flip_20.yaml configs/ambiguous_20.yaml configs/weak_quality_20.yaml configs/version_a.yaml configs/version_b.yaml tests/test_configs.py src/pipeline.py
git commit -m "feat: add experiment plan manifests"
```

---

### Task 2: Add append-only experiment storage

**Files:**
- Create: `src/experiment_store.py`
- Create: `tests/test_experiment_store.py`
- Modify: `results/experiments.tsv`
- Modify: `results/metrics.csv`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.experiment_store import (
    append_experiment_row,
    append_metrics_row,
    build_experiment_id,
    write_run_artifacts,
)


def test_build_experiment_id_is_stable_for_same_spec():
    spec = {
        "config_name": "label_flip_20",
        "version": "A",
        "noise_type": "label_flip",
        "noise_rate": 0.2,
        "seed": 42,
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_name": "trl-lib/ultrafeedback_binarized",
    }
    assert build_experiment_id(spec) == build_experiment_id(dict(reversed(list(spec.items()))))


def test_write_run_artifacts_creates_expected_files(tmp_path: Path):
    run_dir = tmp_path / "results" / "runs" / "label_flip_20_42"
    write_run_artifacts(
        run_dir,
        config={"experiment": {"version": "A"}},
        metrics={"status": "success", "train_loss": 1.0},
        samples=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        train_log="hello",
    )
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "samples.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_experiment_store.py -v`
Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def build_experiment_id(spec: Mapping[str, Any]) -> str:
    payload = {
        "config_name": spec["config_name"],
        "version": spec["version"],
        "noise_type": spec["noise_type"],
        "noise_rate": spec["noise_rate"],
        "seed": spec["seed"],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f'{payload["config_name"]}_{digest}'


def write_run_artifacts(run_dir: Path, *, config: Mapping[str, Any], metrics: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], train_log: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    (run_dir / "train.log").write_text(train_log, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(dict(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(dict(sample), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_experiment_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiment_store.py tests/test_experiment_store.py
git commit -m "feat: add experiment result storage"
```

---

### Task 3: Add orchestration pipeline and resume logic

**Files:**
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `src/smoke.py` only if it is useful to route smoke through the same dispatch layer

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL because pipeline helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.configs import list_experiment_configs, load_yaml_config
from src.experiment_store import build_experiment_id


def should_skip_experiment(run_dir: Path, *, resume: bool) -> bool:
    return resume and (run_dir / "metrics.json").exists()


def resolve_experiment_plan(path: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for config_path in list_experiment_configs(path):
        config = load_yaml_config(config_path)
        experiment = config["experiment"]
        spec = {
            "config_name": config_path.stem,
            "version": experiment["version"],
            "noise_type": experiment["noise_type"],
            "noise_rate": experiment["noise_rate"],
            "seed": experiment["seed"],
            "model_name": config["model"]["primary"],
            "dataset_name": config["dataset"]["name"],
            "train_size": config["dataset"]["train_size"],
            "eval_size": config["dataset"]["eval_size"],
            "max_steps": config["training"]["max_steps"],
        }
        plan.append(spec)
    return plan


def run_batch(plan_path: Path, *, resume: bool, results_root: Path) -> list[dict[str, Any]]:
    plan = resolve_experiment_plan(plan_path)
    scheduled: list[dict[str, Any]] = []
    for spec in plan:
        experiment_id = build_experiment_id(spec)
        run_dir = results_root / "runs" / experiment_id
        if should_skip_experiment(run_dir, resume=resume):
            scheduled.append({"experiment_id": experiment_id, "status": "skipped"})
            continue
        scheduled.append({"experiment_id": experiment_id, "status": "planned"})
    return scheduled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add experiment orchestration pipeline"
```

---

### Task 4: Add the thin top-level `main.py` entry point

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`
- Modify: `Makefile`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys


def test_main_help_lists_expected_subcommands():
    result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "smoke" in result.stdout
    assert "train" in result.stdout
    assert "batch" in result.stdout
    assert "resume" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL because `main.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import resolve_experiment_plan, run_batch
from src.smoke import run_cpu_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke")
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--plan", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--plan", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--plan", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        run_cpu_smoke(
            input_path=Path("data/processed/train_clean.jsonl"),
            output_dir=Path("data/noisy"),
            preview_report_path=Path("reports/noise_preview.md"),
            cpu_report_path=Path("reports/cpu_smoke_report.md"),
            schema_path=Path("results/smoke_metrics_schema.csv"),
            seed=42,
        )
        return 0
    if args.command == "inspect":
        plan = resolve_experiment_plan(Path(args.plan))
        print("\n".join(spec["config_name"] for spec in plan))
        return 0
    if args.command == "batch":
        print(run_batch(Path(args.plan), resume=False, results_root=Path("results")))
        return 0
    if args.command == "resume":
        print(run_batch(Path(args.plan), resume=True, results_root=Path("results")))
        return 0
    if args.command == "train":
        print(f"Train entry received config: {args.config}")
        return 0
    raise ValueError(f"unsupported command: {args.command}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: Update usage docs**

Add a short section to `README.md` showing:

```bash
python main.py smoke
python main.py inspect --plan configs/version_a.yaml
python main.py batch --plan configs/version_a.yaml
```

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py README.md Makefile
git commit -m "feat: add project entry point"
```

---

### Task 5: Final integration check

**Files:**
- Verify: `reports/cpu_smoke_report.md`
- Verify: `results/experiments.tsv`
- Verify: `results/metrics.csv`
- Verify: `docs/superpowers/specs/2026-05-10-main-entry-design.md`
- Verify: `docs/superpowers/plans/2026-05-10-main-entry-plan.md`

- [ ] **Step 1: Run project tests**

Run: `python -m pytest tests`
Expected: PASS

- [ ] **Step 2: Run smoke through the new entry**

Run: `python main.py smoke`
Expected: writes `reports/cpu_smoke_report.md` and `results/smoke_metrics_schema.csv`

- [ ] **Step 3: Confirm append-only bookkeeping**

Run:

```bash
Get-Content results/experiments.tsv
Get-Content results/metrics.csv
```

Expected: headers remain intact, rows append without rewriting prior entries.

- [ ] **Step 4: Stop after the entry layer is ready**

Do not start formal DPO training in this task set.
Phase 5 will reuse the entry point to launch Version A on Colab GPU.

---

### Coverage Check

- Research question and noise setup are untouched.
- `main.py` is introduced as a thin dispatcher, not a monolith.
- Training scheduling is config-driven through plan files.
- Result recording is append-only and run-specific.
- Resume behavior is explicitly covered.
- Local CPU-only smoke remains separate from Colab GPU training.
