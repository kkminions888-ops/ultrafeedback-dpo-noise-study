# Main Entry and Experiment Orchestration Design

## Goal

Provide one thin project entry point for Colab and local orchestration without turning the repository into a monolith.

The entry point will:

- select a workflow from the command line
- load config-driven experiment definitions
- schedule training jobs sequentially
- record run artifacts and status deterministically
- support resume / skip behavior for partially completed batches
- keep local CPU-only usage separate from later Colab GPU training

## Proposed Architecture

### `main.py`

`main.py` will be the top-level dispatcher only. It should not contain training logic.

Suggested subcommands:

- `smoke`: run the local CPU smoke workflow
- `train`: run one experiment from a config file
- `batch`: run a list of experiments from a plan file
- `resume`: continue a batch by skipping experiments that already have completed artifacts
- `inspect`: print resolved config and planned experiment IDs without running training

### `src/experiment_store.py`

This module will own result recording and status bookkeeping.

Responsibilities:

- create `results/runs/{experiment_id}/`
- write `config.yaml`, `train.log`, `metrics.json`, `samples.jsonl`
- append rows to `results/experiments.tsv`
- append rows to `results/metrics.csv`
- create stable experiment IDs from config identity plus noise settings
- classify status as `success`, `crash`, `invalid`, or `skipped`

### `src/pipeline.py`

This module will own orchestration only.

Responsibilities:

- load the base config plus per-experiment overrides
- resolve experiment matrices for Version A and Version B
- call training and evaluation functions
- hand structured outputs to `experiment_store`
- enforce CPU-only guardrails locally

## Training Scheduling

The batch runner will read a plan file that expands into a list of experiments.

For Phase 5, the plan will contain Version A only:

- `clean`
- `label_flip_20`
- `ambiguous_20`
- `weak_quality_20`

The runner will execute experiments sequentially.

If an experiment already has a completed `metrics.json`, `resume` will skip it.
If an experiment has partial files but no successful completion, the runner will mark it as `crash` or continue based on the plan mode.

## Result Recording

Each experiment will write a self-contained run directory:

- `results/runs/{experiment_id}/config.yaml`
- `results/runs/{experiment_id}/train.log`
- `results/runs/{experiment_id}/metrics.json`
- `results/runs/{experiment_id}/samples.jsonl`

Then the repository-level logs will be updated:

- `results/experiments.tsv`
- `results/metrics.csv`

These files will be append-only.

No completed or failed experiment should be deleted.

## Error Handling

- Invalid config: fail fast, write a clear `invalid` status, do not start training.
- Missing dataset or model: mark the experiment as `invalid` or `crash` depending on when the failure occurs.
- Out-of-memory on Colab: record `crash`, preserve logs, and continue only if the batch mode is configured to do so.
- Already completed experiment: mark as `skipped` in resume mode.
- Local CPU training request: reject the request with a clear message, because formal DPO training is reserved for Colab GPU.

## Testing Plan

Add tests for:

- experiment ID generation
- config merge / resolution
- append-only writing to `experiments.tsv`
- append-only writing to `metrics.csv`
- resume skip logic
- status classification for success, crash, invalid, and skipped
- CPU-only guard that blocks formal training locally

## Scope Boundaries

This design does not add a new training algorithm.
It does not change the dataset source.
It does not change the research question.
It only adds a practical orchestration layer so Phase 5 and later phases can be run repeatably.
