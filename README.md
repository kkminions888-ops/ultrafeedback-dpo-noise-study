# A Small-Scale Controlled Study of Heterogeneous Preference Noise in Direct Preference Optimization

This repository is being set up in phases.

Phase 0 completed the local project scaffolding, environment check, and CPU-only boundary.

Local work is CPU-only.
DPO training will run later on free Colab GPU.

## Current scope

- project structure
- configuration skeletons
- environment documentation
- experiment log scaffold

## Not in scope yet

- model downloads
- DPO training
- Version A / Version B runs
- experiment analysis
- paper drafting

## Entry points

- `python main.py smoke`
- `python main.py inspect --plan configs/version_a.yaml`
- `python main.py train --config configs/clean.yaml`
- `python main.py batch --plan configs/version_a.yaml`
- `python main.py resume --plan configs/version_a.yaml`

## Training Flow

`main.py` is the thin dispatcher for both local validation and Colab training.
`train` runs one config, while `batch` and `resume` expand a plan and write append-only run records under `results/runs/{experiment_id}/`.
