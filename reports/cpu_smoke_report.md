# CPU Smoke Test Report

## Scope

- Local CPU-only smoke validation for the data and noise pipeline.
- No model downloads.
- No formal DPO training.
- Colab free GPU remains reserved for the later training phases.

## Inputs and Outputs

- Loaded subset: `data\processed\train_clean.jsonl`
- Loaded records: 1000
- Clean output: `data\noisy\clean.jsonl`
- Label-flip 20% output: `data\noisy\label_flip_20.jsonl`
- Ambiguous 20% output: `data\noisy\ambiguous_20.jsonl`
- Weak-quality 20% output: `data\noisy\weak_quality_20.jsonl`
- Preview report: `reports\noise_preview.md`
- placeholder metrics schema: `results\smoke_metrics_schema.csv`

## Checks

- pytest target: `tests/`
- pytest status: `success`
- Training status: `not_run`
- Noise artifacts generated: `4`

## Notes

- This report documents a CPU smoke path only.
- It is not a final experiment result and does not contain DPO metrics.
- Later DPO training will happen in Colab on free GPU resources.
