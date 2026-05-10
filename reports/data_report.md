# Data Report

## Dataset

- Source: trl-lib/ultrafeedback_binarized
- Dataset family: UltraFeedback-derived preference data
- 62135 raw records
- 62135 retained records
- 0 dropped records
- Train size: 1000
- Eval size: 200
- Seed: 42

## Outputs

- `data/processed/train_clean.jsonl`
- `data/processed/eval_clean.jsonl`
- `data/processed/data_report.md`

## Notes

- Records are filtered to prompt/chosen/rejected triples.
- Splits are deterministic for a fixed seed.
