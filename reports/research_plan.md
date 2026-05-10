# Research Plan

## Research Question

How do different structures of preference noise affect DPO training and alignment behavior in a small-scale controlled study?

This project keeps the dataset source fixed to `trl-lib/ultrafeedback_binarized`, keeps the base model fixed to `Qwen/Qwen2.5-0.5B-Instruct`, and changes only the noise structure and noise rate.

## Core Hypotheses

### H1: Label-flip noise is the most disruptive

If a fraction of preference pairs has chosen and rejected swapped, DPO should receive directly contradictory supervision and show the largest degradation in train loss, eval loss, reward margin, and win rate.

### H2: Ambiguous preference noise weakens learning signal more than it breaks directionality

If the chosen and rejected items are made close in preference strength, DPO should still usually learn the correct pair direction, but the gradient signal should become weaker and less stable than in the clean condition.

### H3: Weak-quality preference noise preserves direction but lowers output quality

If chosen remains better than rejected but the chosen response is itself low quality, DPO should preserve pair direction while learning from a lower-quality target, which may reduce final eval performance without the same direct corruption as label flips.

## Variables

### Independent Variables

- noise type: `clean`, `label_flip`, `ambiguous`, `weak_quality`
- noise rate: `0.0`, `0.1`, `0.2`, `0.3`
- experiment version: `A` or `B`

### Dependent Variables

- `train_loss`
- `eval_loss`
- `reward_margin`
- `chosen_logprob`
- `rejected_logprob`
- `win_rate`
- runtime and stability signals such as crash or skip status

### Controlled Variables

- dataset source: `trl-lib/ultrafeedback_binarized`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- fallback model: `EleutherAI/pythia-410m`
- seed: `42`
- train/eval subset construction method
- train_size and eval_size within each version
- maximum sequence lengths
- batch size and gradient accumulation policy
- evaluation metric definitions
- config-driven experiment naming and logging
- local CPU-only work versus Colab GPU training split

## Noise Definitions

### 1. Label-flip noise

Operation: randomly select a fraction of pairs and swap `chosen` and `rejected`.

Purpose: simulate direct annotation reversal or pairwise preference labeling error.

Required properties:

- the swap must be literal
- the selection must obey the specified noise rate
- the sampling must be seed-controlled
- previews should show before/after examples

### 2. Ambiguous preference noise

Operation: construct or select pairs where the preference gap is small, so the distinction between `chosen` and `rejected` is weak.

Primary proxy order:

1. score gap if the data contains `score` or `rating`
2. otherwise a documented proxy such as response length difference, text similarity, reward proxy, or available quality metadata

Purpose: simulate uncertain or nearly tied preferences rather than reversed preferences.

Required properties:

- `chosen` and `rejected` must not be simply swapped
- the construction rule must be documented
- the logic must be distinguishable from label-flip noise

### 3. Weak-quality preference noise

Operation: keep `chosen` better than `rejected`, but make the chosen response itself lower quality than the clean baseline target.

Possible proxy order:

1. chosen sample with lower quality score than clean chosen, but still above rejected
2. otherwise a documented proxy based on available metadata

Purpose: simulate correct direction with degraded supervision quality.

Required properties:

- no swap of `chosen` and `rejected`
- pair direction must remain intact
- the definition must be distinct from ambiguous noise

## Experimental Matrices

### Version A

Version A is the minimal publishable set and must include one clean baseline and one 20% condition for each noise type.

| experiment_id | noise_type | noise_rate | role |
|---|---|---:|---|
| A-clean | clean | 0.0 | baseline |
| A-label_flip_20 | label_flip | 0.2 | direct corruption |
| A-ambiguous_20 | ambiguous | 0.2 | weak signal |
| A-weak_quality_20 | weak_quality | 0.2 | degraded supervision |

Version A is enough to test whether the three noise structures behave differently under one fixed corruption level.

### Version B

Version B expands Version A into a noise-rate sensitivity study.

| experiment_id | noise_type | noise_rate | role |
|---|---|---:|---|
| B-clean | clean | 0.0 | baseline |
| B-label_flip_10 | label_flip | 0.1 | sensitivity point |
| B-label_flip_20 | label_flip | 0.2 | sensitivity point |
| B-label_flip_30 | label_flip | 0.3 | sensitivity point |
| B-ambiguous_10 | ambiguous | 0.1 | sensitivity point |
| B-ambiguous_20 | ambiguous | 0.2 | sensitivity point |
| B-ambiguous_30 | ambiguous | 0.3 | sensitivity point |
| B-weak_quality_10 | weak_quality | 0.1 | sensitivity point |
| B-weak_quality_20 | weak_quality | 0.2 | sensitivity point |
| B-weak_quality_30 | weak_quality | 0.3 | sensitivity point |

Version B is the superset of Version A and is used to test whether the effect size changes smoothly or nonlinearly with noise rate.

## Expected Risks

- The available data may not contain a strong `score` or `rating` field, so ambiguous and weak-quality definitions may need documented proxies.
- The three noise types could partially overlap if proxy rules are too loose, so the definitions must remain operationally distinct.
- Colab free GPU availability may limit later training, but this phase only prepares the plan.
- Small-scale results may be noisy, so conclusions should stay modest and tied to this setup.
- Some metrics may be unavailable later; if so, they should be recorded as `NA` rather than invented.

## Narrative Thread

The paper should tell a constrained, evidence-based story:

1. DPO depends on pairwise preference supervision.
2. Real preference data can be noisy in more than one way.
3. Noise structure may matter as much as noise rate.
4. A small-scale controlled study can isolate these differences without claiming broad generality.
5. The clean baseline anchors the comparison, while Version A and Version B progressively expose how each noise type changes training behavior.

The conclusion should stay limited to this experimental setting and should not generalize beyond the tested model, dataset subset, and compute budget.
