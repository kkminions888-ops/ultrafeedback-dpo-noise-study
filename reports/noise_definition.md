# Noise Definitions

## Label-flip noise

- Operation: randomly swap `chosen` and `rejected` for a seed-controlled fraction of pairs.
- Purpose: simulate direct annotation reversal.
- Source signal: pairwise preference corruption.

## Ambiguous preference noise

- Operation: keep the pair direction unchanged but select pairs with the smallest preference gap.
- Primary proxy: `score_chosen` minus `score_rejected` when available.
- Fallback proxy: response length gap or other documented quality proxy if scores are unavailable.
- Purpose: simulate weak or uncertain preference signal.

## Weak-quality preference noise

- Operation: keep `chosen` better than `rejected`, but preferentially select low-quality chosen responses.
- Primary proxy: low `score_chosen` while still exceeding `score_rejected`.
- Fallback proxy: documented quality proxy that preserves pair direction.
- Purpose: simulate correct direction with degraded supervision quality.
