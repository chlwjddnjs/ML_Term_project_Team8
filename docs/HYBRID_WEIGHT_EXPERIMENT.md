# Hybrid weight ablation (2026-11-03)

## Reproduce

```bash
python experiments/compare_hybrid_weights.py
python experiments/compare_hybrid_weights.py --data sample
```

Outputs: `outputs/hybrid_weights/slim/` (or `sample/`). Existing main-pipeline results are preserved. CSVs: summary, groups, paired_users; settings.json records configuration; ndcg_comparison.png shows group scores.

## Controlled comparison

Train MF and Content once on the same seed-42 per-user 80/20 split as run.py. Reuse identical fitted components, candidates, normalization and evaluation users. Fixed CF weight = 0.5. Adaptive CF weight = 0.5 * min(1, n_train / 5). No hyperparameters selected using these test results.

Primary evaluation uses the same seeded 2,000 eligible users as run.py. Supplementary evaluation uses all 8,904 users with at least one held-out rating >=4. This is not all users in the dataset. The sparse group has fewer than 5 training ratings; users at/above 5 must have identical metrics under both policies (checked by assertions).

## Slim results

| Cohort/group | n | Fixed NDCG@10 | Adaptive NDCG@10 | Fixed Recall@10 | Adaptive Recall@10 |
|---|---:|---:|---:|---:|---:|
| Primary/all | 2000 | 0.044431 | 0.044420 | 0.075481 | 0.075481 |
| Primary/sparse | 13 | 0.268241 | 0.266482 | 0.384615 | 0.384615 |
| All eligible/all | 8904 | 0.049986 | 0.050013 | 0.082707 | 0.082594 |
| All eligible/sparse | 79 | 0.176274 | 0.179316 | 0.291139 | 0.278481 |

For the 79 sparse users, NDCG improved for 5 users, worsened for 3, and stayed equal for 71. Sparse-group P@10 fell from 0.029114 to 0.027848. Primary Coverage rose from 0.353525 to 0.360477; all-eligible Coverage rose from 0.479643 to 0.490566.

## Interpretation and limits

The previously observed Hybrid improvement over MF/Content does not establish that adaptive weights cause the gain: fixed-weight Hybrid performs nearly identically. Adaptive weights show mixed changes (slightly higher supplementary sparse NDCG, lower precision/recall), with little overall effect. Only 79 eligible users have a different CF weight; 8,825 receive the same weight under both policies.

This is a single-split diagnostic, not evidence of statistically established superiority. The sparse group is small; users with zero ratings are not evaluated. Do not tune the threshold against these test results. Further model selection requires a separate validation set; repeated held-out evaluations can assess robustness.
