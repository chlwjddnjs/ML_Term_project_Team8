"""Controlled ablation of fixed vs adaptive hybrid weights.

Run: python experiments/compare_hybrid_weights.py [--data sample]
Train MF/content ONCE and reuse identical fitted components. Only user_weight
changes. Primary cohort matches run.py's seeded 2,000-user sample; a second
cohort evaluates every eligible test user to retain more sparse histories.
This is a test-set diagnostic, not a hyperparameter search.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys.config import load_config
from recsys.data import adapter, cleaner, loader
from recsys.data.split import split_by_user
from recsys.eval.evaluator import evaluate_model
from recsys.models.hybrid import HybridRecommender


class FixedWeightHybrid(HybridRecommender):
    def user_weight(self, user_id):
        return self.alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', choices=['sample', 'slim'], default='slim')
    args = parser.parse_args()
    cfg = load_config(data_profile=args.data)
    out = cfg['paths']['output_dir'] / 'hybrid_weights' / args.data
    out.mkdir(parents=True, exist_ok=True)
    wines, raw = loader.load_xwines(cfg['paths']['xwines_wines'], cfg['paths']['xwines_ratings'])
    items, ratings = adapter.xwines_to_common(wines, raw, cfg)
    items, ratings = cleaner.clean_items(items), cleaner.clean_ratings(ratings)
    train, test = split_by_user(ratings, cfg['split']['test_ratio'], cfg['min_user_ratings'], cfg['seed'])
    params = dict(cfg['model_params']['hybrid'])
    params['mf_params'] = {**cfg['model_params']['mf'], 'seed': cfg['seed']}
    params['content_params'] = cfg['model_params']['content']
    print('Training shared MF and content components...', flush=True)
    adaptive = HybridRecommender(**params).fit(train, items)
    adaptive.name = 'Adaptive'
    fixed = FixedWeightHybrid(**params)
    fixed.__dict__.update(copy.copy(adaptive.__dict__))
    fixed.name = 'Fixed'
    assert fixed.cf is adaptive.cf and fixed.content is adaptive.content
    summaries, groups, pairs = [], [], []
    k = cfg['evaluation']['k']
    metrics = [f'P@{k}', f'R@{k}', f'NDCG@{k}']
    threshold = adaptive.cold_start_threshold
    for cohort, max_users in [('primary', cfg['evaluation']['max_eval_users']),
                              ('all_eligible', test['user_id'].nunique())]:
        per_user = {}
        for model in (fixed, adaptive):
            eval_cfg = {**cfg['evaluation'], 'max_eval_users': max_users}
            summary, rows = evaluate_model(model, train, test, eval_cfg, cfg['seed'])
            summary['cohort'] = cohort
            summaries.append(summary)
            per_user[model.name] = rows.set_index('user_id')
        a, f = per_user['Adaptive'], per_user['Fixed']
        assert a.index.equals(f.index)
        paired = a[['n_train']].copy()
        paired['cohort'] = cohort
        paired['fixed_cf_weight'] = adaptive.alpha
        paired['adaptive_cf_weight'] = [adaptive.user_weight(u) for u in paired.index]
        for metric in metrics:
            paired[f'fixed_{metric}'] = f[metric]
            paired[f'adaptive_{metric}'] = a[metric]
            paired[f'delta_{metric}'] = a[metric] - f[metric]
        # Identical weight => identical scores/recommendations at/above threshold.
        assert np.allclose((a[metrics] - f[metrics])[a.n_train >= threshold], 0)
        for label, mask in [('all', np.ones(len(a), dtype=bool)),
                            ('sparse', a.n_train < threshold),
                            ('regular', a.n_train >= threshold)]:
            row = {'cohort': cohort, 'group': label, 'n_users': int(np.sum(mask))}
            for metric in metrics:
                row[f'fixed_{metric}'] = f.loc[mask, metric].mean()
                row[f'adaptive_{metric}'] = a.loc[mask, metric].mean()
                row[f'delta_{metric}'] = (a.loc[mask, metric] - f.loc[mask, metric]).mean()
            delta = paired.loc[mask, f'delta_NDCG@{k}']
            row.update(improved=int((delta > 1e-12).sum()),
                       worsened=int((delta < -1e-12).sum()),
                       unchanged=int((delta.abs() <= 1e-12).sum()))
            groups.append(row)
        pairs.append(paired.reset_index())
    summary = pd.DataFrame(summaries)
    grouped = pd.DataFrame(groups)
    summary.to_csv(out / 'summary.csv', index=False)
    grouped.to_csv(out / 'groups.csv', index=False)
    pd.concat(pairs).to_csv(out / 'paired_users.csv', index=False)
    settings = {'seed': cfg['seed'], 'split': cfg['split'], 'evaluation': cfg['evaluation'],
                'model_params': cfg['model_params'], 'data': args.data,
                'train_rows': len(train), 'test_rows': len(test),
                'note': 'Single split; fixed/adaptive share fitted models. No tuning on these test results.'}
    (out / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cohort in zip(axes, ['primary', 'all_eligible']):
        g = grouped[grouped.cohort == cohort]
        x = np.arange(len(g))
        ax.bar(x - .18, g[f'fixed_NDCG@{k}'], .36, label='Fixed')
        ax.bar(x + .18, g[f'adaptive_NDCG@{k}'], .36, label='Adaptive')
        ax.set_xticks(x, [f'{r.group}\n(n={r.n_users})' for r in g.itertuples()])
        ax.set_ylabel(f'NDCG@{k}')
        ax.set_title(cohort)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'ndcg_comparison.png', dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(grouped.to_string(index=False))
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
