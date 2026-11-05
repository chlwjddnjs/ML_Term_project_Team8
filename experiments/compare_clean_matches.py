"""Compare cross-dataset validation before/after removing reviewed bad matches.

Run: python experiments/compare_clean_matches.py
Uses the same split/settings as run.py. Each model is fitted once and its
quality scores are reused for both match policies. No scores select labels.
Original outputs/cross_dataset.csv is preserved. Exact matches are retained;
fuzzy-only matches require a 'same' label. Missing labels fail explicitly.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys.config import load_config
from recsys.data import adapter, cleaner, loader
from recsys.data.split import split_by_user
from recsys.eval import cross_dataset
from run import build_model


def filter_matches(matches, exact, expert, labels):
    keys = ['item_id', 'winery_expert', 'name_expert']
    if labels.duplicated(keys).any():
        raise ValueError('Duplicate review-label keys')
    if not labels.label.isin(['same', 'different', 'unsure']).all():
        raise ValueError('Unexpected review label')
    audit = matches.merge(exact[['item_id', 'expert_id']].assign(is_exact=True),
                          on=['item_id', 'expert_id'], how='left', validate='one_to_one')
    audit = audit.merge(expert[['expert_id', 'winery', 'name']], on='expert_id',
                        validate='many_to_one').rename(columns={'winery':'winery_expert', 'name':'name_expert'})
    audit = audit.merge(labels[keys + ['label', 'reason']], on=keys, how='left', validate='many_to_one')
    exact_mask = audit.is_exact.eq(True)
    if audit.loc[~exact_mask, 'label'].isna().any():
        raise ValueError('Unreviewed fuzzy-only matches; label these before evaluation')
    audit['decision'] = np.where(exact_mask, 'keep_exact',
                                 np.where(audit.label.eq('same'), 'keep_same', 'remove_' + audit.label.fillna('unlabeled')))
    kept = audit[audit.decision.str.startswith('keep')][['item_id','expert_id','score']].copy()
    assert len(kept) <= len(matches)
    return kept, audit


def main():
    cfg = load_config()
    out = cfg['paths']['output_dir'] / 'clean_matches'
    out.mkdir(parents=True, exist_ok=True)
    original = cfg['paths']['output_dir'] / 'cross_dataset.csv'
    if original.exists() and not (out / 'previous_cross_dataset.csv').exists():
        shutil.copy2(original, out / 'previous_cross_dataset.csv')
    wines, raw = loader.load_xwines(cfg['paths']['xwines_wines'], cfg['paths']['xwines_ratings'])
    items, ratings = adapter.xwines_to_common(wines, raw, cfg)
    items, ratings = cleaner.clean_items(items), cleaner.clean_ratings(ratings)
    expert_raw = loader.load_winemag(cfg['paths']['winemag'])
    if expert_raw is None:
        raise FileNotFoundError('Wine Enthusiast CSV required')
    expert = cleaner.clean_expert(adapter.winemag_to_common(expert_raw, cfg, scale_method='linear'))
    print('Matching wines...', flush=True)
    before = adapter.match_items(items, expert, method='fuzzy', threshold=.85)
    exact = adapter.match_items(items, expert, method='exact')
    label_path = Path(__file__).resolve().parent / 'fuzzy_match_labels.csv'
    after, audit = filter_matches(before, exact, expert, pd.read_csv(label_path))
    audit.to_csv(out / 'match_audit.csv', index=False, encoding='utf-8-sig')
    audit[audit.decision.str.startswith('remove')].to_csv(out / 'removed_matches.csv', index=False, encoding='utf-8-sig')
    before.to_csv(out / 'matches_before.csv', index=False)
    after.to_csv(out / 'matches_after.csv', index=False)
    counts = pd.DataFrame([{'policy': p, 'wines': m.item_id.nunique(), 'reviews': len(m)}
                           for p,m in [('before',before),('after',after)]])
    counts.to_csv(out / 'match_counts.csv', index=False)
    print(counts.to_string(index=False), flush=True)
    train, test = split_by_user(ratings, cfg['split']['test_ratio'], cfg['min_user_ratings'], cfg['seed'])
    expert_before = cross_dataset.expert_score_per_item(before, expert)
    expert_after = cross_dataset.expert_score_per_item(after, expert)
    common = expert_before.index.intersection(expert_after.index)
    quality = {'X-Wines mean rating (reference)':train.groupby('item_id').rating.mean()}
    for name in cfg['models']:
        print('Training once:', name, flush=True)
        model = build_model(name, cfg).fit(train, items)
        quality[model.name] = cross_dataset.model_quality_per_item(model, train, items.item_id)
    rows = []
    for cohort in ['policy_items', 'common_items']:
        for policy, scores in [('before', expert_before),('after',expert_after)]:
            if cohort == 'common_items':
                scores = scores.loc[common]
            for name,q in quality.items():
                rows.append({'cohort':cohort,'policy':policy,'model':name,
                             **cross_dataset.rank_agreement(q,scores,cfg['evaluation']['k'])})
    results = pd.DataFrame(rows)
    results.to_csv(out / 'comparison.csv', index=False)
    for policy in ['before','after']:
        results[(results.policy==policy)&(results.cohort=='policy_items')].to_csv(out / f'cross_dataset_{policy}.csv', index=False)
    delta = results.pivot(index=['cohort','model'], columns='policy', values=['spearman','topk_overlap','n_items'])
    for metric in ['spearman','topk_overlap']:
        delta[(metric,'delta_after_minus_before')] = delta[(metric,'after')] - delta[(metric,'before')]
    delta.to_csv(out / 'deltas.csv')
    settings = {'config':{k:v for k,v in cfg.items() if k!='paths'},
                'threshold':.85,'labels':str(label_path),'removed':audit.decision.value_counts().to_dict(),
                'limitations':'Exact matches assumed correct; same labels are team judgements. No tuning from expert correlations.'}
    (out/'settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    assert np.isfinite(results[['spearman','p_value','topk_overlap']].to_numpy()).all()
    print(results.to_string(index=False), flush=True)
    print('Saved:',out,flush=True)


if __name__ == '__main__':
    main()
