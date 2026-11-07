"""Compare cross-dataset validation before/after removing reviewed bad matches.

Run: python experiments/compare_clean_matches.py [--include-blend]
Uses the same split/settings as run.py. Each model is fitted once and its
quality scores are reused for both match policies. No scores select labels.
With --include-blend, also export catalogue-level Top-K lists and metrics
for Hybrid and the 50:50 blend against the experts' Top-K wines.
Original outputs/cross_dataset.csv is preserved. Exact matches are retained;
fuzzy-only matches require a 'same' label. Missing labels fail explicitly.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compare_popularity_hybrid import PopularityHybridBlend
from recsys.config import load_config
from recsys.data import adapter, cleaner, loader
from recsys.data.split import split_by_user
from recsys.eval import cross_dataset
from recsys.eval.metrics import ndcg_at_k, recall_at_k
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


def expert_topk_comparison(model_scores, expert_scores, items, k):
    """Rank the matched catalogue and use the experts' top wines as binary relevance."""
    first_scores = next(iter(model_scores.values()))
    common = first_scores.index.intersection(expert_scores.index)
    for scores in model_scores.values():
        common = common.intersection(scores.index)
    if len(common) < k:
        raise ValueError(f'Expert top-{k} evaluation requires at least {k} matched wines')

    expert_top = expert_scores.loc[common].nlargest(k)
    relevant = set(expert_top.index)
    results, rankings = [], []
    top_lists = {'Wine Enthusiast (reference)': expert_top}
    for name, scores in model_scores.items():
        top = scores.loc[common].nlargest(k)
        recommended = top.index.tolist()
        top_lists[name] = top
        results.append({'model': name, 'n_candidates': len(common), 'n_expert_relevant': len(relevant),
                        f'hits@{k}': len(relevant.intersection(recommended)),
                        f'NDCG@{k}': ndcg_at_k(recommended, relevant, k),
                        f'Recall@{k}': recall_at_k(recommended, relevant, k)})
    for name, top in top_lists.items():
        for rank, (item_id, score) in enumerate(top.items(), start=1):
            rankings.append({'model': name, 'rank': rank, 'item_id': item_id,
                             'score': score, 'expert_score': expert_scores.loc[item_id],
                             'in_expert_top_k': item_id in relevant})
    rankings = pd.DataFrame(rankings).merge(
        items[['item_id', 'name', 'winery']], on='item_id', validate='many_to_one')
    return pd.DataFrame(results), rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-blend', action='store_true',
                        help='Also evaluate the fixed 50:50 blend in a separate output directory')
    args = parser.parse_args()
    cfg = load_config()
    out = cfg['paths']['output_dir'] / ('clean_matches_with_blend' if args.include_blend else 'clean_matches')
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
    model_names = list(cfg['models'])
    if args.include_blend:
        model_names = list(dict.fromkeys([*model_names, 'popularity', 'hybrid']))
    components = {}
    for name in model_names:
        print('Training once:', name, flush=True)
        model = build_model(name, cfg).fit(train, items)
        quality[model.name] = cross_dataset.model_quality_per_item(model, train, items.item_id)
        if args.include_blend and name in ('popularity', 'hybrid'):
            components[name] = model
    if args.include_blend:
        blend = PopularityHybridBlend(components['popularity'], components['hybrid'])
        quality[blend.name] = cross_dataset.model_quality_per_item(blend, train, items.item_id)
        expert_metrics, expert_lists = expert_topk_comparison(
            {name: quality[name] for name in (components['hybrid'].name, blend.name)},
            expert_after, items, cfg['evaluation']['k'])
        expert_metrics.to_csv(out / 'expert_topk_metrics.csv', index=False)
        expert_lists.to_csv(out / 'expert_topk_rankings.csv', index=False, encoding='utf-8-sig')
        print(expert_metrics.to_string(index=False), flush=True)
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
                'include_blend':args.include_blend,
                'threshold':.85,'labels':str(label_path),'removed':audit.decision.value_counts().to_dict(),
                'limitations':'Exact matches assumed correct; same labels are team judgements. No tuning from expert correlations.'}
    (out/'settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    assert np.isfinite(results[['spearman','p_value','topk_overlap']].to_numpy()).all()
    print(results.to_string(index=False), flush=True)
    print('Saved:',out,flush=True)


if __name__ == '__main__':
    main()
