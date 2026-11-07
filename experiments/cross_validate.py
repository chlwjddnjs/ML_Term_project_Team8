"""Run 10-fold evaluation on the training portion of X-Wines.

Run: python experiments/cross_validate.py [--data sample] [--models popularity_hybrid]
The original test set remains separate. CSVs contain per-fold metrics and
unweighted means and sample standard deviations across folds.
The popularity/Hybrid blend uses a fixed 50:50 weight in every fold.
"""
import argparse
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compare_popularity_hybrid import evaluate as evaluate_blend
from recsys.config import load_config
from recsys.data import adapter, cleaner, loader
from recsys.data.split import split_by_user
from recsys.eval.evaluator import evaluate_model
from run import REGISTRY, build_model

N_FOLDS = 10
BLEND = "popularity_hybrid"


def assign_folds(ratings, min_user_ratings, seed):
    """Assign eligible rows once, balancing each user's ratings across folds.

    Users below the minimum stay in training in every fold. A minimum of two
    ratings guarantees a training history for every validation user.
    """
    rng = np.random.default_rng(seed)
    assignments = np.full(len(ratings), -1, dtype=int)
    offset = 0
    for indices in ratings.groupby("user_id").indices.values():
        if len(indices) < max(2, min_user_ratings):
            continue
        assignments[rng.permutation(indices)] = (offset + np.arange(len(indices))) % N_FOLDS
        offset += len(indices)
    if offset < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} eligible ratings for cross-validation")
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", choices=["slim", "sample"])
    parser.add_argument("--models", nargs="+", choices=[*REGISTRY, BLEND])
    args = parser.parse_args()
    cfg = load_config(args.config, args.data)
    seed = cfg["seed"]
    minimum = cfg["min_user_ratings"]
    model_names = list(dict.fromkeys(args.models or [*cfg["models"], BLEND]))

    wines, raw = loader.load_xwines(cfg["paths"]["xwines_wines"], cfg["paths"]["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(wines, raw, cfg)
    items, ratings = cleaner.clean_items(items), cleaner.clean_ratings(ratings)
    cv_ratings, held_out = split_by_user(ratings, cfg["split"]["test_ratio"], minimum, seed)
    assignments = assign_folds(cv_ratings, minimum, seed)
    print(f"CV pool: {len(cv_ratings):,} ratings; original test kept aside: {len(held_out):,}",
          flush=True)

    rows = []
    for fold in range(N_FOLDS):
        is_validation = assignments == fold
        train, validation = cv_ratings[~is_validation], cv_ratings[is_validation]
        print(f"Fold {fold + 1}/{N_FOLDS}: train={len(train):,}, validation={len(validation):,}",
              flush=True)
        fitted = {}
        fold_info = {"fold": fold + 1, "train_rows": len(train),
                     "validation_rows": len(validation)}
        for name in model_names:
            if name == BLEND:
                continue
            model = build_model(name, cfg).fit(train, items)
            summary, _ = evaluate_model(model, train, validation, cfg["evaluation"], seed)
            rows.append({**fold_info, **summary})
            if name in ("popularity", "hybrid"):
                fitted[name] = model
        if BLEND in model_names:
            components = tuple(fitted[name] if name in fitted else build_model(name, cfg).fit(train, items)
                               for name in ("popularity", "hybrid"))
            start = perf_counter()
            blend = evaluate_blend(items, train, validation, (0.5,), cfg,
                                   cfg["evaluation"]["max_eval_users"], components=components)
            scores = blend.drop(columns="hybrid_weight").iloc[0].to_dict()
            scores["n_eval_users"] = int(scores["n_eval_users"])
            rows.append({**fold_info, "model": "Popularity + Hybrid (50:50)", **scores,
                         "eval_seconds": round(perf_counter() - start, 2)})

    folds = pd.DataFrame(rows)
    k = cfg["evaluation"]["k"]
    metrics = ["RMSE", "MAE", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage"]
    scores = folds.reindex(columns=["model", *metrics])
    summary = scores.groupby("model", sort=False).agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    out = cfg["paths"]["output_dir"] / "cross_validation" / cfg["data"]
    if args.models:
        out = out / ("models_" + "_".join(model_names))
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "folds.csv", index=False)
    summary.to_csv(out / "summary.csv")
    print(summary.filter(like="_mean").to_string())
    print(f"Saved fold results and summary to {out}")


if __name__ == "__main__":
    main()
