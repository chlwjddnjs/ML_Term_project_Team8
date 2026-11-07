"""Compare Popularity, Hybrid, and a blend of their normalized scores.

Run from the project root: python experiments/compare_popularity_hybrid.py
For a smoke test, pass --data sample. The Slim validation split selects a
50:50 blend. The selected weight is frozen before evaluating the test set.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys.config import load_config
from recsys.data import adapter, cleaner, loader
from recsys.data.split import split_by_user
from recsys.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from recsys.models.hybrid import minmax
from run import build_model


HYBRID_WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
VALIDATION_RATIO = 0.10


def evaluate(items, train, holdout, weights, cfg, max_users=None, *, components=None):
    """Fit once per split and reuse each user's scores across blend weights."""
    k = cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevant_threshold"]
    relevant = holdout.loc[holdout["rating"] >= threshold].groupby("user_id")["item_id"].apply(set)
    users = relevant.index.to_numpy()
    if max_users is not None and len(users) > max_users:
        users = np.random.default_rng(cfg["seed"]).choice(users, max_users, replace=False)
    if not len(users):
        raise ValueError("The evaluation split has no users with a relevant held-out rating")

    if components is None:
        print(f"Training on {len(train):,} ratings and evaluating {len(users):,} users", flush=True)
        components = (
            build_model("popularity", cfg).fit(train, items),
            build_model("hybrid", cfg).fit(train, items),
        )
    popularity, hybrid = components
    totals = np.zeros((len(weights), 3))
    covered_items = [set() for _ in weights]

    for user in users:
        seen = hybrid.seen.get(user, set())
        candidates = np.array([item for item in hybrid.item_ids if item not in seen])
        if not candidates.size:
            continue
        pop_scores = minmax(popularity.score(user, candidates))
        hybrid_scores = minmax(hybrid.score(user, candidates))
        for index, weight in enumerate(weights):
            scores = (1 - weight) * pop_scores + weight * hybrid_scores
            order = np.argsort(-scores, kind="stable")[:k]
            recommendations = candidates[order].tolist()
            positives = relevant.loc[user]
            totals[index] += (
                ndcg_at_k(recommendations, positives, k),
                precision_at_k(recommendations, positives, k),
                recall_at_k(recommendations, positives, k),
            )
            covered_items[index].update(recommendations)

    results = pd.DataFrame(totals / len(users), columns=[f"NDCG@{k}", f"P@{k}", f"R@{k}"])
    results.insert(0, "hybrid_weight", weights)
    results["Coverage"] = [len(covered) / len(items) for covered in covered_items]
    results["n_eval_users"] = len(users)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("slim", "sample"), default="slim")
    args = parser.parse_args()
    cfg = load_config(data_profile=args.data)
    paths = cfg["paths"]
    raw_items, raw_ratings = loader.load_xwines(paths["xwines_wines"], paths["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(raw_items, raw_ratings, cfg)
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    minimum = cfg["min_user_ratings"]
    seed = cfg["seed"]
    train, test = split_by_user(ratings, cfg["split"]["test_ratio"], minimum, seed)
    train_sub, validation = split_by_user(train, VALIDATION_RATIO, minimum, seed)

    out = paths["output_dir"] / "popularity_hybrid" / args.data
    out.mkdir(parents=True, exist_ok=True)
    validation_results = evaluate(items, train_sub, validation, HYBRID_WEIGHTS, cfg)
    ndcg_column = f"NDCG@{cfg['evaluation']['k']}"
    best_index = validation_results[ndcg_column].idxmax()
    best_weight = float(validation_results.loc[best_index, "hybrid_weight"])
    validation_results["selected"] = validation_results.index == best_index
    validation_results.to_csv(out / "validation.csv", index=False)
    print(validation_results.to_string(index=False), flush=True)
    print(f"Selected Hybrid weight {best_weight:.2f} on validation", flush=True)

    # Retrain on the full training split, then compare the frozen choice with both baselines.
    test_weights = tuple(sorted({0.0, best_weight, 1.0}))
    test_results = evaluate(items, train, test, test_weights, cfg, cfg["evaluation"]["max_eval_users"])
    test_results["selected"] = test_results["hybrid_weight"].eq(best_weight)
    test_results.to_csv(out / "test.csv", index=False)
    print(test_results.to_string(index=False), flush=True)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
