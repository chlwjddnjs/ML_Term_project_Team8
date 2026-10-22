"""Runs the same evaluation for every model.  Owner: shared (member 4 maintains).

evaluate_model() returns
  - summary : dict with RMSE/MAE (rating models) and P@K/R@K/NDCG@K/coverage
  - per_user: DataFrame, one row per evaluated user (for cold-start analysis)
"""
import time

import numpy as np
import pandas as pd

from . import metrics


def evaluate_rating(model, test):
    """RMSE / MAE on every held-out rating (only for rating models)."""
    preds = []
    for user_id, group in test.groupby("user_id"):
        scores = model.score(user_id, group["item_id"].to_numpy())
        preds.append(pd.Series(np.clip(scores, 1, 5), index=group.index))
    pred = pd.concat(preds).reindex(test.index)
    return {"RMSE": metrics.rmse(test["rating"], pred),
            "MAE": metrics.mae(test["rating"], pred)}


def evaluate_ranking(model, train, test, k=10, relevant_threshold=4.0,
                     max_users=2000, seed=42):
    """Top-k ranking metrics on users that have at least one relevant test item."""
    relevant = (test[test["rating"] >= relevant_threshold]
                .groupby("user_id")["item_id"].apply(set))
    users = relevant.index.to_numpy()
    if len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
    n_train = train.groupby("user_id").size()

    rows, all_recs = [], []
    for u in users:
        rec = model.recommend(u, k=k)
        rel = relevant[u]
        all_recs.append(rec)
        rows.append({"user_id": u,
                     "n_train": int(n_train.get(u, 0)),
                     f"P@{k}": metrics.precision_at_k(rec, rel, k),
                     f"R@{k}": metrics.recall_at_k(rec, rel, k),
                     f"NDCG@{k}": metrics.ndcg_at_k(rec, rel, k)})
    per_user = pd.DataFrame(rows)
    if per_user.empty:
        return {}, per_user
    summary = per_user[[f"P@{k}", f"R@{k}", f"NDCG@{k}"]].mean().to_dict()
    summary["Coverage"] = metrics.coverage(all_recs, len(model.item_ids))
    summary["n_eval_users"] = len(per_user)
    return summary, per_user


def evaluate_model(model, train, test, eval_cfg, seed=42):
    """Full evaluation of one fitted model."""
    start = time.time()
    summary = {"model": model.name}
    if model.predicts_ratings:
        summary.update(evaluate_rating(model, test))
    ranking, per_user = evaluate_ranking(
        model, train, test, k=eval_cfg["k"],
        relevant_threshold=eval_cfg["relevant_threshold"],
        max_users=eval_cfg["max_eval_users"], seed=seed)
    summary.update(ranking)
    summary["eval_seconds"] = round(time.time() - start, 2)
    return summary, per_user
