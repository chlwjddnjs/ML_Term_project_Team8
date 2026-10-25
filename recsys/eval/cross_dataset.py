"""Cross-dataset validation: X-Wines models vs Wine Enthusiast experts.
Owner: member 4 (with member 1 for matching).

Question: do wines our model ranks high also get high expert points?

Steps:
  1. member 1's match_items() links X-Wines wines to expert reviews
  2. average expert rating per matched X-Wines wine
  3. compare with a model-side quality score per wine
     (e.g. mean predicted rating over users, or the MF item bias b_i)
  4. rank_agreement() -> Spearman correlation + top-k overlap
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_agreement(model_scores, expert_scores, k=10):
    """Agreement between two scorings of the SAME wines (implemented).

    Args:
        model_scores, expert_scores: pandas Series indexed by item_id.
    Returns:
        dict(spearman, p_value, topk_overlap, n_items)
        topk_overlap = |top-k by model  intersect  top-k by experts| / k
    """
    common = model_scores.index.intersection(expert_scores.index)
    a, b = model_scores.loc[common], expert_scores.loc[common]
    if len(common) < 3:
        return {"spearman": np.nan, "p_value": np.nan, "topk_overlap": np.nan,
                "n_items": len(common)}
    rho, p = spearmanr(a, b)
    k = min(k, len(common))
    overlap = len(set(a.nlargest(k).index) & set(b.nlargest(k).index)) / k
    return {"spearman": float(rho), "p_value": float(p),
            "topk_overlap": overlap, "n_items": len(common)}


def expert_score_per_item(matches, expert):
    """Average converted expert rating (1-5) per matched X-Wines item_id."""
    joined = matches.merge(expert[["expert_id", "rating"]], on="expert_id")
    return joined.groupby("item_id")["rating"].mean()


def model_quality_per_item(model, train, item_ids, n_users=500, seed=42):
    """One "how good is this wine" score per wine from a fitted model.

    The expert data has no users, so a user-specific prediction cannot be
    compared with it directly. We turn the model into a single score per wine:
      - MF (has an item bias b_i): mu + b_i, the wine's quality after removing
        user effects - exactly the model's notion of "generally good wine".
      - any other model: mean of model.score(u, items) over a random sample of
        `n_users` train users ("how much the average user would like it").
    Only the ORDER matters afterwards (Spearman, top-k overlap), so the scale
    of the score does not need to match the 1-5 rating scale.

    Returns:
        pandas Series indexed by item_id.
    """
    item_ids = np.asarray(item_ids)

    b_i = getattr(model, "b_i", None)
    if b_i is not None and hasattr(model, "item_index"):
        idx = [model.item_index[i] for i in item_ids]
        values = model.global_mean + np.asarray(b_i)[idx]
        return pd.Series(values, index=item_ids, name=model.name)

    users = train["user_id"].unique()
    if len(users) > n_users:
        users = np.random.default_rng(seed).choice(users, size=n_users, replace=False)
    total = np.zeros(len(item_ids))
    for u in users:
        total += np.asarray(model.score(u, item_ids), dtype=float)
    return pd.Series(total / len(users), index=item_ids, name=model.name)
