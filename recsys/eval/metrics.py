"""Evaluation metrics.  Owner: member 4 (implemented; add new ones here).

Rating metrics compare predicted and true ratings.
Ranking metrics compare a top-k list with the set of relevant items
(held-out items the user rated >= relevant_threshold).
"""
import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error (penalizes large errors more)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended, relevant, k):
    """Fraction of the top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for i in recommended[:k] if i in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Fraction of the relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for i in recommended[:k] if i in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Normalized discounted cumulative gain with binary relevance.

    DCG  = sum over positions p (1-based) of rel_p / log2(p + 1)
    IDCG = DCG of the ideal ordering (all relevant items first)
    """
    dcg = sum(1.0 / np.log2(p + 2) for p, i in enumerate(recommended[:k]) if i in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(p + 2) for p in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations, n_items):
    """Share of the catalogue that appears in at least one user's top-k
    (low coverage = the model keeps recommending the same popular wines)."""
    unique = set()
    for rec in all_recommendations:
        unique.update(rec)
    return len(unique) / n_items if n_items else 0.0
