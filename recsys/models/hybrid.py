"""Hybrid filtering: weighted combination of CF and content scores.  Owner: member 3.

final(u, i) = w_u * norm(cf(u, i)) + (1 - w_u) * norm(content(u, i))

w_u = alpha for normal users, and smaller for cold-start users (few train
ratings), so the content score takes over when CF has little to learn from.
The CF weight increases linearly with the number of training ratings until
it reaches alpha at cold_start_threshold.
"""
import numpy as np

from .base import Recommender
from .content import ContentBasedRecommender
from .mf import MatrixFactorization


def minmax(x):
    """Scale an array to 0-1 (constant array -> all zeros)."""
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


class HybridRecommender(Recommender):
    name = "Hybrid"
    predicts_ratings = False

    def __init__(self, alpha=0.5, cold_start_threshold=5, mf_params=None, content_params=None):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not np.isfinite(cold_start_threshold) or cold_start_threshold <= 0:
            raise ValueError("cold_start_threshold must be positive")
        self.alpha = alpha
        self.cold_start_threshold = cold_start_threshold
        self.cf = MatrixFactorization(**(mf_params or {}))
        self.content = ContentBasedRecommender(**(content_params or {}))

    def _fit(self, train, items):
        self.cf.fit(train, items)
        self.content.fit(train, items)
        self.n_train = train.groupby("user_id").size().to_dict()

    def user_weight(self, user_id):
        """CF weight based only on this user's training history.

        Below the threshold, fewer ratings mean more content weight (1-w).
        At/above the threshold, use alpha. An unseen user has CF weight zero;
        without a content profile either, scores remain tied at zero.
        """
        n_ratings = self.n_train.get(user_id, 0)
        return self.alpha * min(1.0, n_ratings / self.cold_start_threshold)

    def score(self, user_id, item_ids):
        w = self.user_weight(user_id)
        cf = minmax(self.cf.score(user_id, item_ids))
        cb = minmax(self.content.score(user_id, item_ids))
        return w * cf + (1 - w) * cb
