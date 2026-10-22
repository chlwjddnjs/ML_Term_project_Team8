"""Hybrid filtering: weighted combination of CF and content scores.  Owner: member 3.

final(u, i) = w_u * norm(cf(u, i)) + (1 - w_u) * norm(content(u, i))

w_u = alpha for normal users, and smaller for cold-start users (few train
ratings), so the content score takes over when CF has little to learn from.
The two sub-models are created and fitted here (plumbing is implemented);
the combination rule is TODO(3).
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
        self.alpha = alpha
        self.cold_start_threshold = cold_start_threshold
        self.cf = MatrixFactorization(**(mf_params or {}))
        self.content = ContentBasedRecommender(**(content_params or {}))

    def _fit(self, train, items):
        self.cf.fit(train, items)
        self.content.fit(train, items)
        self.n_train = train.groupby("user_id").size().to_dict()

    def user_weight(self, user_id):
        """Weight of the CF score for this user.

        TODO(3): e.g. alpha * min(1, n_ratings / cold_start_threshold)
        Currently a fixed alpha.
        """
        return self.alpha

    def score(self, user_id, item_ids):
        w = self.user_weight(user_id)
        cf = minmax(self.cf.score(user_id, item_ids))
        cb = minmax(self.content.score(user_id, item_ids))
        return w * cf + (1 - w) * cb
