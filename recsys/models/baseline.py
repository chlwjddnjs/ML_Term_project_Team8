"""Baselines for comparison.  Owner: member 4.

PopularityRecommender is implemented (it is the reference every model must beat).
ItemKNNRecommender (memory-based collaborative filtering) is TODO(4).
"""
import numpy as np

from .base import Recommender


class PopularityRecommender(Recommender):
    """Recommend the wines with the most ratings, for every user.

    Score = number of ratings + mean rating / 10 (the mean only breaks ties).
    Non-personalized, so it does not predict ratings.
    """
    name = "Popularity"

    def _fit(self, train, items):
        stats = train.groupby("item_id")["rating"].agg(["count", "mean"])
        pop = stats["count"] + stats["mean"] / 10
        self.pop = np.array([pop.get(i, 0.0) for i in self.item_ids])

    def score(self, user_id, item_ids):
        return np.array([self.pop[self.item_index[i]] for i in item_ids])


class ItemKNNRecommender(Recommender):
    """Memory-based CF: item-based k-nearest neighbors.

    TODO(4):
      _fit : build the user x item rating matrix (mean-centered per user),
             compute item-item cosine similarity (sklearn cosine_similarity or numpy)
      score: for each target item, weighted average of the user's ratings on
             its n_neighbors most similar rated items:
                 pred(u,i) = mean_u + sum(sim(i,j) * (r_uj - mean_u)) / sum(|sim(i,j)|)
             unknown user -> self.global_mean
    """
    name = "ItemKNN"
    predicts_ratings = True

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def _fit(self, train, items):
        raise NotImplementedError("ItemKNNRecommender._fit - TODO(4)")

    def score(self, user_id, item_ids):
        raise NotImplementedError("ItemKNNRecommender.score - TODO(4)")
