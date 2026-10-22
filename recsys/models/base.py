"""Common model interface.  Owner: shared - DO NOT change without telling the team.

Every model inherits `Recommender` and implements only two methods:

    _fit(self, train, items)        learn from the train RATINGS + ITEMS tables
    score(self, user_id, item_ids)  return one number per item (higher = better)

Everything else (recommend, predict, index maps, seen items) is provided here,
so the evaluation code works the same way for every model.

Usage:
    model = SomeModel(**params).fit(train, items)
    model.recommend(user_id, k=10)       # -> list of item_id
    model.predict(user_id, item_id)      # -> float (rating models only)
"""
from abc import ABC, abstractmethod

import numpy as np


class Recommender(ABC):
    name = "base"
    # True if score() returns a rating on the 1-5 scale (RMSE/MAE are then computed)
    predicts_ratings = False

    def fit(self, train, items):
        """Register users/items, then call the model-specific `_fit`."""
        self.items = items
        self.item_ids = items["item_id"].to_numpy()
        self.item_index = {iid: i for i, iid in enumerate(self.item_ids)}
        users = train["user_id"].unique()
        self.user_index = {uid: i for i, uid in enumerate(users)}
        self.seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
        self.global_mean = float(train["rating"].mean())
        self.history = []          # per-epoch logs, e.g. {"epoch": 1, "train_rmse": 0.9}
        self._fit(train, items)
        return self

    @abstractmethod
    def _fit(self, train, items):
        """Learn model parameters. Use self.user_index / self.item_index."""

    @abstractmethod
    def score(self, user_id, item_ids):
        """Return a numpy array of scores, one per item in `item_ids`.

        Must also work for a user_id that was not in train (cold start).
        """

    # ----- provided methods -------------------------------------------------
    def predict(self, user_id, item_id):
        """Predicted rating (1-5) for one user-item pair."""
        if not self.predicts_ratings:
            raise TypeError(f"{self.name} does not predict ratings")
        return float(np.clip(self.score(user_id, [item_id])[0], 1, 5))

    def recommend(self, user_id, k=10, exclude_seen=True):
        """Top-k item_ids for a user, excluding items rated in train."""
        candidates = self.item_ids
        if exclude_seen:
            seen = self.seen.get(user_id, set())
            candidates = np.array([i for i in self.item_ids if i not in seen])
        if len(candidates) == 0:
            return []
        scores = np.asarray(self.score(user_id, candidates), dtype=float)
        top = np.argsort(-scores, kind="stable")[:k]
        return candidates[top].tolist()
