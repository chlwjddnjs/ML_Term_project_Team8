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

    Training (_fit):
      1. Build the user x item rating matrix R (missing ratings = 0).
      2. Mean-center every user's ratings: D[u, j] = r_uj - mean_u
         (removes "this user always gives high scores" bias).
      3. Item-item cosine similarity on the columns of D
         ("adjusted cosine"):  sim(i, j) = D[:, i] . D[:, j] / (|D[:, i]| |D[:, j]|)

    Prediction (score), for user u and target wine i:
      N = the n_neighbors wines rated by u that are most similar to i (sim > 0)
      pred(u, i) = mean_u + sum_{j in N} sim(i, j) * (r_uj - mean_u) / sum_{j in N} |sim(i, j)|
      - no positive neighbor  -> mean_u
      - unknown user          -> global mean (cold start)
    """
    name = "ItemKNN"
    predicts_ratings = True

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def _fit(self, train, items):
        n_users, n_items = len(self.user_index), len(self.item_ids)

        # Row / column number of every train rating (wines not in `items` are ignored)
        u = train["user_id"].map(self.user_index).to_numpy()
        i = train["item_id"].map(self.item_index)
        keep = i.notna().to_numpy()
        u, i = u[keep], i[keep].astype(int).to_numpy()
        r = train["rating"].to_numpy(dtype=float)[keep]

        # 1-2. mean per user and mean-centered deviations
        sums = np.bincount(u, weights=r, minlength=n_users)
        counts = np.bincount(u, minlength=n_users)
        self.user_mean = np.divide(sums, counts, out=np.full(n_users, self.global_mean),
                                   where=counts > 0)
        dev = r - self.user_mean[u]

        # 3. adjusted cosine similarity between items (dense: ~1K wines is small)
        D = np.zeros((n_users, n_items), dtype=np.float32)
        D[u, i] = dev
        norms = np.linalg.norm(D, axis=0)
        norms[norms == 0] = 1.0                       # items without ratings -> sim 0
        sim = (D.T @ D) / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)                    # a wine is not its own neighbor
        self.sim = sim.astype(np.float32)

        # Per user: which items they rated (column numbers) and their deviations
        # (sort once, then slice: every slice is a view of the same two arrays)
        order = np.argsort(u, kind="stable")
        i_sorted, dev_sorted = i[order], dev[order]
        starts = np.searchsorted(u[order], np.arange(n_users + 1))
        self.user_items = [i_sorted[starts[x]:starts[x + 1]] for x in range(n_users)]
        self.user_devs = [dev_sorted[starts[x]:starts[x + 1]] for x in range(n_users)]

    def score(self, user_id, item_ids):
        item_ids = np.asarray(item_ids)
        if user_id not in self.user_index:            # cold start: no ratings at all
            return np.full(len(item_ids), self.global_mean)
        x = self.user_index[user_id]
        mean_u = self.user_mean[x]
        rated, devs = self.user_items[x], self.user_devs[x]

        targets = np.array([self.item_index.get(t, -1) for t in item_ids])
        preds = np.full(len(item_ids), mean_u)
        known = targets >= 0
        if len(rated) == 0 or not known.any():
            return preds

        S = self.sim[np.ix_(targets[known], rated)]   # (n_targets, n_rated)
        S = np.where(S > 0, S, 0.0)                   # only positively similar wines
        k = self.n_neighbors
        if S.shape[1] > k:                            # keep the k most similar per row
            cut = np.argpartition(-S, k - 1, axis=1)[:, :k]
            mask = np.zeros_like(S, dtype=bool)
            np.put_along_axis(mask, cut, True, axis=1)
            S = np.where(mask, S, 0.0)

        num = S @ devs
        den = np.abs(S).sum(axis=1)
        preds[known] = mean_u + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return preds
