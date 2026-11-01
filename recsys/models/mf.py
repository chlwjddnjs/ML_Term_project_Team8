"""Model-based collaborative filtering: Matrix Factorization.  Owner: member 2.

Prediction:      r_hat(u, i) = mu + b_u + b_i + p_u . q_i
Loss (per pair): (r - r_hat)^2 + reg * (|p_u|^2 + |q_i|^2) + reg_bias * (b_u^2 + b_i^2)
                 + bias_damping * b_i^2 counted ONCE per wine (not once per rating)
Training:        stochastic gradient descent (SGD), e = r - r_hat
                 b_u += lr * (e - reg_bias * b_u)
                 b_i += lr * (e - (reg_bias + bias_damping / n_i) * b_i)
                 p_u += lr * (e * q_i - reg * p_u)
                 q_i += lr * (e * p_u - reg * q_i)   (p_u = value BEFORE its update)
                 (the constant 2 from differentiating the squares is folded into lr;
                  n_i = number of train ratings of wine i)

What each part means:
  mu       global mean rating of the train set
  b_u      user bias: this user rates higher / lower than average
  b_i      item bias: this wine is rated higher / lower than average
           (also used as the model's "wine quality" in cross_dataset.py)
  p_u, q_i n_factors-long "taste" vectors of the user and the wine;
           their dot product says how well the wine fits this user's taste
  reg      L2 regularization of p_u, q_i: keeps them small (less over-fitting)
  reg_bias L2 regularization of b_u, b_i (default: same as reg). Because SGD
           applies it once per rating, it shrinks EVERY wine's bias by about the
           same factor, 1 / (1 + reg_bias), so the ranking of wines hardly changes.
  bias_damping  (lambda, default 0 = off) count-aware shrinkage of b_i. The pull
           is split over the wine's n_i ratings, so after training roughly
               b_i ~ (sum of the wine's residuals) / (n_i + lambda)
           With lambda 10 (the value in config.yaml) a wine rated by 15 people
           keeps about 15 / (15 + 10) = 60% of its bias; one rated by 528 people
           keeps about 98%. A few high ratings are no longer enough to make a
           wine everybody's top pick.

Reference (cite in the final report if you use it):
  - Hints in the term project specification (towardsdatascience MF articles)
  - Koren, Bell & Volinsky (2009), "Matrix Factorization Techniques for
    Recommender Systems", IEEE Computer.
"""
import numpy as np

from .base import Recommender


class MatrixFactorization(Recommender):
    name = "MF"
    predicts_ratings = True

    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.05, n_epochs=20, seed=42,
                 reg_bias=None, bias_damping=0.0):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.reg_bias = reg if reg_bias is None else reg_bias   # None -> same as reg
        self.bias_damping = bias_damping                         # 0 -> off (old behaviour)
        self.valid = None          # optional validation ratings, see set_validation()

    def set_validation(self, valid):
        """Also log the RMSE on `valid` (a RATINGS table) after every epoch.

        For the hyper-parameter experiments only (train vs validation curve ->
        over-fitting plot, best n_epochs). run.py does not call it, so the final
        evaluation is not affected. `valid` must be held out from the TRAIN set
        (see experiments/tune_mf.py) - never the test set, or the test score
        would no longer be an honest final score.
        Usage:  MatrixFactorization(**params).set_validation(valid).fit(train_sub, items)
        """
        self.valid = valid
        return self

    # ------------------------------------------------------------------ training
    def _fit(self, train, items):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = len(self.user_index), len(self.item_ids)

        # 1. Initialize: small random taste vectors, zero biases.
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))    # user factors
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))    # item factors
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        # ids -> row numbers of P / Q (ratings of wines missing from `items` are skipped)
        u, i = self._to_rows(train)
        r = train["rating"].to_numpy(dtype=float)
        keep = i >= 0
        u, i, r = u[keep], i[keep], r[keep]
        if self.valid is not None:
            vu, vi = self._to_rows(self.valid)
            vr = self.valid["rating"].to_numpy(dtype=float)

        # Short local names: the loop below reads like the formulas in the docstring
        # (and local variables are faster in Python than self.xxx lookups).
        P, Q, b_u, b_i = self.P, self.Q, self.b_u, self.b_i
        mu, lr, reg, reg_b = self.global_mean, self.learning_rate, self.reg, self.reg_bias

        # Pull on each wine's bias per update: reg_bias + bias_damping / n_i.
        # Spread over the wine's n_i ratings, the extra part adds up to bias_damping
        # per epoch whatever n_i is, so rarely rated wines are shrunk the most.
        n_ratings = np.bincount(i, minlength=n_items)            # train ratings per wine
        damp = np.divide(self.bias_damping, n_ratings, out=np.zeros(n_items),
                         where=n_ratings > 0)
        pull_i = reg_b + damp

        for epoch in range(1, self.n_epochs + 1):
            # 2. One SGD step per train rating, in a new random order every epoch.
            for n in rng.permutation(len(r)):
                uu, ii = u[n], i[n]
                p_u, q_i = P[uu], Q[ii]       # row views: changing them changes P and Q
                e = r[n] - (mu + b_u[uu] + b_i[ii] + p_u @ q_i)   # prediction error

                b_u[uu] += lr * (e - reg_b * b_u[uu])
                b_i[ii] += lr * (e - pull_i[ii] * b_i[ii])
                p_old = p_u.copy()            # q_i's update must use p_u before the change
                p_u += lr * (e * q_i - reg * p_u)
                q_i += lr * (e * p_old - reg * q_i)

            # 3. Log the error after every epoch -> outputs/learning_curve_MF.png
            log = {"epoch": epoch, "train_rmse": self._rmse(u, i, r)}
            if self.valid is not None:
                log["valid_rmse"] = self._rmse(vu, vi, vr)
            self.history.append(log)

    # ---------------------------------------------------------------- prediction
    def score(self, user_id, item_ids):
        """Predicted rating of `user_id` for every wine in `item_ids` (vectorized).

        known user  : mu + b_u + b_i + Q[idx] @ P[u]
        unknown user: mu + b_i               (cold start: no user factors yet)
        unknown wine: mu
        """
        idx = np.array([self.item_index.get(x, -1) for x in item_ids], dtype=int)
        known = idx >= 0                                 # wine is in the catalogue
        scores = np.full(len(idx), self.global_mean)     # start from mu
        scores[known] += self.b_i[idx[known]]            # + wine bias
        if user_id in self.user_index:                   # + user bias + taste match
            x = self.user_index[user_id]
            scores[known] += self.b_u[x] + self.Q[idx[known]] @ self.P[x]
        return scores

    # ------------------------------------------------------------------- helpers
    def _to_rows(self, ratings):
        """User / item ids -> row numbers of P / Q (-1 = not seen in train)."""
        u = ratings["user_id"].map(self.user_index).fillna(-1).astype(int).to_numpy()
        i = ratings["item_id"].map(self.item_index).fillna(-1).astype(int).to_numpy()
        return u, i

    def _rmse(self, u, i, r):
        """RMSE of many (user row, item row, rating) triples at once.
        Same formula as score(); predictions are clipped to 1-5 like the evaluator."""
        pred = np.full(len(r), self.global_mean)
        known_i = i >= 0
        pred[known_i] += self.b_i[i[known_i]]
        both = known_i & (u >= 0)
        pred[both] += self.b_u[u[both]] + np.sum(self.P[u[both]] * self.Q[i[both]], axis=1)
        pred = np.clip(pred, 1, 5)
        return float(np.sqrt(np.mean((r - pred) ** 2)))
