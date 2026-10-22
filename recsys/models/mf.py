"""Model-based collaborative filtering: Matrix Factorization.  Owner: member 2.

Prediction:      r_hat(u, i) = mu + b_u + b_i + p_u . q_i
Loss (per pair): (r - r_hat)^2 + reg * (|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2)
Training:        stochastic gradient descent (SGD), e = r - r_hat
                 b_u += lr * (e - reg * b_u)
                 b_i += lr * (e - reg * b_i)
                 p_u += lr * (e * q_i - reg * p_u)
                 q_i += lr * (e * p_u - reg * q_i)

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

    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.05, n_epochs=20, seed=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

    def _fit(self, train, items):
        """TODO(2):
        1. rng = np.random.default_rng(self.seed)
           P   = rng.normal(0, 0.1, (n_users, n_factors))   # user factors
           Q   = rng.normal(0, 0.1, (n_items, n_factors))   # item factors
           b_u = zeros(n_users), b_i = zeros(n_items)
           (use self.user_index / self.item_index to turn ids into row numbers)
        2. for each epoch: shuffle the train rows, apply the SGD updates above
        3. after each epoch: self.history.append({"epoch": e, "train_rmse": ...})
           (optional: also pass a validation set to draw a validation curve)
        4. store the learned arrays as self.P, self.Q, self.b_u, self.b_i
        """
        raise NotImplementedError("MatrixFactorization._fit - TODO(2)")

    def score(self, user_id, item_ids):
        """TODO(2): vectorized prediction for many items at once.

        known user  : mu + b_u + b_i + Q[idx] @ P[u]
        unknown user: mu + b_i              (cold start: no user factors)
        """
        raise NotImplementedError("MatrixFactorization.score - TODO(2)")
