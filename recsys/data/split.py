"""Train/test split.  Owner: shared (member 4 checks it).

Everybody must use THIS split with the seed from config.yaml, otherwise the
model scores in the final comparison table are not comparable.
"""
import numpy as np
import pandas as pd


def split_by_user(ratings, test_ratio=0.2, min_user_ratings=5, seed=42):
    """Hold out a fraction of EACH user's ratings for testing.

    Users with fewer than `min_user_ratings` ratings stay entirely in train
    (they still help the model learn, and become cold-start users).
    Every test user keeps at least one rating in train.

    Returns:
        (train, test) DataFrames with the RATINGS schema.
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in ratings.groupby("user_id"):
        n = len(group)
        if n < min_user_ratings:
            continue
        n_test = min(max(1, int(round(n * test_ratio))), n - 1)
        test_idx.extend(rng.choice(group.index.to_numpy(), size=n_test, replace=False))
    test_mask = ratings.index.isin(test_idx)
    train = ratings[~test_mask].reset_index(drop=True)
    test = ratings[test_mask].reset_index(drop=True)
    return train, test
