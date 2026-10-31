"""Wine attribute vectors and cosine-based user profiles (member 3).

Fit encoders on the catalogue; build user profiles using TRAIN ratings only.
Scores are similarities, not predicted ratings.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import Recommender
from ..data.adapter import parse_list


class ItemFeatureEncoder:
    """Reusable feature space: unseen categories are ignored on transform.

    Each categorical/list block has unit row norm before concatenation, so
    many grape/pairing labels do not outweigh a single categorical attribute.
    Missing attributes contribute zero. ABV uses a fixed 0..25 percent scale.
    """
    categorical = ("type", "country", "region", "body", "acidity")
    list_columns = ("grapes", "harmonize")

    def __init__(self, use_text=False):
        self.use_text = use_text

    @staticmethod
    def _column(items, column):
        return items[column] if column in items else pd.Series("", index=items.index)

    def _records(self, items, column):
        records = []
        for value in self._column(items, column):
            values = parse_list(value) if column in self.list_columns else [value]
            labels = {str(v).strip().casefold() for v in values
                      if pd.notna(v) and str(v).strip()}
            records.append({f"{column}={label}": 1.0 for label in labels})
        return records

    def fit(self, items):
        self.encoders = {}
        for column in self.categorical + self.list_columns:
            encoder = DictVectorizer(dtype=np.float64)
            encoder.fit(self._records(items, column))
            self.encoders[column] = encoder
        self.text_encoder = None
        if self.use_text:
            text = self._column(items, "description").fillna("").astype(str)
            # No descriptions (normal X-Wines schema) means no text block.
            if text.str.contains(r"(?u)\b\w\w+\b", regex=True).any():
                self.text_encoder = TfidfVectorizer(max_features=5000, sublinear_tf=True)
                self.text_encoder.fit(text)
        self.feature_names = [name for enc in self.encoders.values()
                              for name in enc.get_feature_names_out()] + ["abv/25"]
        if self.text_encoder is not None:
            self.feature_names += [f"text={s}" for s in self.text_encoder.get_feature_names_out()]
        return self

    def transform(self, items):
        blocks = []
        for column, encoder in self.encoders.items():
            block = encoder.transform(self._records(items, column))
            # sklearn normalize requires at least one feature.
            blocks.append(normalize(block) if block.shape[1] else block)
        abv = pd.to_numeric(self._column(items, "abv"), errors="coerce")
        abv = abv.where((abv > 0) & (abv <= 25)).fillna(0).to_numpy(float) / 25
        blocks.append(sparse.csr_matrix(abv.reshape(-1, 1)))
        if self.text_encoder is not None:
            text = self._column(items, "description").fillna("").astype(str)
            blocks.append(self.text_encoder.transform(text))
        matrix = sparse.hstack(blocks, format="csr")
        return normalize(matrix) if len(items) else matrix


def build_item_features(items, use_text=False):
    """Return (CSR feature matrix, names), preserving catalogue row order.

    For later data use ItemFeatureEncoder.fit(items).transform(other_items),
    or model.feature_encoder.transform(other_items), to reuse the vocabulary.
    """
    encoder = ItemFeatureEncoder(use_text).fit(items)
    return encoder.transform(items), encoder.feature_names


class ContentBasedRecommender(Recommender):
    name = "Content"
    predicts_ratings = False

    def __init__(self, use_text=False):
        self.use_text = use_text

    def _fit(self, train, items):
        self.feature_encoder = ItemFeatureEncoder(self.use_text).fit(items)
        self.item_matrix = self.feature_encoder.transform(items)
        self.feature_names = self.feature_encoder.feature_names
        self.user_profiles = {}
        for user_id, group in train.groupby("user_id"):
            group = group[group["item_id"].isin(self.item_index)]
            if group.empty:
                continue
            idx = [self.item_index[i] for i in group["item_id"]]
            ratings = group["rating"].to_numpy(float)
            weights = ratings - ratings.mean()
            # A single rating / equal ratings have no relative preference.
            # Use their rated wines equally instead of a zero profile.
            if np.allclose(weights, 0):
                weights = np.ones(len(ratings))
            profile = np.asarray(self.item_matrix[idx].T @ weights).ravel()
            norm = np.linalg.norm(profile)
            self.user_profiles[user_id] = profile / norm if norm else profile

    def score(self, user_id, item_ids):
        """Cosine scores in requested order; unknown users/items get zero."""
        item_ids = list(item_ids)
        scores = np.zeros(len(item_ids))
        profile = self.user_profiles.get(user_id)
        if profile is None or not item_ids:
            return scores
        positions = [p for p, iid in enumerate(item_ids) if iid in self.item_index]
        if positions:
            idx = [self.item_index[item_ids[p]] for p in positions]
            scores[positions] = np.asarray(self.item_matrix[idx] @ profile).ravel()
        return np.clip(scores, -1, 1)

    def similar_items(self, item_id, k=10):
        """Return up to k other item IDs, ordered by attribute similarity."""
        if item_id not in self.item_index or k <= 0:
            return []
        idx = self.item_index[item_id]
        scores = (self.item_matrix @ self.item_matrix[idx].T).toarray().ravel()
        scores[idx] = -np.inf
        order = np.argsort(-scores, kind="stable")[:min(k, len(scores) - 1)]
        return self.item_ids[order].tolist()
