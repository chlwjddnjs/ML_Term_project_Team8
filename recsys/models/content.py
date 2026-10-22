"""Content-based filtering with wine attributes.  Owner: member 3.

Idea:
  1. every wine -> feature vector (type, grapes, country, region, body,
     acidity, food pairing, ABV)  [vector space model]
  2. user profile = average of the vectors of wines the user liked,
     weighted by (rating - user mean)
  3. score = cosine similarity(user profile, wine vector)

Because it only needs item attributes, the same feature function can embed
Wine Enthusiast wines too -> used for the cross-dataset validation.
"""
import numpy as np

from .base import Recommender


def build_item_features(items, use_text=False):
    """Turn the ITEMS table into a 2-D feature matrix (one row per wine).

    TODO(3):
      - categorical (type, country, region, body, acidity): one-hot
        (pandas.get_dummies or sklearn OneHotEncoder)
      - lists (grapes, harmonize): multi-hot (sklearn MultiLabelBinarizer)
      - abv: scale to 0-1
      - use_text: TF-IDF on text (sklearn TfidfVectorizer)
      - weight blocks so one block (e.g. 300 regions) does not dominate
    Returns:
        (matrix, feature_names)  - matrix rows follow the order of `items`.
    Keep the fitted encoders (return them or store them) so Wine Enthusiast
    wines can be transformed into the SAME feature space later.
    """
    raise NotImplementedError("build_item_features - TODO(3)")


class ContentBasedRecommender(Recommender):
    name = "Content"
    predicts_ratings = False      # scores are similarities, not ratings

    def __init__(self, use_text=False):
        self.use_text = use_text

    def _fit(self, train, items):
        """TODO(3):
        self.item_matrix, self.feature_names = build_item_features(items, self.use_text)
        build self.user_profiles: {user_id: profile vector} from train ratings
        """
        raise NotImplementedError("ContentBasedRecommender._fit - TODO(3)")

    def score(self, user_id, item_ids):
        """TODO(3): cosine similarity between the user profile and each item.
        Unknown user (no profile) -> return zeros (hybrid falls back to CF/popularity).
        """
        raise NotImplementedError("ContentBasedRecommender.score - TODO(3)")

    def similar_items(self, item_id, k=10):
        """TODO(3): k most similar wines to one wine (for the demo / analysis)."""
        raise NotImplementedError("ContentBasedRecommender.similar_items - TODO(3)")
