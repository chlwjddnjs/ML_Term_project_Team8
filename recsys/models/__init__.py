"""Model registry: maps the names used in config.yaml to model classes."""
from .baseline import ItemKNNRecommender, PopularityRecommender
from .content import ContentBasedRecommender
from .hybrid import HybridRecommender
from .mf import MatrixFactorization

REGISTRY = {
    "popularity": PopularityRecommender,
    "item_knn": ItemKNNRecommender,
    "mf": MatrixFactorization,
    "content": ContentBasedRecommender,
    "hybrid": HybridRecommender,
}
