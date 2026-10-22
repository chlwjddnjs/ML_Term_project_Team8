"""Raw data loading.  Owner: shared.

This module ONLY reads files. Renaming, parsing and cleaning are done by
`adapter.py` and `cleaner.py`, so every dataset goes through the same path:

    raw CSV --(loader)--> raw DataFrame --(adapter)--> common schema --(cleaner)--> clean data
"""
from pathlib import Path

import pandas as pd


def _find_file(path: Path, keyword: str) -> Path:
    """Return `path` if it exists, otherwise the first CSV in the same folder
    whose name contains `keyword` (the Google Drive file names may differ)."""
    if path.exists():
        return path
    candidates = sorted(path.parent.glob(f"*{keyword}*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"{path} not found. Download X-Wines Slim (see README) and put the "
            f"CSV files in {path.parent}, or run with --data sample."
        )
    print(f"[loader] {path.name} not found, using {candidates[0].name}")
    return candidates[0]


def load_xwines(wines_path, ratings_path):
    """Load the X-Wines wines table and ratings table as-is.

    Returns:
        (wines, ratings) raw DataFrames with the original X-Wines columns.
    """
    wines = pd.read_csv(_find_file(Path(wines_path), "wines"))
    # Vintage holds values like "N.V." (non-vintage), so keep it as text.
    ratings = pd.read_csv(_find_file(Path(ratings_path), "ratings"),
                          dtype={"Vintage": str})
    return wines, ratings


def load_winemag(path):
    """Load the Wine Enthusiast review CSV, or return None if it is missing.

    The Kaggle file has an unnamed index column, which becomes the row index.
    """
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)
