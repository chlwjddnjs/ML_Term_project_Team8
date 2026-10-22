"""Outlier / error handling on the common schema.  Owner: member 1.

`quality_report` is implemented (use it in data exploration). The three
`clean_*` functions currently only drop exact duplicates; member 1 fills in
the TODO items. Each function must return a NEW DataFrame in the same schema.
"""
import pandas as pd


def quality_report(df, name):
    """Print size, missing values and duplicates of a DataFrame (for EDA).

    List columns are converted to strings before counting duplicates,
    because lists are not hashable.
    """
    hashable = df.apply(lambda c: c.map(str) if c.map(type).eq(list).any() else c)
    print(f"\n[quality] {name}: {len(df):,} rows x {df.shape[1]} cols, "
          f"duplicates={hashable.duplicated().sum():,}")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("  missing values:", ", ".join(f"{c}={n:,}" for c, n in missing.items()))


def clean_items(items):
    """Clean the ITEMS table (X-Wines wines).

    TODO(1):
      - ABV outliers (e.g. <= 0 or > 25) -> NaN or median by wine type
      - unify body/acidity labels into ordered levels (e.g. 1..5) for content features
      - empty grapes / harmonize lists
    """
    out = items.copy()
    return out


def clean_ratings(ratings):
    """Clean the RATINGS table.

    Already done: ratings outside 1-5 are removed.
    TODO(1): duplicated (user_id, item_id) pairs -> keep the latest by date or the mean.
    """
    out = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)].copy()
    removed = len(ratings) - len(out)
    if removed:
        print(f"[cleaner] removed {removed} ratings outside 1-5")
    return out.reset_index(drop=True)


def clean_expert(expert):
    """Clean the EXPERT table (Wine Enthusiast).

    Already done: exact duplicate reviews are removed.
    TODO(1):
      - price outliers (log-scale IQR rule or cap at a percentile)
      - missing country / variety
    """
    key = ["winery", "name", "taster", "points", "description"]
    out = expert.drop_duplicates(subset=key).copy()
    removed = len(expert) - len(out)
    if removed:
        print(f"[cleaner] removed {removed:,} duplicate expert reviews")
    return out.reset_index(drop=True)
