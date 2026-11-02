"""Outlier / error handling on the common schema.  Owner: member 1.

`quality_report` prints the size, missing values and duplicates of a table
(used in data exploration). The three `clean_*` functions fix the problems
found there. Each function returns a NEW DataFrame in the same schema and
prints what it changed, so the effect of cleaning is visible in run.py.
"""
import numpy as np
import pandas as pd

# A table wine has roughly 5-16 % alcohol and fortified wine (port) up to ~22 %.
# Anything above this limit is a spirit (e.g. grappa ~40 %) or a data error.
MAX_WINE_ABV = 25


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


def iqr_bounds(values, k=1.5):
    """Lower / upper outlier limits with the IQR rule.

    IQR = Q3 - Q1 (the width of the middle 50 % of the data).
    Values below Q1 - k*IQR or above Q3 + k*IQR are outliers.
    """
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_items(items):
    """Clean the ITEMS table (X-Wines wines).

    - ABV outliers (<= 0 or > MAX_WINE_ABV) -> median ABV of the same wine type.
      In X-Wines Slim these are 8 grappas / marcs (38-50 %) labelled "Dessert".
      The rows are kept so that no rating is lost; only the ABV is fixed.
    - empty grapes / harmonize lists -> ["Unknown"], so every wine has at
      least one value in the content features.
    - body / acidity: the labels are already consistent in X-Wines
      ("Light-bodied", ..., "High"), so they are only stripped of spaces and
      kept as categories (the content model one-hot encodes them).
    """
    out = items.copy()

    # ABV: to_numeric turns broken values into NaN as well
    abv = pd.to_numeric(out["abv"], errors="coerce")
    bad = abv.isna() | (abv <= 0) | (abv > MAX_WINE_ABV)
    if bad.any():
        names = ", ".join(out.loc[bad, "name"].astype(str).head(5))
        print(f"[cleaner] {bad.sum()} wines with ABV outside 0-{MAX_WINE_ABV}% "
              f"-> median of their type (e.g. {names})")
        # median of the normal wines of each type, e.g. Dessert -> 12.5
        type_median = abv[~bad].groupby(out.loc[~bad, "type"]).median()
        abv[bad] = out.loc[bad, "type"].map(type_median).fillna(abv[~bad].median())
    out["abv"] = abv

    # empty lists -> ["Unknown"]
    for col in ("grapes", "harmonize"):
        empty = out[col].map(len) == 0
        if empty.any():
            print(f"[cleaner] {empty.sum()} wines with empty {col} -> ['Unknown']")
            out.loc[empty, col] = pd.Series([["Unknown"]] * empty.sum(),
                                            index=out.index[empty])

    # body / acidity: remove stray spaces, missing -> "Unknown"
    for col in ("body", "acidity"):
        out[col] = out[col].astype("string").str.strip().fillna("Unknown").astype(object)

    return out.reset_index(drop=True)


def clean_ratings(ratings):
    """Clean the RATINGS table.

    - ratings outside 1-5 are removed.
    - a user who rated the same wine twice: keep only the latest rating
      (sorted by date), so one (user, wine) pair appears once.
    """
    out = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)].copy()
    removed = len(ratings) - len(out)
    if removed:
        print(f"[cleaner] removed {removed} ratings outside 1-5")

    before = len(out)
    out = (out.sort_values("date")
              .drop_duplicates(subset=["user_id", "item_id"], keep="last")
              .sort_index())
    if before - len(out):
        print(f"[cleaner] removed {before - len(out):,} repeated (user, wine) "
              f"ratings, kept the latest")
    return out.reset_index(drop=True)


def clean_expert(expert):
    """Clean the EXPERT table (Wine Enthusiast).

    - exact duplicate reviews are removed.
    - price outliers: prices are very skewed (median $25, max $3,300), so the
      IQR rule is applied on log(price). Prices outside the limits are capped
      to the limit instead of deleting the review (price is not used for
      matching, and the review is still useful).
    - missing country -> "Unknown"
    - reviews without a grape variety are removed (cannot be compared by
      content).
    """
    key = ["winery", "name", "taster", "points", "description"]
    out = expert.drop_duplicates(subset=key).copy()
    removed = len(expert) - len(out)
    if removed:
        print(f"[cleaner] removed {removed:,} duplicate expert reviews")

    # price: IQR on the log scale, then back with exp
    log_price = np.log(out["price"].dropna())
    low, high = (np.exp(b) for b in iqr_bounds(log_price))
    n_out = ((out["price"] < low) | (out["price"] > high)).sum()
    if n_out:
        print(f"[cleaner] capped {n_out:,} price outliers to ${low:.0f}-${high:.0f}")
    out["price"] = out["price"].clip(lower=low, upper=high)   # NaN stays NaN

    # missing country
    n_missing = out["country"].isna().sum()
    if n_missing:
        print(f"[cleaner] {n_missing} reviews without country -> 'Unknown'")
    out["country"] = out["country"].fillna("Unknown")

    # no grape variety
    no_grape = out["grapes"].map(len) == 0
    if no_grape.any():
        print(f"[cleaner] removed {no_grape.sum()} reviews without grape variety")
    out = out[~no_grape]

    return out.reset_index(drop=True)