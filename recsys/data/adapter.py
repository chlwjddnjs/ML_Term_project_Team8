"""DatasetAdapter: converts each raw dataset into one common schema and links
the two datasets.  Owner: member 1 (data & alignment).

Common schema (every other module relies ONLY on these column names):

    ITEMS   : item_id, name, winery, country, region, type, grapes(list),
              harmonize(list), abv, body, acidity
    RATINGS : user_id, item_id, rating (1-5 scale), date
    EXPERT  : expert_id, name, winery, country, region, grapes(list),
              price, points (raw 80-100), rating (converted to 1-5),
              description, taster

The column renaming is driven by `config.yaml -> adapter`, so a new dataset is
added by writing a new column map instead of changing code.

Already implemented (shared plumbing):
    apply_column_map, xwines_to_common, winemag_to_common (renaming only),
    scale_points(method="linear"), match_items(method="exact")
TODO for member 1 (marked "TODO(1)"):
    parse_winemag_title, normalize_text, scale_points(method="quantile"),
    match_items(method="fuzzy")
"""
import ast

import numpy as np
import pandas as pd

ITEM_COLUMNS = ["item_id", "name", "winery", "country", "region", "type",
                "grapes", "harmonize", "abv", "body", "acidity"]
RATING_COLUMNS = ["user_id", "item_id", "rating", "date"]
EXPERT_COLUMNS = ["expert_id", "name", "winery", "country", "region", "grapes",
                  "price", "points", "rating", "description", "taster"]


# ---------------------------------------------------------------------------
# Generic helpers (implemented)
# ---------------------------------------------------------------------------
def parse_list(value):
    """Turn a string such as "['Merlot', 'Syrah']" into a Python list.

    Uses ast.literal_eval, which only evaluates Python literals (safe, unlike
    eval). Non-parsable strings become a one-element list; missing -> [].
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return [str(parsed)]


def apply_column_map(df, column_map):
    """Select and rename columns using {common_name: raw_name}.

    A raw column that does not exist becomes an all-NaN column, so downstream
    code can always rely on the common column names.
    """
    out = pd.DataFrame(index=df.index)
    for common, raw in column_map.items():
        if raw in df.columns:
            out[common] = df[raw]
        else:
            print(f"[adapter] warning: column '{raw}' not found -> '{common}' set to NaN")
            out[common] = np.nan
    return out


# ---------------------------------------------------------------------------
# X-Wines (model-building dataset)
# ---------------------------------------------------------------------------
def xwines_to_common(wines_raw, ratings_raw, cfg):
    """Convert raw X-Wines tables into the common ITEMS / RATINGS schema."""
    items = apply_column_map(wines_raw, cfg["adapter"]["xwines_items"])
    for col in cfg["list_columns"]:
        if col in items:
            items[col] = items[col].apply(parse_list)

    ratings = apply_column_map(ratings_raw, cfg["adapter"]["xwines_ratings"])
    ratings["rating"] = ratings["rating"].astype(float)   # already a 1-5 scale

    return items[ITEM_COLUMNS].reset_index(drop=True), \
        ratings[RATING_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wine Enthusiast (model-testing dataset)
# ---------------------------------------------------------------------------
def parse_winemag_title(title, winery):
    """Extract the wine name from a Wine Enthusiast title.

    Example: title "Rainstorm 2013 Pinot Gris (Willamette Valley)", winery
    "Rainstorm"  ->  "Pinot Gris"   (winery, vintage year and region removed)

    TODO(1): implement with regular expressions (re module). The current
    version only strips the winery prefix, so matching will be weak.
    """
    if not isinstance(title, str):
        return ""
    name = title
    if isinstance(winery, str) and name.startswith(winery):
        name = name[len(winery):]
    return name.strip()


def winemag_to_common(raw, cfg, scale_method="linear"):
    """Convert the raw Wine Enthusiast table into the common EXPERT schema."""
    df = apply_column_map(raw, cfg["adapter"]["winemag_items"])
    expert = pd.DataFrame({
        "expert_id": np.arange(len(df)),
        "name": [parse_winemag_title(t, w) for t, w in zip(df["title"], df["winery"])],
        "winery": df["winery"],
        "country": df["country"],
        # region_1 is often missing; fall back to the province
        "region": df["region"].fillna(df["province"]),
        # one variety per review -> wrap in a list to match X-Wines "grapes"
        "grapes": df["variety"].apply(lambda v: [v] if isinstance(v, str) else []),
        "price": df["price"],
        "points": df["points"],
        "description": df["description"],
        "taster": df["taster"],
    })
    expert["rating"] = scale_points(expert["points"], method=scale_method)
    return expert[EXPERT_COLUMNS].reset_index(drop=True)


def scale_points(points, method="linear", reference=None):
    """Convert expert points (80-100, truncated below 80) to the 1-5 scale.

    Args:
        points: Series of Wine Enthusiast points.
        method: "linear"   - min-max map 80..100 -> 1..5 (simple baseline,
                             used as the "before" side of the ablation study)
                "quantile" - TODO(1): map each point to the X-Wines rating
                             with the same percentile (handles the truncated
                             and skewed distribution)
        reference: X-Wines ratings Series, needed for "quantile".
    """
    points = points.astype(float)
    if method == "linear":
        return 1 + (points - 80) / (100 - 80) * 4
    if method == "quantile":
        # TODO(1): hint - percentile rank of each point (points.rank(pct=True)),
        # then reference.quantile(pct). Round to the nearest 0.5 if you like.
        raise NotImplementedError("scale_points(method='quantile') - TODO(1)")
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Linking the two datasets
# ---------------------------------------------------------------------------
def normalize_text(text):
    """Normalize a name for matching (lower-case, trimmed).

    TODO(1): also remove accents (e.g. "Château" -> "chateau"), punctuation
    and common words such as "winery", "estate", "bodegas", "domaine".
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def match_items(items, expert, method="exact", threshold=0.85):
    """Find the same wine in X-Wines (items) and Wine Enthusiast (expert).

    Args:
        method: "exact" - identical normalized (winery, name)   [baseline]
                "fuzzy" - TODO(1): string similarity, e.g. difflib.SequenceMatcher
                          on winery first, then on name; keep pairs >= threshold
    Returns:
        DataFrame with columns item_id, expert_id, score (1.0 for exact).
        One X-Wines wine can match several reviews (different vintages).
    """
    left = items.assign(_w=items["winery"].map(normalize_text),
                        _n=items["name"].map(normalize_text))
    right = expert.assign(_w=expert["winery"].map(normalize_text),
                          _n=expert["name"].map(normalize_text))
    if method == "exact":
        merged = left.merge(right, on=["_w", "_n"], how="inner")
        merged = merged[merged["_w"] != ""]
        return merged[["item_id", "expert_id"]].assign(score=1.0).reset_index(drop=True)
    if method == "fuzzy":
        raise NotImplementedError("match_items(method='fuzzy') - TODO(1)")
    raise ValueError(f"unknown method: {method}")
