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
Implemented by member 1:
    normalize_text, parse_winemag_title, match_items(method="fuzzy")
TODO for member 1 (marked "TODO(1)"):
    scale_points(method="quantile")
"""
import ast
import difflib
import re
import unicodedata

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

    Steps:
        1. remove the winery name at the start of the title
        2. remove the region in brackets at the end, also when the bracket
           contains another bracket: "(Walla Walla Valley (WA))"
        3. remove the vintage year (1900-2099) and "NV" (non-vintage)
        4. collapse the spaces left behind
    """
    if not isinstance(title, str):
        return ""
    name = title
    # 1. winery prefix: "Rainstorm 2013 Pinot Gris (...)" -> " 2013 Pinot Gris (...)"
    if isinstance(winery, str) and name.startswith(winery):
        name = name[len(winery):]
    # 2. trailing "(...)" at the end of the title ($ = end of string).
    #    Inside the bracket: any non-bracket character, or one inner "(...)".
    name = re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)\s*$", "", name)
    # 3. a 4-digit year starting with 19 or 20, or the word "NV".
    #    \b = word boundary, so "2013" is removed but "12013" is not.
    name = re.sub(r"\b(?:19|20)\d{2}\b|\bNV\b", " ", name)
    # 4. "  Pinot   Gris " -> "Pinot Gris"
    return " ".join(name.split())


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
# Common words that one dataset attaches to a name and the other does not
# (e.g. "Anne Amie Vineyards" vs "Anne Amie"). They are removed before matching.
STOPWORDS = {"winery", "wines", "wine", "vineyards", "vineyard", "estate", "estates",
             "cellars", "cellar", "bodegas", "bodega", "domaine", "domaines", "chateau",
             "chateaux", "cantina", "cantine", "tenuta", "weingut", "quinta", "casa",
             "the", "de", "du", "des", "di", "del", "la", "le", "les", "and"}


def normalize_text(text):
    """Normalize a name so that the two datasets can be compared.

    Example: "Domaine Nathalie et Gilles Fèvre" -> "nathalie et gilles fevre"

    Steps:
        1. remove accents      "Fèvre" -> "Fevre"
        2. lower-case          "Fevre" -> "fevre"
        3. punctuation -> space "Grand'Arte" -> "grand arte"
        4. drop STOPWORDS      "domaine nathalie ..." -> "nathalie ..."
    Missing values (NaN) become an empty string.
    """
    if not isinstance(text, str):
        return ""
    # 1. NFKD splits "è" into "e" + a separate accent mark;
    #    encoding to ASCII with "ignore" then drops the accent mark
    #    (non-Latin letters such as Cyrillic are dropped as well).
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # 2. lower-case
    text = text.lower()
    # 3. every run of characters that is not a letter or digit -> one space
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # 4. split into words, drop the common words, join back with single spaces
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def _similarity(a, b):
    """String similarity between 0 (nothing in common) and 1 (identical).

    difflib.SequenceMatcher.ratio() = 2 * (matching characters) / (len(a) + len(b))
    e.g. "djf vinhos" vs "dfj vinhos" -> 0.9
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_items(items, expert, method="exact", threshold=0.85):
    """Find the same wine in X-Wines (items) and Wine Enthusiast (expert).

    Args:
        method: "exact" - identical normalized (winery, name)   [baseline]
                "fuzzy" - string similarity (difflib): winery first, then name;
                          keep pairs whose winery AND name are >= threshold
        threshold: minimum similarity (0-1) for the fuzzy method.
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
        # Rows without a winery name cannot be matched.
        left = left[left["_w"] != ""]
        right = right[right["_w"] != ""]

        # Group the ~130K reviews by winery once, so that for each X-Wines wine
        # we only compare names inside a few similar wineries instead of
        # comparing against every review (much faster).
        reviews_by_winery = {w: g for w, g in right.groupby("_w")}
        expert_wineries = list(reviews_by_winery)

        rows = []
        for item_id, item_winery, item_name in zip(left["item_id"], left["_w"], left["_n"]):
            # Step 1: up to 3 expert wineries whose name is similar enough
            # (get_close_matches returns them sorted, most similar first).
            close_wineries = difflib.get_close_matches(
                item_winery, expert_wineries, n=3, cutoff=threshold)
            for winery in close_wineries:
                winery_score = _similarity(item_winery, winery)
                # Step 2: inside that winery, compare the wine names.
                reviews = reviews_by_winery[winery]
                for expert_id, expert_name in zip(reviews["expert_id"], reviews["_n"]):
                    name_score = _similarity(item_name, expert_name)
                    if name_score >= threshold:
                        # final score = average of the two similarities
                        score = round((winery_score + name_score) / 2, 4)
                        rows.append((item_id, expert_id, score))
        return pd.DataFrame(rows, columns=["item_id", "expert_id", "score"])
    raise ValueError(f"unknown method: {method}")