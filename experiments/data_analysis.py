"""Data exploration + ablation study of the DatasetAdapter.  Owner: member 1.

Answers two questions for the PPT ("Data exploration", "Preprocessing"):

  A. EDA - what is wrong with the raw data?
     ratings per wine (long tail), ABV outliers (grappas), expert points and
     price distributions.
  B. Ablation - does each part of the alignment module help?
     B1. matching : original code vs exact (normalize + title parsing)
                    vs fuzzy at several thresholds
     B2. scaling  : linear vs quantile conversion of expert points,
                    compared with the X-Wines mean rating of the same wine

Outputs (outputs/):
  eda_summary.csv              key numbers of the raw data
  eda_ratings_per_wine.png     how many ratings each wine has
  eda_abv.png                  ABV of every wine, outliers marked
  eda_expert_points_price.png  expert points and log-price distributions
  ablation_matching.csv        matched wines / reviews per method
  ablation_matching_examples.csv  pairs found only by fuzzy matching
  ablation_scaling.csv         linear vs quantile (mean, MAE, RMSE vs X-Wines)
  ablation_scaling.png         rating distributions after each conversion

Run from the project root (about 2 minutes on X-Wines Slim):
    python experiments/data_analysis.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # draw straight to PNG files (no window needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

# Python only looks in experiments/ for imports when this file is run directly,
# so add the project root to find the recsys package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import load_config           # noqa: E402
from recsys.data import adapter, cleaner, loader  # noqa: E402

FUZZY_THRESHOLDS = [0.80, 0.85, 0.90]   # 0.85 is the default used by run.py
WINE_COLOR = "#7A1F3D"                  # same color as recsys/utils/plots.py


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# The ORIGINAL helpers (before member 1's work), kept only for the ablation
# ---------------------------------------------------------------------------
def original_normalize(text):
    """Old normalize_text: lower-case and trim only."""
    return " ".join(text.lower().split()) if isinstance(text, str) else ""


def original_match(items, expert_raw_names, expert):
    """Old exact matching: no accent/stopword removal, title only lost the winery."""
    left = items.assign(_w=items["winery"].map(original_normalize),
                        _n=items["name"].map(original_normalize))
    right = expert.assign(_w=expert["winery"].map(original_normalize),
                          _n=expert_raw_names.map(original_normalize))
    merged = left.merge(right, on=["_w", "_n"])
    return merged[merged["_w"] != ""][["item_id", "expert_id"]]


def old_title(title, winery):
    """Old parse_winemag_title: only removed the winery prefix."""
    if not isinstance(title, str):
        return ""
    if isinstance(winery, str) and title.startswith(winery):
        title = title[len(winery):]
    return title.strip()


# ---------------------------------------------------------------------------
# A. EDA
# ---------------------------------------------------------------------------
def eda(items, ratings, expert_raw, out):
    section("A. Data exploration")
    rows = []

    # A1. ratings per wine: a few wines have many ratings, most have few
    per_wine = ratings.groupby("item_id").size()
    rows += [("X-Wines", "wines", len(items)),
             ("X-Wines", "users", ratings["user_id"].nunique()),
             ("X-Wines", "ratings", len(ratings)),
             ("X-Wines", "ratings per wine (median)", per_wine.median()),
             ("X-Wines", "wines with < 20 ratings", int((per_wine < 20).sum())),
             ("X-Wines", "mean rating", round(ratings["rating"].mean(), 3))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_wine, bins=50, color=WINE_COLOR)
    ax.set_yscale("log")
    ax.set_xlabel("number of ratings of a wine")
    ax.set_ylabel("number of wines (log scale)")
    ax.set_title("X-Wines: ratings per wine (long tail)")
    save(fig, out / "eda_ratings_per_wine.png")

    # A2. ABV: values above 25 % are spirits (grappa / marc), not wine
    abv = pd.to_numeric(items["abv"], errors="coerce")
    bad = (abv <= 0) | (abv > cleaner.MAX_WINE_ABV)
    rows.append(("X-Wines", f"ABV > {cleaner.MAX_WINE_ABV}% (spirits)", int(bad.sum())))
    print("  ABV outliers:", ", ".join(f"{n} ({a}%)" for n, a in
                                       zip(items.loc[bad, "name"], abv[bad])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(range(len(abv)), abv, s=8, color="grey", label="wine")
    ax.scatter(np.where(bad)[0], abv[bad], s=30, color=WINE_COLOR,
               label="outlier (grappa / marc)")
    ax.axhline(cleaner.MAX_WINE_ABV, ls="--", color="black", lw=1)
    ax.set_xlabel("wine index")
    ax.set_ylabel("ABV (%)")
    ax.set_title(f"X-Wines: ABV (dashed line = {cleaner.MAX_WINE_ABV}%)")
    ax.legend()
    save(fig, out / "eda_abv.png")

    # A3. expert data: missing values, duplicates, points and price
    price = expert_raw["price"].dropna()
    low, high = (np.exp(b) for b in cleaner.iqr_bounds(np.log(price)))
    rows += [("Wine Enthusiast", "reviews", len(expert_raw)),
             ("Wine Enthusiast", "duplicate reviews",
              int(expert_raw.duplicated(subset=["winery", "title", "taster_name",
                                                "points", "description"]).sum())),
             ("Wine Enthusiast", "missing price", int(expert_raw["price"].isna().sum())),
             ("Wine Enthusiast", "missing country", int(expert_raw["country"].isna().sum())),
             ("Wine Enthusiast", "missing taster", int(expert_raw["taster_name"].isna().sum())),
             ("Wine Enthusiast", "points min / max",
              f"{expert_raw['points'].min()} / {expert_raw['points'].max()}"),
             ("Wine Enthusiast", "price median / max",
              f"{price.median():.0f} / {price.max():.0f}"),
             ("Wine Enthusiast", f"price outliers (outside ${low:.0f}-${high:.0f}, "
                                 "before removing duplicates)",
              int(((price < low) | (price > high)).sum()))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    expert_raw["points"].value_counts().sort_index().plot(kind="bar", ax=axes[0],
                                                          color=WINE_COLOR)
    axes[0].set_title("Expert points (only 80+ are reviewed)")
    axes[0].set_xlabel("points")
    axes[1].hist(np.log10(price), bins=50, color=WINE_COLOR)
    for b in (low, high):
        axes[1].axvline(np.log10(b), ls="--", color="black", lw=1)
    axes[1].set_title("Expert price (log10 $), dashed = IQR limits")
    axes[1].set_xlabel("log10(price)")
    save(fig, out / "eda_expert_points_price.png")

    summary = pd.DataFrame(rows, columns=["dataset", "item", "value"])
    summary.to_csv(out / "eda_summary.csv", index=False)
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# B1. Matching ablation
# ---------------------------------------------------------------------------
def matching_ablation(items, expert, expert_raw_names, out):
    section("B1. Ablation - matching X-Wines wines to expert reviews")
    rows = []

    def add(method, m):
        rows.append({"method": method, "matched_wines": m["item_id"].nunique(),
                     "matched_reviews": len(m)})
        print(f"  {method:<28} wines={m['item_id'].nunique():>4}  reviews={len(m):>5}")

    add("original code", original_match(items, expert_raw_names, expert))
    exact = adapter.match_items(items, expert, method="exact")
    add("exact (normalize + title)", exact)
    fuzzy_085 = None
    for t in FUZZY_THRESHOLDS:
        m = adapter.match_items(items, expert, method="fuzzy", threshold=t)
        add(f"fuzzy (threshold {t:.2f})", m)
        if t == 0.85:
            fuzzy_085 = m
    pd.DataFrame(rows).to_csv(out / "ablation_matching.csv", index=False)

    # pairs that only fuzzy matching finds -> good examples for the PPT
    extra = fuzzy_085.merge(exact[["item_id", "expert_id"]], on=["item_id", "expert_id"], how="left",
                            indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    extra = (extra.merge(items[["item_id", "winery", "name"]], on="item_id")
                  .merge(expert[["expert_id", "winery", "name"]], on="expert_id",
                         suffixes=("_xwines", "_expert"))
                  .drop_duplicates(subset=["item_id", "name_expert"])
                  .sort_values("score"))
    extra.to_csv(out / "ablation_matching_examples.csv", index=False)
    print(f"  {len(extra)} pairs found only by fuzzy -> ablation_matching_examples.csv")
    return fuzzy_085


# ---------------------------------------------------------------------------
# B2. Scaling ablation
# ---------------------------------------------------------------------------
def scaling_ablation(ratings, expert, matches, out):
    section("B2. Ablation - converting expert points (80-100) to 1-5")
    converted = {
        "linear": adapter.scale_points(expert["points"], "linear"),
        "quantile": adapter.scale_points(expert["points"], "quantile",
                                         reference=ratings["rating"]),
    }
    # the same wine rated by users (X-Wines mean) and by experts (converted)
    user_mean = ratings.groupby("item_id")["rating"].mean().rename("user")
    rows = []
    for method, values in converted.items():
        per_wine = (matches.merge(expert[["expert_id"]].assign(expert=values),
                                  on="expert_id")
                           .groupby("item_id")["expert"].mean())
        both = pd.concat([per_wine, user_mean], axis=1, join="inner")
        diff = both["expert"] - both["user"]
        rows.append({"method": method,
                     "mean": round(values.mean(), 3),
                     "std": round(values.std(), 3),
                     "x_wines_mean": round(ratings["rating"].mean(), 3),
                     "MAE_vs_users": round(diff.abs().mean(), 3),
                     "RMSE_vs_users": round(np.sqrt((diff ** 2).mean()), 3),
                     # Spearman = Pearson correlation of the ranks
                     "spearman_vs_users": round(both["expert"].rank()
                                                .corr(both["user"].rank()), 3),
                     "n_wines": len(both)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation_scaling.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0.75, 5.5, 0.25)
    ax.hist(ratings["rating"], bins=bins, density=True, alpha=0.5, label="X-Wines users")
    ax.hist(converted["linear"], bins=bins, density=True, alpha=0.5, label="experts - linear")
    ax.hist(converted["quantile"], bins=bins, density=True, alpha=0.5,
            label="experts - quantile")
    ax.set_xlabel("rating (1-5)")
    ax.set_ylabel("density")
    ax.set_title("Expert points converted to 1-5: linear vs quantile")
    ax.legend()
    save(fig, out / "ablation_scaling.png")


def main():
    cfg = load_config("config.yaml")
    out = cfg["paths"]["output_dir"]

    wines_raw, ratings_raw = loader.load_xwines(cfg["paths"]["xwines_wines"],
                                                cfg["paths"]["xwines_ratings"])
    expert_raw = loader.load_winemag(cfg["paths"]["winemag"])
    if expert_raw is None:
        sys.exit("winemag-data-130k-v2.csv not found in data/raw/")
    items, ratings = adapter.xwines_to_common(wines_raw, ratings_raw, cfg)

    eda(items, ratings, expert_raw, out)

    # same steps as run.py: convert -> clean -> match
    expert = adapter.winemag_to_common(expert_raw, cfg)
    raw_titles = pd.Series([old_title(t, w) for t, w in
                            zip(expert_raw["title"], expert_raw["winery"])])
    expert = expert.assign(_old_name=raw_titles.to_numpy())
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    expert = cleaner.clean_expert(expert)

    matches = matching_ablation(items, expert, expert["_old_name"], out)
    scaling_ablation(ratings, expert, matches, out)
    print(f"\nSaved tables and plots to {out}/")


if __name__ == "__main__":
    main()