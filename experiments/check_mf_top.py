"""Cause check: are the wines MF recommends rated only a few times?

Hypothesis (from the first Slim run): MF loses to Popularity on P@10 because its
top-k is dominated by wines with a high item bias b_i but very few train ratings.
Such wines almost never appear in a user's test set, so recommending them
rarely counts as a hit.

What this script does (same data, split and evaluation users as run.py):
  1. trains Popularity and MF with the settings in config.yaml
  2. recommends top-k wines to the same sampled users as the evaluator
  3. compares, for the recommended wines of each model:
       - how many different wines are recommended
       - how many train ratings those wines have (median)
       - how often those wines were rated >= 4 in test (median)
       - how many recommendations were hits
  4. prints the most-recommended wines of each model and the wines with the
     highest MF item bias b_i, with their train / test rating counts
  5. saves the tables to outputs/check_mf_top_*.csv

How to read it: if MF's recommended wines have far fewer train ratings than
Popularity's (and appear in test far less often), the hypothesis holds.

Run from the project root (the folder with run.py):
    python experiments/check_mf_top.py                 # X-Wines Slim
    python experiments/check_mf_top.py --data sample   # quick smoke test
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Python only looks in experiments/ for imports when this file is run directly,
# so add the project root to find the recsys package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import load_config                    # noqa: E402
from recsys.data import adapter, cleaner, loader          # noqa: E402
from recsys.data.split import split_by_user               # noqa: E402
from recsys.models.baseline import PopularityRecommender  # noqa: E402
from recsys.models.mf import MatrixFactorization          # noqa: E402

COLUMNS = ["name", "rec_count", "n_train", "mean_train", "b_i", "n_test", "n_test_liked"]


def main():
    parser = argparse.ArgumentParser(description="Train-rating counts of the wines MF recommends")
    parser.add_argument("--data", choices=["slim", "sample"], default=None,
                        help="data profile (overrides config.yaml)")
    parser.add_argument("--top", type=int, default=10, help="rows shown per table")
    args = parser.parse_args()

    # Wine names can contain accents; never crash on a console that cannot show them.
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    cfg = load_config("config.yaml", args.data)
    k = cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevant_threshold"]
    out = cfg["paths"]["output_dir"]

    # 1. Data: exactly the same steps as run.py
    wines_raw, ratings_raw = loader.load_xwines(cfg["paths"]["xwines_wines"],
                                                cfg["paths"]["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(wines_raw, ratings_raw, cfg)
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    train, test = split_by_user(ratings, cfg["split"]["test_ratio"],
                                cfg["min_user_ratings"], cfg["seed"])
    print(f"data={cfg['data']}  train={len(train):,}  test={len(test):,}")

    # 2. Per-wine statistics (one row per wine in the catalogue)
    ids = pd.Index(items["item_id"], name="item_id")
    stats = pd.DataFrame(index=ids)
    stats["name"] = items.set_index("item_id")["name"].astype(str).str.slice(0, 32)
    stats["n_train"] = train.groupby("item_id").size().reindex(ids, fill_value=0)
    stats["mean_train"] = train.groupby("item_id")["rating"].mean().reindex(ids).round(2)
    stats["n_test"] = test.groupby("item_id").size().reindex(ids, fill_value=0)
    stats["n_test_liked"] = (test[test["rating"] >= threshold]
                             .groupby("item_id").size().reindex(ids, fill_value=0))

    # 3. Train both models with the config.yaml settings
    print("training Popularity and MF ...")
    pop = PopularityRecommender().fit(train, items)
    mf = MatrixFactorization(**cfg["model_params"]["mf"], seed=cfg["seed"]).fit(train, items)
    stats["b_i"] = pd.Series(mf.b_i, index=mf.item_ids).round(3)

    # 4. The same evaluation users as evaluator.evaluate_ranking
    #    (users with at least one test rating >= threshold, sampled with the same seed)
    relevant = test[test["rating"] >= threshold].groupby("user_id")["item_id"].apply(set)
    users = relevant.index.to_numpy()
    max_users = cfg["evaluation"]["max_eval_users"]
    if len(users) > max_users:
        users = np.random.default_rng(cfg["seed"]).choice(users, size=max_users, replace=False)
    print(f"recommending top-{k} to {len(users):,} users ...")

    summary, tops = {}, {}
    for model in (pop, mf):
        counts, hits = Counter(), 0
        for u in users:
            rec = model.recommend(u, k=k)
            counts.update(rec)
            hits += len(set(rec) & relevant[u])
        rec_count = pd.Series(counts, name="rec_count")          # item_id -> times recommended
        rec = stats.loc[rec_count.index].assign(rec_count=rec_count)
        times = rec["rec_count"].to_numpy()                      # weight = times recommended
        summary[model.name] = {
            "distinct wines recommended": len(rec),
            "median train ratings of recommended wines": np.median(np.repeat(rec["n_train"].to_numpy(), times)),
            "median test ratings >= 4 of recommended wines": np.median(np.repeat(rec["n_test_liked"].to_numpy(), times)),
            "hits (recommended and liked in test)": hits,
        }
        tops[model.name] = rec.sort_values("rec_count", ascending=False)[COLUMNS]

    # 5. Report
    print("\n=== Summary ===")
    print(pd.DataFrame(summary).to_string())
    print(f"\n(catalogue: median train ratings per wine = {stats['n_train'].median():.0f})")
    for name, table in tops.items():
        print(f"\n=== {name}: most recommended wines ===")
        print(table.head(args.top).to_string())
    bias_top = stats.sort_values("b_i", ascending=False).assign(rec_count=np.nan)[COLUMNS]
    print("\n=== MF: wines with the highest item bias b_i ===")
    print(bias_top.drop(columns="rec_count").head(args.top).to_string())

    for name, table in tops.items():
        table.to_csv(out / f"check_mf_top_{name}.csv", encoding="utf-8-sig")
    bias_top.drop(columns="rec_count").to_csv(out / "check_mf_top_bias.csv", encoding="utf-8-sig")
    print(f"\nSaved tables to {out}/check_mf_top_*.csv")


if __name__ == "__main__":
    main()
