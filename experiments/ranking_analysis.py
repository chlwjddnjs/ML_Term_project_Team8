"""Ranking failure analysis: why does non-personalized Popularity beat the
personalized models (MF, Content, Hybrid, ItemKNN) on P@K / R@K / NDCG@K?

Owner: member 4 (신우엽).  Material for the PPT "Evaluation & analysis".

What this script does (same data, split, models and evaluation users as run.py):
  1. trains the five models with the settings in config.yaml
  2. checks how concentrated the held-out "relevant" ratings (>= threshold) are
     on the most-rated wines: if a few wines collect most test hits, simply
     recommending them (Popularity) scores well offline
  3. for each model, describes its top-k lists: how many different wines it
     recommends, how often those wines were rated in train (popularity of the
     recommendations), and how many hits it gets
  4. re-evaluates every model on the "long tail": the relevant test items
     outside the `--head` most-rated wines.  Popularity cannot find these;
     a personalized model should
  5. compares Popularity and Hybrid user by user (who wins for how many users)
  6. adds a random recommender as a floor for the metrics
  7. saves outputs/ranking_*.csv and outputs/ranking_*.png

Run from the project root (the folder with run.py):
    python experiments/ranking_analysis.py                 # X-Wines Slim
    python experiments/ranking_analysis.py --data sample   # quick smoke test
    python experiments/ranking_analysis.py --head 50       # size of the "popular head"
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # save figures without opening a window
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

# Python only looks in experiments/ for imports when this file is run directly,
# so add the project root to find the recsys package and run.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import load_config                  # noqa: E402
from recsys.data import adapter, cleaner, loader        # noqa: E402
from recsys.data.split import split_by_user             # noqa: E402
from recsys.eval import metrics                         # noqa: E402
from run import build_model                             # noqa: E402


def eval_users(test, threshold, max_users, seed):
    """Users and relevant sets exactly as evaluator.evaluate_ranking picks them."""
    relevant = test[test["rating"] >= threshold].groupby("user_id")["item_id"].apply(set)
    users = relevant.index.to_numpy()
    if len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
    return users, relevant


def ranking_scores(recs, relevant_sets, k):
    """Mean P@k / R@k / NDCG@k over users that still have a relevant item."""
    rows = [(metrics.precision_at_k(r, rel, k), metrics.recall_at_k(r, rel, k),
             metrics.ndcg_at_k(r, rel, k)) for r, rel in zip(recs, relevant_sets) if rel]
    p, r, n = np.mean(rows, axis=0) if rows else (np.nan,) * 3
    return {f"P@{k}": p, f"R@{k}": r, f"NDCG@{k}": n, "n_users": len(rows)}


def main():
    parser = argparse.ArgumentParser(description="Why Popularity wins the ranking metrics")
    parser.add_argument("--data", choices=["slim", "sample"], default=None,
                        help="data profile (overrides config.yaml)")
    parser.add_argument("--head", type=int, default=50,
                        help="the HEAD most-rated wines count as 'popular' (long-tail = the rest)")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    cfg = load_config("config.yaml", args.data)
    out = cfg["paths"]["output_dir"]
    k = cfg["evaluation"]["k"]
    threshold = cfg["evaluation"]["relevant_threshold"]

    # 1. Data and models: exactly the same steps as run.py
    wines_raw, ratings_raw = loader.load_xwines(cfg["paths"]["xwines_wines"],
                                                cfg["paths"]["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(wines_raw, ratings_raw, cfg)
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    train, test = split_by_user(ratings, cfg["split"]["test_ratio"],
                                cfg["min_user_ratings"], cfg["seed"])
    print(f"data={cfg['data']}  train={len(train):,}  test={len(test):,}  wines={len(items):,}")

    models = []
    for name in cfg["models"]:
        print(f"training {name} ...")
        models.append(build_model(name, cfg).fit(train, items))

    users, relevant = eval_users(test, threshold, cfg["evaluation"]["max_eval_users"], cfg["seed"])
    rel_sets = [relevant[u] for u in users]
    print(f"evaluation users={len(users):,}  (top-{k}, relevant = test rating >= {threshold})")

    # 2. How concentrated are the relevant test items on popular wines?
    n_train = train.groupby("item_id").size().reindex(items["item_id"], fill_value=0)
    ranked = n_train.sort_values(ascending=False)
    head = set(ranked.index[:args.head])
    rel_items = test[test["rating"] >= threshold]["item_id"]
    concentration = []
    for top in (10, args.head, 100):
        top_set = set(ranked.index[:top])
        concentration.append({
            "top_n_wines": top,
            "share_of_catalogue": top / len(items),
            "share_of_train_ratings": ranked.iloc[:top].sum() / ranked.sum(),
            "share_of_relevant_test_ratings": rel_items.isin(top_set).mean(),
        })
    concentration = pd.DataFrame(concentration)
    print("\n=== Concentration of relevant test ratings on the most-rated wines ===")
    print(concentration.round(3).to_string(index=False))

    # 3 + 4. Each model's top-k lists, overall and on the long tail
    rng = np.random.default_rng(cfg["seed"])
    all_ids = np.asarray(items["item_id"])
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    def random_rec(u):
        cand = all_ids[~np.isin(all_ids, list(seen.get(u, ())))]
        return rng.choice(cand, size=min(k, len(cand)), replace=False).tolist()

    tail_sets = [rel - head for rel in rel_sets]
    summary, per_user, top_wines = [], {}, []
    recommenders = [(m.name, m.recommend) for m in models] + [("Random", random_rec)]
    for name, recommend in recommenders:
        recs = [recommend(u, k=k) if name != "Random" else recommend(u) for u in users]
        flat = pd.Series([i for r in recs for i in r])
        hits = [i for r, rel in zip(recs, rel_sets) for i in r if i in rel]
        overall = ranking_scores(recs, rel_sets, k)
        tail = ranking_scores(recs, tail_sets, k)
        summary.append({
            "model": name,
            f"NDCG@{k}": overall[f"NDCG@{k}"], f"P@{k}": overall[f"P@{k}"], f"R@{k}": overall[f"R@{k}"],
            f"tail NDCG@{k}": tail[f"NDCG@{k}"], f"tail R@{k}": tail[f"R@{k}"],
            "distinct wines recommended": flat.nunique(),
            "share of recs in popular head": flat.isin(head).mean(),
            "median train ratings of recs": flat.map(n_train).median(),
            "hits": len(hits),
            "share of hits in popular head": pd.Series(hits).isin(head).mean() if hits else np.nan,
        })
        per_user[name] = [metrics.ndcg_at_k(r, rel, k) for r, rel in zip(recs, rel_sets)]
        counts = flat.value_counts().head(5)
        for item_id, c in counts.items():
            top_wines.append({"model": name, "item_id": item_id, "times_recommended": c,
                              "users_share": c / len(users), "train_ratings": n_train.get(item_id, 0)})

    summary = pd.DataFrame(summary)
    top_wines = pd.DataFrame(top_wines).merge(items[["item_id", "name"]], on="item_id", how="left")
    print(f"\n=== Top-{k} lists per model (head = {args.head} most-rated wines) ===")
    print(summary.round(4).to_string(index=False))
    print(f"\n=== Most recommended wines per model (times = users out of {len(users):,}) ===")
    print(top_wines.to_string(index=False))

    # 5. Popularity vs Hybrid, user by user
    pop, hyb = np.array(per_user["Popularity"]), np.array(per_user["Hybrid"])
    duel = pd.DataFrame([{
        "Hybrid better": int((hyb > pop).sum()), "Popularity better": int((pop > hyb).sum()),
        "tie (both 0)": int(((hyb == pop) & (pop == 0)).sum()),
        "tie (same > 0)": int(((hyb == pop) & (pop > 0)).sum())}])
    print("\n=== NDCG per user: Popularity vs Hybrid ===")
    print(duel.to_string(index=False))

    # 6. Save
    concentration.to_csv(out / "ranking_concentration.csv", index=False)
    summary.to_csv(out / "ranking_summary.csv", index=False)
    top_wines.to_csv(out / "ranking_top_wines.csv", index=False, encoding="utf-8-sig")
    duel.to_csv(out / "ranking_popularity_vs_hybrid.csv", index=False)

    # Figure 1: overall vs long-tail NDCG
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary[f"NDCG@{k}"], 0.4, label="all relevant items")
    ax.bar(x + 0.2, summary[f"tail NDCG@{k}"], 0.4, label=f"long tail (outside top {args.head})")
    ax.set_xticks(x, summary["model"])
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_title("Ranking quality: all items vs long tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ranking_ndcg_overall_vs_tail.png", dpi=150)
    plt.close(fig)

    # Figure 2: how popular are the recommended wines?
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["model"], summary["share of recs in popular head"] * 100, color="#8c2f4a")
    ax.axhline(args.head / len(items) * 100, color="gray", ls="--",
               label=f"share of catalogue ({args.head / len(items):.0%})")
    ax.set_ylabel(f"% of recommendations in top {args.head} wines")
    ax.set_title("How popular are the recommended wines?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ranking_popularity_of_recs.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved tables and plots to {out}/ranking_*")


if __name__ == "__main__":
    main()
