"""Failure case analysis: which ratings do the rating models get most wrong, and why?

Owner: member 4 (신우엽).  Material for the PPT "Evaluation & analysis" (failures).

What this script does (same data, cleaning and split as run.py):
  1. trains the rating models (ItemKNN, MF) with the settings in config.yaml
  2. for each model, takes the n held-out ratings with the largest |pred - true|
     (analysis.worst_predictions) and adds what the model knew about the user
     and the wine: train rating counts, the user's / wine's mean train rating
  3. checks which failure patterns are over-represented in those worst cases
     (analysis.failure_patterns): cold users, rarely rated wines, extreme ratings
  4. turns the worst-case list into general statements:
       - MAE by number of train ratings of the user / of the wine
       - mean error by true rating (does the model pull ratings toward the mean?)
  5. saves everything to outputs/failure_*.csv and outputs/failure_*.png

Content and Hybrid are ranking-only models (no rating prediction), so they are
not part of this rating-error analysis.

Run from the project root (the folder with run.py):
    python experiments/failure_analysis.py                 # X-Wines Slim
    python experiments/failure_analysis.py --data sample   # quick smoke test
    python experiments/failure_analysis.py --n 30          # more worst cases
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # save figures without opening a window
import matplotlib.pyplot as plt        # noqa: E402
import pandas as pd                    # noqa: E402

# Python only looks in experiments/ for imports when this file is run directly,
# so add the project root to find the recsys package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import load_config                  # noqa: E402
from recsys.data import adapter, cleaner, loader        # noqa: E402
from recsys.data.split import split_by_user             # noqa: E402
from recsys.eval import analysis                        # noqa: E402
from recsys.models.baseline import ItemKNNRecommender   # noqa: E402
from recsys.models.mf import MatrixFactorization        # noqa: E402

# Columns shown in the console / saved for each worst case, in this order
WORST_COLUMNS = ["user_id", "item_id", "name", "type", "country", "rating", "pred", "error",
                 "user_n_train", "user_mean_train", "item_n_train", "item_mean_train", "grapes"]


def add_context(worst, train):
    """Add the user's and the wine's mean train rating to the worst cases.
    A big gap between the true rating and these means explains most failures
    (e.g. a harsh user rating a wine everybody else loves)."""
    worst = worst.copy()
    worst["user_mean_train"] = worst["user_id"].map(train.groupby("user_id")["rating"].mean())
    worst["item_mean_train"] = worst["item_id"].map(train.groupby("item_id")["rating"].mean())
    for col in ("pred", "error", "user_mean_train", "item_mean_train"):
        worst[col] = worst[col].round(2)
    return worst[[c for c in WORST_COLUMNS if c in worst.columns]]


def plot_error_by_rating(by_rating, path):
    """Mean signed error per true rating, one line per model.
    Points above 0: predicted too high; below 0: predicted too low."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in by_rating.groupby("model"):
        ax.plot(g["rating"], g["mean_error"], marker="o", label=name)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("true rating")
    ax.set_ylabel("mean error (pred - true)")
    ax.set_title("Prediction bias by true rating")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mae_by_group(by_group, by, path):
    """MAE per group of users or wines (by number of train ratings), one bar group per model."""
    table = by_group.pivot(index="group", columns="model", values="MAE")
    ax = table.plot(kind="bar", figsize=(6, 4), rot=0)
    label = "user" if by == "user_n_train" else "wine"
    ax.set_xlabel(f"train ratings of the {label}")
    ax.set_ylabel("MAE")
    ax.set_title(f"Error by number of train ratings of the {label}")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def main():
    parser = argparse.ArgumentParser(description="Failure case analysis of the rating models")
    parser.add_argument("--data", choices=["slim", "sample"], default=None,
                        help="data profile (overrides config.yaml)")
    parser.add_argument("--n", type=int, default=20, help="number of worst predictions per model")
    args = parser.parse_args()

    # Wine names can contain accents; never crash on a console that cannot show them.
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 28)

    cfg = load_config("config.yaml", args.data)
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

    # 2. Train the rating models with the config.yaml settings
    print("training ItemKNN and MF ...")
    models = [
        ItemKNNRecommender(**(cfg["model_params"].get("item_knn") or {})).fit(train, items),
        MatrixFactorization(**cfg["model_params"]["mf"], seed=cfg["seed"]).fit(train, items),
    ]

    patterns, by_rating, by_user, by_item = [], [], [], []
    for model in models:
        # 3. The worst predictions, with context
        worst = add_context(analysis.worst_predictions(model, test, n=args.n), train)
        worst.to_csv(out / f"failure_worst_{model.name}.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== {model.name}: {args.n} worst predictions ===")
        print(worst.drop(columns=["grapes"], errors="ignore").to_string(index=False))

        # 4. Patterns and general statements
        patterns.append(analysis.failure_patterns(model, test, n=args.n))
        by_rating.append(analysis.error_by_true_rating(model, test))
        by_user.append(analysis.error_by_group(model, test, by="user_n_train"))
        by_item.append(analysis.error_by_group(model, test, by="item_n_train"))

    patterns = pd.concat(patterns, ignore_index=True)
    by_rating = pd.concat(by_rating, ignore_index=True)
    by_user = pd.concat(by_user, ignore_index=True)
    by_item = pd.concat(by_item, ignore_index=True)

    print("\n=== Failure patterns (lift > 1: more common among the worst cases) ===")
    print(patterns.round(3).to_string(index=False))
    print("\n=== Error by true rating (mean_error > 0: predicted too high) ===")
    print(by_rating.round(3).to_string(index=False))
    print("\n=== Error by train ratings of the USER ===")
    print(by_user.round(3).to_string(index=False))
    print("\n=== Error by train ratings of the WINE ===")
    print(by_item.round(3).to_string(index=False))

    # 5. Save
    patterns.to_csv(out / "failure_patterns.csv", index=False)
    by_rating.to_csv(out / "failure_error_by_rating.csv", index=False)
    by_user.to_csv(out / "failure_error_by_user_group.csv", index=False)
    by_item.to_csv(out / "failure_error_by_item_group.csv", index=False)
    plot_error_by_rating(by_rating, out / "failure_error_by_rating.png")
    plot_mae_by_group(by_user, "user_n_train", out / "failure_mae_by_user_group.png")
    plot_mae_by_group(by_item, "item_n_train", out / "failure_mae_by_item_group.png")
    print(f"\nSaved tables and plots to {out}/failure_*")


if __name__ == "__main__":
    main()
