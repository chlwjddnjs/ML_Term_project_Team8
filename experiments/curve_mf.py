"""Learning curve of the final MF setting: train' vs validation RMSE per epoch.

run.py only has train and test, so its learning_curve_MF.png shows the train
RMSE alone (watching the test set while training would make the test score
dishonest). This script holds out 10% of each user's TRAIN ratings as a
validation set (same split_by_user and seed as run.py and tune_mf.py), trains
MF with the setting in config.yaml for up to --epochs epochs and records both
RMSEs after every epoch.

How to read it: the train RMSE keeps going down. If the validation RMSE stops
going down (or goes up) at the same time, more epochs only memorise the train
ratings, i.e. the model starts to over-fit. The epoch with the lowest
validation RMSE is printed.

Outputs (outputs/):
  learning_curve_MF_valid.png   train_rmse and valid_rmse per epoch
  learning_curve_MF_valid.csv   the same numbers

Run from the project root (about 1 minute on X-Wines Slim):
    python experiments/curve_mf.py
    python experiments/curve_mf.py --epochs 60
    python experiments/curve_mf.py --data sample      # quick smoke test
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Python only looks in experiments/ for imports when this file is run directly,
# so add the project root to find the recsys package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import load_config                    # noqa: E402
from recsys.data import adapter, cleaner, loader          # noqa: E402
from recsys.data.split import split_by_user               # noqa: E402
from recsys.models.mf import MatrixFactorization          # noqa: E402
from recsys.utils.plots import plot_learning_curve        # noqa: E402

VALID_RATIO = 0.1          # share of each user's TRAIN ratings held out (same as tune_mf.py)


def main():
    parser = argparse.ArgumentParser(description="MF learning curve with a validation set")
    parser.add_argument("--data", choices=["slim", "sample"], default=None,
                        help="data profile (overrides config.yaml)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="epochs to train (config.yaml uses n_epochs 20; more shows what happens after)")
    args = parser.parse_args()

    cfg = load_config("config.yaml", args.data)
    out = cfg["paths"]["output_dir"]

    # 1. Data: exactly the same steps and split as run.py
    wines_raw, ratings_raw = loader.load_xwines(cfg["paths"]["xwines_wines"],
                                                cfg["paths"]["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(wines_raw, ratings_raw, cfg)
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    train, _test = split_by_user(ratings, cfg["split"]["test_ratio"],
                                 cfg["min_user_ratings"], cfg["seed"])

    # 2. Validation ratings come from TRAIN only; the test set is never used here
    train_sub, valid = split_by_user(train, VALID_RATIO, cfg["min_user_ratings"], cfg["seed"])
    print(f"data={cfg['data']}  train'={len(train_sub):,}  valid={len(valid):,}")

    # 3. The final setting from config.yaml, trained for --epochs epochs
    params = {**cfg["model_params"]["mf"], "n_epochs": args.epochs}
    print(f"training MF {params} ...")
    model = MatrixFactorization(**params, seed=cfg["seed"]).set_validation(valid).fit(train_sub, items)

    # 4. Save the numbers and the chart (team plot function: one line per recorded value)
    history = pd.DataFrame(model.history)
    history.round(4).to_csv(out / "learning_curve_MF_valid.csv", index=False)
    mf = cfg["model_params"]["mf"]
    title = (f"MF train' vs valid RMSE (n_factors {mf['n_factors']}, reg {mf['reg']}, "
             f"bias_damping {mf.get('bias_damping', 0)})")
    plot_learning_curve(model.history, out / "learning_curve_MF_valid.png", title)

    best = history.loc[history["valid_rmse"].idxmin()]
    print(f"\nlowest valid RMSE {best['valid_rmse']:.4f} at epoch {int(best['epoch'])} "
          f"(config.yaml uses n_epochs {mf['n_epochs']})")
    print(history.set_index("epoch").round(4).iloc[[0, 4, 9, 19, -1]].to_string())
    print(f"\nSaved {out}/learning_curve_MF_valid.png and .csv")


if __name__ == "__main__":
    main()
