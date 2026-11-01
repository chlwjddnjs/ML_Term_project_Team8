"""Hyper-parameter tuning for Matrix Factorization on a validation set.

Why a validation set: picking settings by the TEST score makes the reported test
score too optimistic, and picking by the TRAIN score always prefers over-fitted
settings. So the train set of run.py is split once more:

    ratings --+-- train --+-- train' (90%)  -> fit every candidate setting
              |           +-- valid  (10%)  -> compare candidates, pick the best
              +-- test                      -> NOT used here. run.py scores the
                                               chosen setting on it exactly once.

The extra split uses the same split_by_user function and seed as run.py (team
rule), so valid looks like test: per user, and every valid user also has
ratings in train'.

The search runs in three stages, one group of knobs at a time; each stage
starts from the best setting found so far:
  1. the two fixes for the cause found by check_mf_top.py (rarely rated wines
     with a high b_i top every list). Both are always tried against the setting
     BEFORE any remedy (reg_bias = reg, bias_damping = 0), whatever config.yaml
     holds now, so every run reproduces the same comparison:
     1a. reg_bias      remedy 1: stronger L2 on the biases (shrinks every wine
                       by the same factor - kept to show why it does not work)
     1b. bias_damping  remedy 2: count-aware shrinkage, b_i ~ sum / (n_i + lambda)
     Choice: the remedies trade a little RMSE for better ranking, so among the
     candidates whose valid RMSE is at most 1% (RMSE_TOLERANCE) above the best,
     the one with the highest P@10 is taken (-> bias_damping 10 on X-Wines Slim).
  2. n_factors x reg  latent dimension and regularization strength (proposal)
  3. learning_rate    trained for up to MAX_EPOCHS epochs; the valid RMSE curve
                      picks n_epochs (the epoch where it is lowest)
     Stages 2-3 pick the lowest valid RMSE. On X-Wines Slim the gain over the
     defaults was only 0.3-0.4% while P@10 and coverage dropped, so config.yaml
     keeps the defaults (decided by hand; see the report).
R@10, NDCG@10 and coverage are reported next to every candidate so the
trade-off stays visible.

Outputs (outputs/):
  tune_mf_results.csv          every candidate with its valid-set metrics
                               (stage 1a, 1b, 2, 3 and the final setting)
  tune_mf_curve.csv            stage 3: train' and valid RMSE after every epoch,
                               one block per learning rate
  experiments/config_mf_<tag>.yaml   config.yaml with the chosen MF setting and
                                     output_dir outputs/<tag> (tag "tuned" if none)

Run from the project root (roughly 10 minutes on X-Wines Slim):
    python experiments/tune_mf.py                                  # full search
    python experiments/tune_mf.py --bias-damping 10 --tag l10     # stages 2-3 only,
                                                                   # with lambda fixed
    python experiments/tune_mf.py --data sample                    # quick smoke test
Without --tag the outputs above are overwritten. The chosen MF setting lives in
config.yaml; python run.py scores it on the test set.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Python only looks in experiments/ for imports when this file is run directly,
# so add the project root to find the recsys package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import PROJECT_ROOT, load_config  # noqa: E402
from recsys.data import adapter, cleaner, loader      # noqa: E402
from recsys.data.split import split_by_user           # noqa: E402
from recsys.eval.evaluator import evaluate_model      # noqa: E402
from recsys.models.mf import MatrixFactorization      # noqa: E402

# ----------------------------------------------------------------- search space
VALID_RATIO = 0.1                      # share of each user's TRAIN ratings held out
BIAS_REGS = [0.2, 0.5, 1.0]            # stage 1a (reg_bias = reg is added as baseline)
BIAS_DAMPINGS = [5, 10, 25, 50, 100]   # stage 1b (lambda; 0 = off is added as baseline)
FACTORS = [5, 10, 20, 50]              # stage 2
REGS = [0.02, 0.05, 0.1, 0.2]          # stage 2
LEARNING_RATES = [0.005, 0.01, 0.02]   # stage 3
MAX_EPOCHS = 40                        # stage 3
RMSE_TOLERANCE = 0.01                  # stage 1 choice: RMSE at most 1% above the best


def pick_stage1(rows, k):
    """Stage-1 choice: among candidates with valid RMSE <= best * (1 + RMSE_TOLERANCE),
    the one with the highest P@k (ties: lower RMSE)."""
    limit = min(r["RMSE"] for r in rows) * (1 + RMSE_TOLERANCE)
    allowed = [r for r in rows if r["RMSE"] <= limit]
    return max(allowed, key=lambda r: (r[f"P@{k}"], -r["RMSE"]))


def run_candidate(stage, params, train_sub, valid, items, cfg, curve=False):
    """Fit MF with `params` on train', score it on valid, return one result row."""
    model = MatrixFactorization(**params, seed=cfg["seed"])
    if curve:
        model.set_validation(valid)            # also log valid RMSE after every epoch
    start = time.perf_counter()
    model.fit(train_sub, items)
    fit_seconds = time.perf_counter() - start
    summary, _ = evaluate_model(model, train_sub, valid, cfg["evaluation"], cfg["seed"])

    # The cause found by check_mf_top.py, tracked per candidate: how many train
    # ratings do the 10 wines with the highest b_i have? (higher = less noise)
    counts = train_sub.groupby("item_id").size()
    top_bias = model.item_ids[np.argsort(-model.b_i)[:10]]
    top_bias_ratings = float(np.median(counts.reindex(top_bias).fillna(0)))

    k = cfg["evaluation"]["k"]
    row = {"stage": stage, **params,
           "RMSE": summary.get("RMSE"), "MAE": summary.get("MAE"),
           f"P@{k}": summary.get(f"P@{k}"), f"R@{k}": summary.get(f"R@{k}"),
           f"NDCG@{k}": summary.get(f"NDCG@{k}"), "Coverage": summary.get("Coverage"),
           "top10_bias_median_ratings": top_bias_ratings,
           "fit_seconds": round(fit_seconds, 1)}
    knobs = "  ".join(f"{key}={value}" for key, value in params.items())
    print(f"[stage {stage}] {knobs}  ->  RMSE={row['RMSE']:.4f}  "
          f"P@{k}={row[f'P@{k}']:.4f}  ({fit_seconds:.0f}s)", flush=True)
    return row, model


MF_KEYS = {"n_factors": int, "learning_rate": float, "reg": float, "reg_bias": float,
           "bias_damping": float, "n_epochs": int}


def write_experiment_config(mf_setting, tag, data_profile):
    """Write experiments/config_mf_<tag>.yaml: config.yaml with this MF setting and
    output_dir outputs/<tag>. Comments of config.yaml are not copied."""
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["data"] = data_profile
    raw["output_dir"] = f"outputs/{tag}"
    raw["model_params"]["mf"] = {key: cast(mf_setting[key]) for key, cast in MF_KEYS.items()}
    path = PROJECT_ROOT / "experiments" / f"config_mf_{tag}.yaml"
    header = (f"# Generated by experiments/tune_mf.py (tag: {tag}).\n"
              "# Same as config.yaml except model_params.mf and output_dir.\n"
              f"# Run:  python run.py --config experiments/config_mf_{tag}.yaml\n")
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Tune Matrix Factorization on a validation set")
    parser.add_argument("--data", choices=["slim", "sample"], default=None,
                        help="data profile (overrides config.yaml)")
    parser.add_argument("--bias-damping", type=float, default=None,
                        help="fix bias_damping (lambda) and skip stage 1, e.g. 10 to "
                             "re-check stages 2-3 for the chosen lambda")
    parser.add_argument("--tag", default="",
                        help="name of this run: outputs/tune_mf_<tag>_* and "
                             "experiments/config_mf_<tag>.yaml")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    cfg = load_config("config.yaml", args.data)
    out = cfg["paths"]["output_dir"]
    prefix = f"tune_mf_{args.tag}" if args.tag else "tune_mf"   # output file names
    k = cfg["evaluation"]["k"]
    started = time.perf_counter()

    # 1. Same data and train/test split as run.py, then train -> train' + valid
    wines_raw, ratings_raw = loader.load_xwines(cfg["paths"]["xwines_wines"],
                                                cfg["paths"]["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(wines_raw, ratings_raw, cfg)
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    train, _test = split_by_user(ratings, cfg["split"]["test_ratio"],
                                 cfg["min_user_ratings"], cfg["seed"])
    train_sub, valid = split_by_user(train, VALID_RATIO, cfg["min_user_ratings"], cfg["seed"])
    print(f"data={cfg['data']}  train={len(train):,} -> train'={len(train_sub):,} + "
          f"valid={len(valid):,}  (test is not used here)")
    if valid.empty:
        sys.exit("valid set is empty - too few ratings per user in this data profile")

    # Starting point = the current config.yaml setting
    current = dict(cfg["model_params"]["mf"])
    current.setdefault("reg_bias", current["reg"])
    current.setdefault("bias_damping", 0.0)
    rows = []

    if args.bias_damping is not None:
        # Fixed-lambda mode: no stage 1. The starting point is the current setting
        # with this lambda, so the final table shows what tuning the other knobs
        # adds on top of it.
        current["bias_damping"] = args.bias_damping
        print(f"\n--- stage 1 skipped: bias_damping fixed at {args.bias_damping:g} ---")
        row, _ = run_candidate("start", current, train_sub, valid, items, cfg)
        rows.append(row)
        best_reg_bias, best_damping = current["reg_bias"], current["bias_damping"]
    else:
        # 2. Stage 1: the two remedies, each against the setting BEFORE any remedy
        #    (first run = baseline), whatever config.yaml holds now.
        before = {**current, "reg_bias": current["reg"], "bias_damping": 0.0}
        stage1a = []
        print("\n--- stage 1a: reg_bias (remedy 1) ---")
        for reg_bias in [before["reg_bias"]] + [b for b in BIAS_REGS if b != before["reg_bias"]]:
            row, _ = run_candidate("1a", {**before, "reg_bias": reg_bias},
                                   train_sub, valid, items, cfg)
            stage1a.append(row)
        print("\n--- stage 1b: bias_damping (remedy 2) ---")
        stage1b = [stage1a[0]]                 # the baseline already has bias_damping 0
        for damping in BIAS_DAMPINGS:
            row, _ = run_candidate("1b", {**before, "bias_damping": damping},
                                   train_sub, valid, items, cfg)
            stage1b.append(row)
        rows += stage1a + stage1b[1:]
        best = pick_stage1(stage1a + stage1b[1:], k)
        best_reg_bias, best_damping = best["reg_bias"], best["bias_damping"]
        print(f"stage 1 choice: reg_bias={best_reg_bias:g}  bias_damping={best_damping:g}  "
              f"(RMSE within {RMSE_TOLERANCE:.0%} of the best, highest P@{k})")

    # 3. Stage 2: n_factors x reg, with the best stage-1 setting
    print("\n--- stage 2: n_factors x reg ---")
    stage2 = []
    for n_factors in FACTORS:
        for reg in REGS:
            params = {**current, "n_factors": n_factors, "reg": reg,
                      "reg_bias": best_reg_bias, "bias_damping": best_damping}
            row, _ = run_candidate(2, params, train_sub, valid, items, cfg)
            stage2.append(row)
    rows += stage2
    best = min(stage2, key=lambda r: r["RMSE"])
    best_factors, best_reg = best["n_factors"], best["reg"]

    # 4. Stage 3: learning rate; the valid curve picks the number of epochs
    print(f"\n--- stage 3: learning_rate (up to {MAX_EPOCHS} epochs) ---")
    curves = {}
    curve_rows = []                            # every epoch of every learning rate -> csv
    for lr in LEARNING_RATES:
        params = {**current, "n_factors": best_factors, "reg": best_reg,
                  "reg_bias": best_reg_bias, "bias_damping": best_damping,
                  "learning_rate": lr, "n_epochs": MAX_EPOCHS}
        row, model = run_candidate(3, params, train_sub, valid, items, cfg, curve=True)
        history = pd.DataFrame(model.history)
        best_epoch = int(history.loc[history["valid_rmse"].idxmin(), "epoch"])
        row["best_epoch"] = best_epoch
        row["best_epoch_valid_RMSE"] = float(history["valid_rmse"].min())
        rows.append(row)
        curves[lr] = (history, best_epoch)
        curve_rows += [{"learning_rate": lr, **log} for log in model.history]
    best_lr = min(curves, key=lambda lr: curves[lr][0]["valid_rmse"].min())
    best_epochs = curves[best_lr][1]

    # 5. The chosen setting, trained for the chosen number of epochs
    print("\n--- chosen setting ---")
    chosen = {**current, "n_factors": best_factors, "reg": best_reg, "reg_bias": best_reg_bias,
              "bias_damping": best_damping, "learning_rate": best_lr, "n_epochs": best_epochs}
    row, _ = run_candidate("final", chosen, train_sub, valid, items, cfg)
    rows.append(row)

    # 6. Save the tables (charts can be drawn from these files)
    results = pd.DataFrame(rows)
    results.round(4).to_csv(out / f"{prefix}_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).round(4).to_csv(out / f"{prefix}_curve.csv", index=False,
                                             encoding="utf-8-sig")

    # 7. Report
    shown = ["stage", "n_factors", "reg", "reg_bias", "bias_damping", "learning_rate", "n_epochs",
             "RMSE", "MAE", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage",
             "top10_bias_median_ratings", "best_epoch", "fit_seconds"]
    print("\n=== All candidates (valid set) ===")
    print(results[[c for c in shown if c in results]].round(4).to_string(index=False))

    baseline, final = rows[0], rows[-1]
    print("\n=== Starting point vs chosen setting (valid set) ===")
    compare = pd.DataFrame({"start": baseline, "chosen": final}).loc[
        ["n_factors", "reg", "reg_bias", "bias_damping", "learning_rate", "n_epochs", "RMSE", "MAE",
         f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage", "top10_bias_median_ratings"]]
    print(compare.apply(lambda col: col.map(lambda v: f"{float(v):.4g}")).to_string())

    # 8. An experiment config = config.yaml with this MF setting and its own output folder,
    #    so the test run of each model keeps its own results (config.yaml is not touched).
    tag = args.tag or "tuned"
    config_path = write_experiment_config(chosen, tag, cfg["data"])
    print(f"\nChosen MF setting written to {config_path.relative_to(PROJECT_ROOT)}")
    print("run.py uses config.yaml: copy a setting there only if you decide to adopt it,")
    print("then score it on the test set ONCE with  python run.py")
    print(f"\nSaved {out / (prefix + '_results.csv')} and {prefix}_curve.csv "
          f"({(time.perf_counter() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
