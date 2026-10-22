"""End-to-end pipeline: load -> align -> clean -> split -> train -> evaluate -> save.
Owner: shared.

Usage:
    python run.py                     # real experiment (X-Wines Slim)
    python run.py --data sample       # smoke test with the small sample in data/sample
    python run.py --models mf hybrid  # run only some models

A model that is not implemented yet (raises NotImplementedError) is reported
as TODO and skipped, so the pipeline always runs to the end.
All results are written to outputs/ (CSV tables + PNG plots).
"""
import argparse
import json

import pandas as pd

from recsys.config import load_config
from recsys.data import adapter, cleaner, loader
from recsys.data.split import split_by_user
from recsys.eval import analysis, cross_dataset
from recsys.eval.evaluator import evaluate_model
from recsys.models import REGISTRY
from recsys.utils import plots


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_model(name, cfg):
    """Create a model from its config name, passing its hyper-parameters."""
    params = dict(cfg["model_params"].get(name) or {})
    if name == "mf":
        params["seed"] = cfg["seed"]
    if name == "hybrid":
        params["mf_params"] = {**cfg["model_params"]["mf"], "seed": cfg["seed"]}
        params["content_params"] = cfg["model_params"]["content"]
    return REGISTRY[name](**params)


def main():
    parser = argparse.ArgumentParser(description="Wine recommendation pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", choices=["slim", "sample"], default=None,
                        help="data profile (overrides config.yaml)")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"subset of models to run: {list(REGISTRY)}")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data)
    out = cfg["paths"]["output_dir"]
    model_names = args.models or cfg["models"]

    # ------------------------------------------------------------------ 1. data
    section(f"1. Load & align data  (profile: {cfg['data']})")
    wines_raw, ratings_raw = loader.load_xwines(cfg["paths"]["xwines_wines"],
                                                cfg["paths"]["xwines_ratings"])
    items, ratings = adapter.xwines_to_common(wines_raw, ratings_raw, cfg)
    cleaner.quality_report(items, "X-Wines items (raw)")
    cleaner.quality_report(ratings, "X-Wines ratings (raw)")
    items = cleaner.clean_items(items)
    ratings = cleaner.clean_ratings(ratings)
    print(f"\nusers={ratings['user_id'].nunique():,}  wines={len(items):,}  "
          f"ratings={len(ratings):,}")

    expert, matches = None, None
    winemag_raw = loader.load_winemag(cfg["paths"]["winemag"])
    if winemag_raw is None:
        print(f"\n[skip] {cfg['paths']['winemag'].name} not found -> cross-dataset step skipped")
    else:
        expert = adapter.winemag_to_common(winemag_raw, cfg, scale_method="linear")
        cleaner.quality_report(expert, "Wine Enthusiast (raw)")
        expert = cleaner.clean_expert(expert)
        for method in ("fuzzy", "exact"):          # use fuzzy once it is implemented
            try:
                matches = adapter.match_items(items, expert, method=method)
                print(f"[match:{method}] {matches['item_id'].nunique():,} X-Wines wines "
                      f"matched to {len(matches):,} expert reviews")
                break
            except NotImplementedError as e:
                print(f"[TODO] {e}")

    plots.plot_rating_distributions(ratings, expert, out / "eda_rating_distribution.png")

    # ----------------------------------------------------------------- 2. split
    section("2. Train/test split")
    train, test = split_by_user(ratings, cfg["split"]["test_ratio"],
                                cfg["min_user_ratings"], cfg["seed"])
    print(f"train={len(train):,}  test={len(test):,}  "
          f"test users={test['user_id'].nunique():,}")

    # ------------------------------------------------------- 3. train + evaluate
    section("3. Train & evaluate models")
    results, per_user, fitted = [], {}, {}
    for name in model_names:
        model = build_model(name, cfg)
        try:
            model.fit(train, items)
            summary, users_df = evaluate_model(model, train, test,
                                               cfg["evaluation"], cfg["seed"])
        except NotImplementedError as e:
            print(f"[TODO] {model.name:<10} skipped: {e}")
            results.append({"model": model.name, "status": "TODO"})
            continue
        summary["status"] = "ok"
        results.append(summary)
        per_user[model.name] = users_df
        fitted[model.name] = model
        print(f"[ok]   {model.name:<10} " + "  ".join(
            f"{k}={v:.4f}" for k, v in summary.items() if isinstance(v, float)))
        if model.history:
            plots.plot_learning_curve(model.history, out / f"learning_curve_{model.name}.png",
                                      title=f"{model.name} learning curve")

    results_df = pd.DataFrame(results)
    first = [c for c in ("model", "status", "RMSE", "MAE") if c in results_df]
    results_df = results_df[first + [c for c in results_df if c not in first]]
    results_df.to_csv(out / "results.csv", index=False)
    k = cfg["evaluation"]["k"]
    for metric in ("RMSE", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage"):
        plots.plot_model_comparison(results_df, metric, out / f"compare_{metric.replace('@', '_at_')}.png")

    cold = analysis.cold_start_analysis(per_user)
    if not cold.empty:
        cold.to_csv(out / "cold_start_analysis.csv", index=False)

    # -------------------------------------------------------- 4. cross-dataset
    if matches is not None and len(matches):
        section("4. Cross-dataset validation (X-Wines vs Wine Enthusiast experts)")
        expert_scores = cross_dataset.expert_score_per_item(matches, expert)
        rows = [{"model": "X-Wines mean rating (reference)",
                 **cross_dataset.rank_agreement(train.groupby("item_id")["rating"].mean(),
                                                expert_scores, k)}]
        for name, model in fitted.items():
            try:
                quality = cross_dataset.model_quality_per_item(model, train, items["item_id"])
                rows.append({"model": name, **cross_dataset.rank_agreement(quality, expert_scores, k)})
            except NotImplementedError as e:
                print(f"[TODO] {e}")
                break
        cross_df = pd.DataFrame(rows)
        cross_df.to_csv(out / "cross_dataset.csv", index=False)
        print(cross_df.to_string(index=False))

    # ----------------------------------------------------------------- summary
    section("Summary")
    print(results_df.to_string(index=False))
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"data": cfg["data"], "seed": cfg["seed"], "split": cfg["split"],
                   "evaluation": cfg["evaluation"], "model_params": cfg["model_params"]},
                  f, indent=2)
    print(f"\nSaved results and plots to {out}/")


if __name__ == "__main__":
    main()
