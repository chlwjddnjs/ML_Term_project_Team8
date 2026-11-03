"""Result analysis for the report/PPT.  Owner: member 4.

`cold_start_analysis` is implemented as an example; add more analyses here
(e.g. error analysis of the worst predictions, popularity bias).
"""
import numpy as np
import pandas as pd


def cold_start_analysis(per_user_results, bins=(0, 2, 5, 10, 20, 10_000)):
    """Average ranking metrics per group of users with similar train size.

    Args:
        per_user_results: {model_name: per_user DataFrame from evaluate_model}
    Returns:
        DataFrame indexed by (model, n_train group).
    Shows whether the hybrid model really helps users with few ratings.
    """
    frames = []
    for name, df in per_user_results.items():
        if df.empty:
            continue
        df = df.copy()
        df["group"] = pd.cut(df["n_train"], bins=list(bins), right=False)
        metric_cols = [c for c in df.columns if "@" in c]
        g = df.groupby("group", observed=True)[metric_cols].mean()
        g["n_users"] = df.groupby("group", observed=True).size()
        g = g.reset_index()
        g.insert(0, "model", name)
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def predictions_with_errors(model, test):
    """Every held-out rating with the model's prediction and context columns.

    Columns: user_id, item_id, rating, pred, error (= pred - true), abs_error,
             user_n_train (ratings the user has in train),
             item_n_train (ratings the wine has in train).
    Only for rating models (model.predicts_ratings is True).
    """
    if not model.predicts_ratings:
        raise TypeError(f"{model.name} does not predict ratings")
    df = test[["user_id", "item_id", "rating"]].copy()
    preds = []
    for user_id, group in df.groupby("user_id"):
        scores = np.clip(model.score(user_id, group["item_id"].to_numpy()), 1, 5)
        preds.append(pd.Series(scores, index=group.index))
    df["pred"] = pd.concat(preds).reindex(df.index)
    df["error"] = df["pred"] - df["rating"]
    df["abs_error"] = df["error"].abs()

    # How much the model knew about this user / wine
    df["user_n_train"] = df["user_id"].map(
        {u: len(s) for u, s in model.seen.items()}).fillna(0).astype(int)
    item_counts = pd.Series([i for s in model.seen.values() for i in s]).value_counts()
    df["item_n_train"] = df["item_id"].map(item_counts).fillna(0).astype(int)
    return df


def worst_predictions(model, test, n=20):
    """The n held-out ratings with the largest |true - predicted| error,
    joined with wine attributes, to explain failures in the presentation.

    Typical patterns to look for: users with few train ratings, rarely rated
    wines, or extreme ratings (1.0 / 5.0) that the model pulls toward the mean.
    """
    df = predictions_with_errors(model, test).nlargest(n, "abs_error")
    attrs = [c for c in ("item_id", "name", "type", "country", "grapes")
             if c in model.items.columns]
    return df.merge(model.items[attrs], on="item_id", how="left").reset_index(drop=True)


def error_by_group(model, test, by="user_n_train",
                   bins=(0, 2, 5, 10, 20, 10_000)):
    """Mean absolute error per group of users (by="user_n_train") or wines
    (by="item_n_train"), with the same bins as cold_start_analysis.
    Turns the worst-case list into a general statement for the report
    (e.g. "error is highest for wines with fewer than 5 ratings")."""
    df = predictions_with_errors(model, test)
    df["group"] = pd.cut(df[by], bins=list(bins), right=False)
    g = df.groupby("group", observed=True).agg(
        MAE=("abs_error", "mean"), mean_error=("error", "mean"), n_ratings=("abs_error", "size"))
    g = g.reset_index()
    g.insert(0, "model", model.name)
    return g


def error_by_true_rating(model, test):
    """Mean error per true rating value (1.0, 1.5, ... 5.0).

    mean_error > 0 means the model predicts too HIGH for that rating, < 0 too LOW.
    A model that pulls every prediction toward the global mean shows a large
    positive error for low ratings and a negative error for 5.0 ratings."""
    df = predictions_with_errors(model, test)
    g = df.groupby("rating").agg(
        MAE=("abs_error", "mean"), mean_error=("error", "mean"),
        mean_pred=("pred", "mean"), n_ratings=("abs_error", "size"))
    g = g.reset_index()
    g.insert(0, "model", model.name)
    return g


def failure_patterns(model, test, n=20, few=5, extreme=(2.0, 5.0)):
    """How much more often the worst predictions have each known failure pattern
    than test ratings in general.

    Patterns (a rating can have several):
        cold user   - user has fewer than `few` train ratings
        rare wine   - wine has fewer than `few` train ratings
        low rating  - true rating <= extreme[0]
        top rating  - true rating >= extreme[1]
    Returns one row per pattern: share among the n worst predictions, share among
    all test ratings, and their ratio (lift; > 1 means the pattern is
    over-represented in the failures)."""
    df = predictions_with_errors(model, test)
    worst = df.nlargest(n, "abs_error")
    checks = {
        f"cold user (<{few} train ratings)": lambda d: d["user_n_train"] < few,
        f"rare wine (<{few} train ratings)": lambda d: d["item_n_train"] < few,
        f"low rating (<= {extreme[0]})": lambda d: d["rating"] <= extreme[0],
        f"top rating (>= {extreme[1]})": lambda d: d["rating"] >= extreme[1],
    }
    rows = []
    for pattern, check in checks.items():
        share_worst, share_all = check(worst).mean(), check(df).mean()
        rows.append({"model": model.name, "pattern": pattern,
                     "share_in_worst": share_worst, "share_in_all_test": share_all,
                     "lift": share_worst / share_all if share_all > 0 else np.nan})
    return pd.DataFrame(rows)
