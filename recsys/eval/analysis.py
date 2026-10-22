"""Result analysis for the report/PPT.  Owner: member 4.

`cold_start_analysis` is implemented as an example; add more analyses here
(e.g. error analysis of the worst predictions, popularity bias).
"""
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


def worst_predictions(model, test, n=20):
    """TODO(4): n held-out ratings with the largest |true - predicted| error,
    joined with wine attributes, to explain failures in the presentation."""
    raise NotImplementedError("worst_predictions - TODO(4)")
