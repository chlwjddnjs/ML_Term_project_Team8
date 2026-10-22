"""Plots saved to outputs/ for the PPT.  Owner: shared (anyone may add a function)."""
import matplotlib

matplotlib.use("Agg")            # draw to files; works on servers and Colab
import matplotlib.pyplot as plt  # noqa: E402


def plot_rating_distributions(ratings, expert, path):
    """EDA: X-Wines user ratings vs converted expert ratings (both 1-5)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratings["rating"], bins=[x / 2 for x in range(2, 12)], alpha=0.6,
            density=True, label="X-Wines users")
    if expert is not None:
        ax.hist(expert["rating"].dropna(), bins=20, alpha=0.6, density=True,
                label="Wine Enthusiast (converted)")
    ax.set_xlabel("rating (1-5)")
    ax.set_ylabel("density")
    ax.set_title("Rating distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, metric, path):
    """Bar chart of one metric for every evaluated model."""
    data = results.dropna(subset=[metric]) if metric in results else results.iloc[0:0]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["model"], data[metric], color="#7A1F3D")
    for x, v in zip(data["model"], data[metric]):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison - {metric}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(history, path, title="Learning curve"):
    """Plot every numeric key of model.history against the epoch number."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in history[0]:
        if key != "epoch":
            ax.plot(epochs, [h[key] for h in history], marker="o", label=key)
    ax.set_xlabel("epoch")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
