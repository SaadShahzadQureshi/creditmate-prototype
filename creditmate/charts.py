"""Charts for the results. Colorblind-friendly blue and orange throughout."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREY = "#8C8C8C"


def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def roc_chart(y_test, pd_test, rule_approved, auc, path):
    """ROC curve for the model, with the land-title rule as a single point."""
    fpr, tpr, _ = roc_curve(y_test, pd_test)
    declined = ~rule_approved
    rule_tpr = declined[y_test == 1].mean()   # share of defaulters the rule turns away
    rule_fpr = declined[y_test == 0].mean()   # share of good payers the rule turns away

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5, label=f"Farm-data model (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="Random guess")
    ax.scatter([rule_fpr], [rule_tpr], color=ORANGE, s=90, zorder=3, marker="D",
               label="Land-title rule")
    ax.set_title("Farm data separates good payers from defaulters;\nthe land-title rule barely beats a coin flip",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Good payers turned away (false positive rate)")
    ax.set_ylabel("Defaulters turned away (true positive rate)")
    ax.legend(loc="lower right", frameon=False)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def comparison_chart(results, path):
    """Defaults and reach at the same approval rate."""
    rule = results["land_title_rule"]
    model = results["model_at_same_approval_rate"]
    labels = ["Land-title rule", "Farm-data model"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    panels = [
        ("Default rate among approved farmers", [rule["default_rate_approved"], model["default_rate_approved"]], 30),
        ("Approved farmers without a land title", [rule["share_approved_without_title"], model["share_approved_without_title"]], 100),
    ]
    for ax, (title, values, top) in zip(axes, panels):
        bars = ax.bar(labels, [v * 100 for v in values], color=[ORANGE, BLUE], width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, top)
        ax.set_ylabel("Percent")
        _tidy(ax)
    fig.suptitle(f"Approving the same {rule['approval_rate']:.0%} of farmers, farm data cuts defaults\nand reaches farmers without titles",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def drivers_chart(drivers, path):
    """What pushes risk up or down, in standardized units."""
    names = [n.replace("_", " ") for n in drivers.index]
    colors = [ORANGE if v > 0 else BLUE for v in drivers.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, drivers.values, color=colors)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_title("Strong yields and healthy herds lower risk;\nunsteady yields and selling through an arthi raise it",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Effect on default risk (standardized coefficient)")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def score_chart(scores, y_test, path):
    """Score distributions for farmers who repaid and who defaulted."""
    bins = np.arange(300, 860, 20)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores[y_test == 0], bins=bins, color=BLUE, alpha=0.75, label="Repaid")
    ax.hist(scores[y_test == 1], bins=bins, color=ORANGE, alpha=0.75, label="Defaulted")
    ax.set_title("Farmers who defaulted cluster at lower scores", fontweight="bold", fontsize=12)
    ax.set_xlabel("CreditMate score (300 to 850, higher is safer)")
    ax.set_ylabel("Farmers")
    ax.legend(frameon=False)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
