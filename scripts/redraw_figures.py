"""Redraw the figures that only need saved result tables (no data download, no training).

    python scripts/redraw_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heartv2 import plots  # noqa: E402

GREY, BLUE = "#9aa3b2", plots.PALETTE[0]


def leakage(audit: pd.DataFrame, path: Path) -> None:
    """Accuracy and AUROC on the shared file with and without its duplicate rows."""
    shared, dedup = audit.iloc[0], audit.iloc[1]
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    x = np.arange(2)
    w = 0.36
    for off, row, colour, label in ((-w / 2, shared, GREY, "As shared: 1,025 rows (97% of test rows also in training)"),
                                    (w / 2, dedup, BLUE, "Duplicates removed: 302 rows")):
        means = [row["accuracy_mean"], row["auroc_mean"]]
        sds = [row["accuracy_sd"], row["auroc_sd"]]
        ax.bar(x + off, means, w, yerr=sds, capsize=5, color=colour, label=label)
        for xi, m in zip(x + off, means):
            ax.text(xi, 0.535, "%.3f" % m, ha="center", color="white", fontweight="bold", fontsize=11)
    ax.set_xticks(x, ["Accuracy", "AUROC"])
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("Mean over 20 random 80/20 splits")
    ax.set_title("Random Forest on the widely shared heart disease file")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    res = ROOT / "results"
    folds = pd.read_csv(res / "internal_cv_folds.csv")
    loso = pd.read_csv(res / "loso_all_metrics.csv")
    loso = loso[loso["features"] == "all"]
    for out in (res / "figures", ROOT / "paper" / "figures"):
        plots.internal_vs_external(folds, loso, out / "internal_vs_external.png")
        plt.close("all")
    leakage(pd.read_csv(res / "leakage_audit.csv"), ROOT / "paper" / "figures" / "leakage.png")
    print("Redrew internal_vs_external.png and leakage.png")


if __name__ == "__main__":
    main()
