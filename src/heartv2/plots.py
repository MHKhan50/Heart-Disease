"""Figures for the internal vs external comparison, calibration and SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def internal_vs_external(cv_folds: pd.DataFrame, loso: pd.DataFrame, path=None):
    """Mean internal CV AUROC next to each held-out hospital's AUROC with 95% CI."""
    models = cv_folds.groupby("model")["auroc"].mean().sort_values(ascending=False).index.tolist()
    sites = loso["held_out"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.8 / (len(sites) + 1)
    x = np.arange(len(models))
    internal = cv_folds.groupby("model")["auroc"].mean().reindex(models)
    ax.bar(x - 0.4 + width / 2, internal, width, label="Internal CV (pooled)", color="0.35")
    for i, site in enumerate(sites):
        s = loso[loso["held_out"] == site].set_index("model").reindex(models)
        err = np.vstack([s["auroc"] - s["auroc_lo"], s["auroc_hi"] - s["auroc"]])
        ax.bar(x - 0.4 + width * (i + 1.5), s["auroc"], width, yerr=err, capsize=2,
               label=f"External: {site}")
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.set_title("Internal cross-validation vs leave-one-hospital-out")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def calibration_curves(df: pd.DataFrame, preds: dict, model: str, n_bins: int = 8, path=None):
    """Reliability diagram per held-out hospital for one model (quantile bins)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    for site in df["site"].unique():
        if (site, model) not in preds:
            continue
        y = df.loc[df["site"] == site, "target"].to_numpy()
        p = preds[(site, model)]
        q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
        xs = [p[b == k].mean() for k in range(len(q) - 1) if (b == k).any()]
        ys = [y[b == k].mean() for k in range(len(q) - 1) if (b == k).any()]
        ax.plot(xs, ys, marker="o", label=site)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration on held-out hospitals: {model}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def shap_summary(df: pd.DataFrame, factory, features, path=None, seed: int = 42):
    """SHAP values for a tree model fitted on the pooled data. Returns None if shap is missing."""
    try:
        import shap
    except ImportError:
        print("[skip] shap not installed")
        return None
    from .models import make_pipeline

    pipe = make_pipeline(factory, features).fit(df[features], df["target"])
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    X = pd.DataFrame(prep.transform(df[features]), columns=prep.get_feature_names_out())
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig
