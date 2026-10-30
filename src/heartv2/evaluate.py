"""Metrics, internal cross-validation, leave-one-hospital-out and the leakage audit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .models import make_pipeline

EPS = 1e-6


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(
        abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
        for b in range(n_bins) if (bins == b).any()
    ))


def calibration_fit(y, p) -> tuple[float, float]:
    """Calibration intercept and slope (logistic recalibration on logit(p)).

    A well calibrated model has intercept 0 and slope 1; slope < 1 means
    predictions are too extreme.
    """
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    intercept, slope = calibration_fit(y, p)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
        "cal_intercept": intercept,
        "cal_slope": slope,
        "obs_exp": y.mean() / max(p.mean(), EPS),
        "accuracy": (pred == y).mean(),
        "sensitivity": tp / max((y == 1).sum(), 1),
        "specificity": tn / max((y == 0).sum(), 1),
    }


def bootstrap_ci(y, p, fn=roc_auc_score, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(fn(y[idx], p[idx]))
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def _fit_predict(factory, features, X_tr, y_tr, X_te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline(factory, features).fit(X_tr[features], y_tr)
        return pipe.predict_proba(X_te[features])[:, 1]


def internal_cv(df, models: dict, features, n_splits=5, n_repeats=5, seed=42, verbose=True):
    """Repeated stratified k-fold on the pooled data (strata = site x label).

    Returns per-fold metrics and out-of-fold probabilities averaged over repeats.
    """
    strata = df["site"].astype(str) + "_" + df["target"].astype(str)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(cv.split(df, strata))
    rows, oof = [], {}
    y = df["target"].to_numpy()
    for name, factory in models.items():
        acc = np.zeros(len(df))
        for i, (tr, te) in enumerate(splits):
            p = _fit_predict(factory, features, df.iloc[tr], y[tr], df.iloc[te])
            acc[te] += p
            rows.append({"model": name, "fold": i, **metrics(y[te], p)})
        oof[name] = acc / n_repeats
        if verbose:
            print(f"  {name}: done")
    return pd.DataFrame(rows), pd.DataFrame(oof, index=df.index)


def summarise_cv(folds: pd.DataFrame, cols=("auroc", "brier", "ece", "accuracy", "sensitivity", "specificity")):
    g = folds.groupby("model", sort=False)[list(cols)]
    mean, sd = g.mean(), g.std()
    out = mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)
    return out.loc[mean["auroc"].sort_values(ascending=False).index]


def leave_one_site_out(df, models: dict, features, n_boot=1000, seed=42, verbose=True):
    """Train on three hospitals, test on the held-out one."""
    rows, preds = [], {}
    y = df["target"].to_numpy()
    for site in df["site"].unique():
        te = (df["site"] == site).to_numpy()
        for name, factory in models.items():
            p = _fit_predict(factory, features, df[~te], y[~te], df[te])
            lo, hi = bootstrap_ci(y[te], p, n=n_boot, seed=seed)
            rows.append({"held_out": site, "model": name, "n": int(te.sum()),
                         "prevalence": y[te].mean(), **metrics(y[te], p),
                         "auroc_lo": lo, "auroc_hi": hi})
            preds[(site, name)] = p
        if verbose:
            print(f"  held out {site}: done")
    return pd.DataFrame(rows), preds


def subgroup_auroc(df, p, n_boot=1000, seed=42) -> pd.DataFrame:
    """AUROC by sex and age band using out-of-fold predictions."""
    p = np.asarray(p)
    groups = {
        "sex=female": df["sex"] == 0,
        "sex=male": df["sex"] == 1,
        "age<50": df["age"] < 50,
        "age 50-59": (df["age"] >= 50) & (df["age"] < 60),
        "age>=60": df["age"] >= 60,
    }
    rows = []
    y = df["target"].to_numpy()
    for g, m in groups.items():
        m = m.to_numpy()
        if m.sum() < 10 or y[m].min() == y[m].max():
            continue
        lo, hi = bootstrap_ci(y[m], p[m], n=n_boot, seed=seed)
        rows.append({"group": g, "n": int(m.sum()), "prevalence": y[m].mean(),
                     "auroc": roc_auc_score(y[m], p[m]), "auroc_lo": lo, "auroc_hi": hi})
    return pd.DataFrame(rows)


def leakage_audit(df_merged: pd.DataFrame, n_repeats=20, test_size=0.2, seed=42) -> pd.DataFrame:
    """Compare random-split scores on the 1,025-row merged file with and without duplicates."""
    target = "target"
    feats = [c for c in df_merged.columns if c != target]
    versions = {"as shared (with duplicates)": df_merged,
                "duplicates removed": df_merged.drop_duplicates()}
    rows = []
    for label, d in versions.items():
        X, y = d[feats].to_numpy(), d[target].to_numpy()
        for r in range(n_repeats):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=seed + r)
            train_rows = {tuple(row) for row in X_tr}
            seen = np.mean([tuple(row) in train_rows for row in X_te])
            rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
            p = rf.predict_proba(X_te)[:, 1]
            rows.append({"data": label, "n_rows": len(d), "repeat": r,
                         "test_rows_seen_in_train": seen,
                         "accuracy": ((p >= 0.5) == y_te).mean(), "auroc": roc_auc_score(y_te, p)})
    res = pd.DataFrame(rows)
    return res.groupby("data", sort=False).agg(
        n_rows=("n_rows", "first"),
        test_rows_seen_in_train=("test_rows_seen_in_train", "mean"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
    ).round(3)


def loso_table(res: pd.DataFrame, value: str = "auroc") -> pd.DataFrame:
    """Model x held-out hospital table formatted as 'AUROC [95% CI]'."""
    cell = res.apply(lambda r: f"{r[value]:.3f} [{r[value + '_lo']:.3f}, {r[value + '_hi']:.3f}]"
                     if value + "_lo" in r else f"{r[value]:.3f}", axis=1)
    t = res.assign(cell=cell).pivot(index="model", columns="held_out", values="cell")
    order = res.groupby("model")[value].mean().sort_values(ascending=False).index
    return t.loc[order, res["held_out"].unique()]
