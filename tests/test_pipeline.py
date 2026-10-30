import numpy as np
import pytest

from heartv2.data import FEATURES_ALL, FEATURES_CORE, drop_duplicates, load_uci, site_summary
from heartv2.evaluate import (
    bootstrap_ci, ece, internal_cv, leakage_audit, leave_one_site_out, metrics, subgroup_auroc,
)
from heartv2.models import get_models


@pytest.fixture(scope="module")
def df(uci_dir):
    d, _ = drop_duplicates(load_uci(uci_dir))
    return d


def test_load_and_clean(uci_dir):
    raw = load_uci(uci_dir)
    assert set(raw["site"]) == {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}
    assert len(raw) == 303 + 295 + 123 + 200
    assert raw.loc[raw.site == "Switzerland", "chol"].isna().all()  # 0 recoded as missing
    assert raw["target"].isin([0, 1]).all()
    deduped, n = drop_duplicates(raw)
    assert n == 1 and len(deduped) == len(raw) - 1


def test_site_summary(df):
    s = site_summary(df)
    assert list(s.index) == ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    assert s.loc["Switzerland", "missing_%_chol"] == 100.0


def test_metrics_perfect_and_ece():
    y = np.array([0, 0, 1, 1])
    m = metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auroc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert ece(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    lo, hi = bootstrap_ci(np.r_[np.zeros(50), np.ones(50)], np.linspace(0, 1, 100), n=200)
    assert lo <= hi <= 1.0


def test_cv_loso_subgroups(df):
    models = get_models(include=["LogisticRegression", "RandomForest"], verbose=False)
    folds, oof = internal_cv(df, models, FEATURES_ALL, n_splits=3, n_repeats=1, verbose=False)
    assert set(folds["model"]) == set(models) and len(folds) == 2 * 3
    assert oof.shape == (len(df), 2) and oof.notna().all().all()
    assert folds["auroc"].mean() > 0.6  # synthetic signal is learnable
    res, preds = leave_one_site_out(df, models, FEATURES_CORE, n_boot=50, verbose=False)
    assert len(res) == 4 * 2 and (res["auroc_lo"] <= res["auroc_hi"]).all()
    sub = subgroup_auroc(df, oof["LogisticRegression"], n_boot=50)
    assert {"sex=female", "sex=male"} <= set(sub["group"])


def test_leakage_audit_detects_duplicates(merged_1025):
    out = leakage_audit(merged_1025, n_repeats=3)
    dup, clean = out.iloc[0], out.iloc[1]
    assert dup["n_rows"] == 1025 and clean["n_rows"] < 1025
    assert dup["test_rows_seen_in_train"] > 0.5 and clean["test_rows_seen_in_train"] == 0
    assert dup["accuracy_mean"] > clean["accuracy_mean"]
