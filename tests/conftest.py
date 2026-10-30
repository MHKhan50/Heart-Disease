"""Synthetic data in the exact UCI ``processed.*.data`` format, for offline tests."""

import numpy as np
import pandas as pd
import pytest

from heartv2.data import COLUMNS

# (stem, n, missing rate for slope/ca/thal, cholesterol recorded as 0?)
SPEC = [("cleveland", 303, 0.02, False), ("hungarian", 294, 0.6, False),
        ("switzerland", 123, 0.8, True), ("va", 200, 0.7, False)]


def _site(stem, n, miss, chol_zero, rng):
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(1, 5, n)
    trestbps = rng.normal(132, 18, n).round()
    chol = np.zeros(n) if chol_zero else rng.normal(240, 50, n).round()
    fbs = rng.integers(0, 2, n)
    restecg = rng.integers(0, 3, n)
    thalach = rng.normal(140, 25, n).round()
    exang = rng.integers(0, 2, n)
    oldpeak = np.abs(rng.normal(1, 1.1, n)).round(1)
    slope = rng.integers(1, 4, n).astype(float)
    ca = rng.integers(0, 4, n).astype(float)
    thal = rng.choice([3.0, 6.0, 7.0], n)
    logit = 0.04 * (age - 54) + 0.8 * sex + 0.6 * (cp == 4) + 0.9 * exang + 0.7 * oldpeak - 0.02 * (thalach - 140) - 1
    num = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), rng.integers(1, 5, n), 0)
    df = pd.DataFrame(dict(zip(COLUMNS, [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                          exang, oldpeak, slope, ca, thal, num])))
    for col in ("slope", "ca", "thal"):
        df.loc[rng.random(n) < miss, col] = np.nan
    return df


@pytest.fixture(scope="session")
def uci_dir(tmp_path_factory):
    rng = np.random.default_rng(7)
    d = tmp_path_factory.mktemp("uci")
    for stem, n, miss, chol_zero in SPEC:
        df = _site(stem, n, miss, chol_zero, rng)
        if stem == "hungarian":
            df = pd.concat([df, df.iloc[[0]]])  # one exact duplicate, as in real data
        df.astype(object).where(df.notna(), "?").to_csv(
            d / f"processed.{stem}.data", header=False, index=False)
    return d


@pytest.fixture(scope="session")
def merged_1025(uci_dir):
    """Mimic the shared 1,025-row file: ~302 unique rows repeated."""
    rng = np.random.default_rng(1)
    base = _site("cleveland", 302, 0.0, False, rng)
    base["target"] = (base.pop("num") > 0).astype(int)
    base = base.rename(columns={})
    idx = np.concatenate([np.arange(302), rng.integers(0, 302, 1025 - 302)])
    return base.iloc[idx].reset_index(drop=True)
