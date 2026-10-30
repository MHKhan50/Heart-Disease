"""Preprocessing and model zoo. Optional libraries are skipped when unavailable."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    """Imputation is fitted inside each training fold, so no test information leaks in.

    Missing-value indicators are kept as features because missingness differs
    strongly between hospitals.
    """
    num = [c for c in NUMERIC if c in features]
    binary = [c for c in BINARY if c in features]
    cat = [c for c in CATEGORICAL if c in features]
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), num),
            ("bin", SimpleImputer(strategy="most_frequent", add_indicator=True), binary),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat),
        ],
        remainder="drop",
    )


def _tabpfn(seed: int):
    import torch
    from tabpfn import TabPFNClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:  # prefer the openly available v2 weights when the API offers a choice
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device, random_state=seed)
    except Exception:
        return TabPFNClassifier(device=device, random_state=seed)


def _candidates(seed: int) -> dict:
    """Name -> zero-argument factory. Defaults are fixed in advance, not tuned on test data."""
    c = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=5000),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        c["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=seed)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        c["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier

        c["CatBoost"] = lambda: CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=4, verbose=0, random_seed=seed)
    except ImportError:
        pass
    try:
        import tabpfn  # noqa: F401

        c["TabPFN"] = lambda: _tabpfn(seed)
    except ImportError:
        pass
    return c


def get_models(seed: int = 42, include=None, verbose: bool = True) -> dict:
    """Return the models that import and pass a small smoke test.

    TabPFN weights are downloaded on first use; if that fails (no internet,
    gated weights) the model is skipped and the reason is printed.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=60) > 0).astype(int)
    ok = {}
    for name, factory in _candidates(seed).items():
        if include is not None and name not in include:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                factory().fit(X, y).predict_proba(X)
            ok[name] = factory
        except Exception as e:  # pragma: no cover - environment dependent
            if verbose:
                print(f"[skip] {name}: {type(e).__name__}: {str(e)[:200]}")
    if verbose:
        print("Models:", ", ".join(ok))
    return ok


def make_pipeline(factory, features: list[str]) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(features)), ("model", factory())])
