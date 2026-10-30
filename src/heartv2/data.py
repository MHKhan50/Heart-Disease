"""Loading and cleaning the UCI heart disease databases (four hospitals)."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# File name stem -> readable site name.
SITES = {
    "cleveland": "Cleveland",
    "hungarian": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}

UCI_DIR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

FEATURES_ALL = [c for c in COLUMNS if c != "num"]
# Attributes that are largely missing outside Cleveland.
HIGH_MISSING = ["slope", "ca", "thal"]
FEATURES_CORE = [c for c in FEATURES_ALL if c not in HIGH_MISSING]


def _parse(raw: bytes | str, site: str) -> pd.DataFrame:
    text = raw.decode("latin-1") if isinstance(raw, bytes) else raw
    df = pd.read_csv(
        io.StringIO(text), header=None, names=COLUMNS,
        na_values=["?", " ?", "-9", "-9.0"], skipinitialspace=True,
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df["site"] = site
    return df


def _find_local(root: Path) -> dict[str, Path]:
    found = {}
    for stem in SITES:
        hits = sorted(root.rglob(f"processed.{stem}.data"))
        if hits:
            found[stem] = hits[0]
    return found


def _download() -> dict[str, bytes]:
    files = {}
    try:
        for stem in SITES:
            with urllib.request.urlopen(f"{UCI_DIR_URL}processed.{stem}.data", timeout=60) as r:
                files[stem] = r.read()
        return files
    except Exception:
        files = {}
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    for stem in SITES:
        name = next(n for n in archive.namelist() if n.endswith(f"processed.{stem}.data"))
        files[stem] = archive.read(name)
    return files


def load_uci(source: str | Path | None = None, search_roots=("/kaggle/input",)) -> pd.DataFrame:
    """Load the four processed UCI files into one cleaned DataFrame.

    Lookup order: ``source`` directory, then ``search_roots`` (e.g. attached
    Kaggle datasets), then a download from the UCI repository.
    """
    roots = [Path(source)] if source is not None else [Path(r) for r in search_roots]
    raw: dict[str, bytes | str] = {}
    for root in roots:
        if root.exists():
            local = _find_local(root)
            if len(local) == len(SITES):
                raw = {k: p.read_bytes() for k, p in local.items()}
                break
    if not raw:
        if source is not None:
            raise FileNotFoundError(f"processed.*.data files for all four sites not found under {source}")
        raw = _download()
    df = pd.concat([_parse(raw[stem], SITES[stem]) for stem in SITES], ignore_index=True)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Recode impossible zeros as missing and build the binary target."""
    df = df.copy()
    # A serum cholesterol or resting blood pressure of 0 is not physiological;
    # in the Switzerland and VA files 0 is used for "not measured".
    for col in ("chol", "trestbps"):
        df.loc[df[col] == 0, col] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate records (same site, features and label)."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def site_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows, disease prevalence and missingness (%) per site."""
    g = df.groupby("site", sort=False)
    out = pd.DataFrame({"n": g.size(), "prevalence": g["target"].mean().round(3)})
    miss = g[FEATURES_ALL].apply(lambda x: x.isna().mean() * 100).round(1)
    miss = miss.loc[:, miss.max() > 0]
    return out.join(miss.add_prefix("missing_%_"))


def load_kaggle_merged(path: str | Path | None = None, search_roots=("/kaggle/input",)) -> pd.DataFrame | None:
    """Load the widely shared 1,025-row merged Kaggle file (``heart.csv``), if available."""
    if path is not None:
        return pd.read_csv(path)
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.csv")):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if len(df) == 1025 and "target" in df.columns and "thalach" in df.columns:
                return df
    return None
