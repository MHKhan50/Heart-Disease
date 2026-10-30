"""Run every experiment and write tables and figures to an output folder.

Usage:
    python scripts/run_all.py --out results [--data DIR] [--merged heart.csv] [--quick]
"""

import argparse
from pathlib import Path

import pandas as pd

from heartv2.data import (FEATURES_ALL, FEATURES_CORE, drop_duplicates, load_kaggle_merged,
                          load_uci, site_summary)
from heartv2.evaluate import (internal_cv, leakage_audit, leave_one_site_out, loso_table,
                              subgroup_auroc, summarise_cv)
from heartv2.models import get_models
from heartv2 import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default=None, help="folder with processed.*.data (default: download)")
    ap.add_argument("--merged", default=None, help="path to the 1,025-row heart.csv for the audit")
    ap.add_argument("--quick", action="store_true", help="fewer repeats/bootstraps for a smoke run")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    reps, boots = (1, 100) if a.quick else (5, 1000)

    df, n_dup = drop_duplicates(load_uci(a.data))
    print(f"Rows: {len(df)} (removed {n_dup} exact duplicates)")
    site_summary(df).to_csv(out / "site_summary.csv")

    merged = load_kaggle_merged(a.merged)
    if merged is not None:
        audit = leakage_audit(merged, n_repeats=5 if a.quick else 20, seed=a.seed)
        audit.to_csv(out / "leakage_audit.csv")
        print(audit)

    models = get_models(a.seed)
    folds, oof = internal_cv(df, models, FEATURES_ALL, n_repeats=reps, seed=a.seed)
    folds.to_csv(out / "internal_cv_folds.csv", index=False)
    summarise_cv(folds).to_csv(out / "internal_cv_summary.csv")

    loso = []
    for name, feats in {"all": FEATURES_ALL, "core": FEATURES_CORE}.items():
        res, preds = leave_one_site_out(df, models, feats, n_boot=boots, seed=a.seed)
        loso.append(res.assign(features=name))
        loso_table(res).to_csv(out / f"loso_auroc_{name}.csv")
        if name == "all":
            best = res.groupby("model")["auroc"].mean().idxmax()
            plots.internal_vs_external(folds, res, out / "figures/internal_vs_external.png")
            plots.calibration_curves(df, preds, best, path=out / "figures/calibration_loso.png")
    pd.concat(loso).to_csv(out / "loso_all_metrics.csv", index=False)

    subgroup_auroc(df, oof[best], n_boot=boots, seed=a.seed).to_csv(out / "subgroups.csv", index=False)
    tree = next((m for m in ("XGBoost", "LightGBM", "RandomForest") if m in models), None)
    plots.shap_summary(df, models[tree], FEATURES_ALL, path=out / "figures/shap_summary.png")
    print(f"Done. Best external model: {best}. Outputs in {out}/")


if __name__ == "__main__":
    main()
