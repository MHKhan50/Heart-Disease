# Heart Disease Prediction V2: Leakage Audit and Cross-Hospital Validation

**Status:** in progress

A rebuild of my 2022 heart disease project (V1), which compared Random
Forest, XGBoost and a neural network on a merged heart disease dataset and
reported accuracy only. V2 asks a stricter question: how well do these models
hold up when the evaluation is done properly, and when a model trained at some
hospitals is tested at a hospital it has never seen?

## Why V2

V1 had three weaknesses that are common in published work on this data:

1. **Duplicate rows.** Widely shared merged versions of this dataset contain
   many repeated records. When copies land in both the training and test
   split, scores are inflated.
2. **Accuracy only.** Clinical use needs discrimination (AUROC), sensitivity,
   calibration and confidence intervals.
3. **No external validation.** All data was pooled and split at random, so the
   results say nothing about a new hospital.

## Data

UCI Heart Disease databases from four sites: Cleveland Clinic Foundation,
Hungarian Institute of Cardiology (Budapest), V.A. Medical Center (Long Beach)
and University Hospital (Zurich). The 14 standard attributes are used, with
the source hospital kept on every row. See [`data/README.md`](data/README.md).

## Planned experiments

| # | Experiment | Question |
|---|---|---|
| 1 | Leakage audit | How much do duplicate rows inflate reported scores? |
| 2 | Leave-one-hospital-out | Train on three hospitals, test on the fourth |
| 3 | Model comparison | Logistic regression, Random Forest, XGBoost, LightGBM, CatBoost, TabPFN v2 |
| 4 | Calibration | Can the predicted probabilities be trusted? |
| 5 | Explainability and subgroups | SHAP feature effects; performance by sex and age band |

Missing values (notably `ca` and `thal` at some sites) are imputed inside the
cross-validation folds, with missing-indicator features, rather than dropped.

## Run it

**On Kaggle (recommended):** open
[`notebooks/heart_disease_v2.ipynb`](notebooks/heart_disease_v2.ipynb) as a
Kaggle notebook, set Accelerator = None and Internet = On, add the dataset
`johnsmith88/heart-disease-dataset` (used only for the leakage audit), then
*Save & Run All*. CPU is enough.

**Locally:**

```bash
pip install -e ".[models,dev]"
python scripts/run_all.py --out results            # downloads the UCI files
python scripts/run_all.py --out results --quick    # fast smoke run
pytest                                             # offline tests on synthetic data
```

Outputs: `site_summary.csv`, `leakage_audit.csv`, `internal_cv_summary.csv`,
`loso_auroc_all.csv`, `loso_auroc_core.csv`, `loso_all_metrics.csv`,
`subgroups.csv` and figures (internal vs external AUROC, calibration on
held-out hospitals, SHAP).

## Repository structure

```
src/heartv2/   package: data loading, models, evaluation, plots
notebooks/     Kaggle notebook
scripts/       run_all.py (all experiments from the command line)
tests/         offline tests using synthetic data in the UCI file format
results/       tables and figures
data/          data instructions (data itself is not committed)
```

## Disclaimer

Research code only. Not a medical device and not for diagnosis.

## License

[MIT](LICENSE) © 2026 Muzammal Hussain
