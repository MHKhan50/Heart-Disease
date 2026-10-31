"""Build the LaTeX tables in paper/tables/ from the CSV files in results/."""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
T = Path(__file__).resolve().parent / "tables"
T.mkdir(exist_ok=True)
NAMES = {"LogisticRegression": "Logistic regression", "RandomForest": "Random forest"}
SITES = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]


def name(m):
    return NAMES.get(m, m)


def write(fname, lines):
    (T / fname).write_text("\n".join(lines) + "\n")


# Leakage audit
a = pd.read_csv(R / "leakage_audit.csv")
rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Data & Rows & Test rows seen in training & Accuracy & AUROC \\", r"\midrule"]
for _, r in a.iterrows():
    label = "As shared (with duplicates)" if "with" in r["data"] else "Duplicates removed"
    rows.append(f"{label} & {r.n_rows:,} & {100*r.test_rows_seen_in_train:.1f}\\% & "
                f"{r.accuracy_mean:.3f} $\\pm$ {r.accuracy_sd:.3f} & {r.auroc_mean:.3f} $\\pm$ {r.auroc_sd:.3f} \\\\")
write("leakage.tex", rows + [r"\bottomrule", r"\end{tabular}"])

# Internal CV
f = pd.read_csv(R / "internal_cv_folds.csv")
g = f.groupby("model")
cols = ["auroc", "auprc", "brier", "ece", "sensitivity", "specificity"]
mean, sd = g[cols].mean(), g[cols].std()
order = mean["auroc"].sort_values(ascending=False).index
rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Model & AUROC & AUPRC & Brier & ECE & Sensitivity & Specificity \\", r"\midrule"]
for m in order:
    cells = [f"{mean.loc[m,c]:.3f} $\\pm$ {sd.loc[m,c]:.3f}" for c in cols]
    rows.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
write("internal_cv.tex", rows + [r"\bottomrule", r"\end{tabular}"])

# Leave-one-site-out AUROC (both feature sets)
L = pd.read_csv(R / "loso_all_metrics.csv")
for fs in ["all", "core"]:
    d = L[L.features == fs]
    order = d.groupby("model")["auroc"].mean().sort_values(ascending=False).index
    rows = [r"\begin{tabular}{l" + "c" * len(SITES) + "}", r"\toprule",
            "Model & " + " & ".join(SITES) + r" \\", r"\midrule"]
    for m in order:
        cells = []
        for s in SITES:
            r = d[(d.model == m) & (d.held_out == s)].iloc[0]
            cells.append(f"{r.auroc:.3f} [{r.auroc_lo:.2f}, {r.auroc_hi:.2f}]")
        rows.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
    write(f"loso_{fs}.tex", rows + [r"\bottomrule", r"\end{tabular}"])

# Calibration on held-out sites (all features)
d = L[L.features == "all"]
order = d.groupby("model")["auroc"].mean().sort_values(ascending=False).index
rows = [r"\begin{tabular}{l" + "c" * len(SITES) + "}", r"\toprule",
        "Model & " + " & ".join(SITES) + r" \\", r"\midrule"]
for m in order:
    cells = []
    for s in SITES:
        r = d[(d.model == m) & (d.held_out == s)].iloc[0]
        cells.append(f"{r.obs_exp:.2f} / {r.cal_intercept:+.2f} / {r.cal_slope:.2f}")
    rows.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
write("calibration.tex", rows + [r"\bottomrule", r"\end{tabular}"])

# Sites
first = d.groupby("held_out")[["n", "prevalence"]].first().loc[SITES]
rows = [r"\begin{tabular}{lrr}", r"\toprule", r"Hospital & Patients & Disease prevalence \\", r"\midrule"]
for s, r in first.iterrows():
    rows.append(f"{s} & {int(r.n)} & {100*r.prevalence:.1f}\\% \\\\")
rows.append(f"\\midrule Total & {int(first.n.sum())} & {100*(first.n*first.prevalence).sum()/first.n.sum():.1f}\\% \\\\")
write("sites.tex", rows + [r"\bottomrule", r"\end{tabular}"])

# Subgroups
s = pd.read_csv(R / "subgroups.csv")
rows = [r"\begin{tabular}{lrrc}", r"\toprule", r"Group & $n$ & Prevalence & AUROC [95\% CI] \\", r"\midrule"]
for _, r in s.iterrows():
    grp = {"sex=female": "Women", "sex=male": "Men", "age<50": "Age $<$ 50",
           "age 50-59": "Age 50--59", "age>=60": "Age $\\geq$ 60"}.get(r.group, r.group)
    rows.append(f"{grp} & {r.n} & {100*r.prevalence:.1f}\\% & {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] \\\\")
write("subgroups.tex", rows + [r"\bottomrule", r"\end{tabular}"])
print("tables written:", sorted(p.name for p in T.glob("*.tex")))
