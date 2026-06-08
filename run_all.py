"""
run_all.py  --  one-command pipeline for the Forest Cover Type project.

What it does, in order:
  1. Loads the Covertype dataset (581,012 instances) via scikit-learn.
  2. Exploratory data analysis  -> figures/ and results/eda_summary.txt
  3. Feature engineering (see features.py)
  4. Trains 3 baselines (Logistic Regression, Gaussian Naive Bayes, Decision Tree)
     and the proposed model (Histogram Gradient-Boosted Trees, imbalance-aware).
  5. Evaluation: accuracy, macro-F1, weighted-F1, per-class report, confusion matrix.
  6. McNemar significance test (proposed vs. best baseline).
  7. Ablation study (with vs. without engineered features).
  8. Hyper-parameter sensitivity sweep -> figure.
  9. Saves the trained model to model.joblib (used by the Streamlit demo).
 10. Writes results/metrics.json and prints a table you can paste into the report.

Usage:
  python run_all.py                # full run on real data (~3-6 min)
  python run_all.py --subsample 150000   # cap TRAIN rows for a faster run
  python run_all.py --smoke        # tiny synthetic run to test the code offline
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

import features as F

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_real():
    from sklearn.datasets import fetch_covtype
    ds = fetch_covtype(as_frame=False)            # 581012 x 54, target 1..7
    X = pd.DataFrame(ds.data, columns=F.RAW_COLUMNS)
    y = pd.Series(ds.target, name="Cover_Type")
    return X, y


def load_smoke(n=6000, seed=0):
    """Tiny synthetic stand-in with the real 54-column schema (for offline tests)."""
    rng = np.random.default_rng(seed)
    cont = pd.DataFrame({
        "Elevation": rng.normal(2960, 380, n),
        "Aspect": rng.uniform(0, 360, n),
        "Slope": rng.uniform(0, 60, n),
        "Horizontal_Distance_To_Hydrology": rng.uniform(0, 1400, n),
        "Vertical_Distance_To_Hydrology": rng.normal(46, 58, n),
        "Horizontal_Distance_To_Roadways": rng.uniform(0, 7000, n),
        "Hillshade_9am": rng.uniform(50, 254, n),
        "Hillshade_Noon": rng.uniform(120, 254, n),
        "Hillshade_3pm": rng.uniform(0, 254, n),
        "Horizontal_Distance_To_Fire_Points": rng.uniform(0, 7000, n),
    })
    wild = pd.DataFrame(0, index=range(n), columns=F.WILDERNESS)
    wsel = rng.integers(0, 4, n)
    for k, c in enumerate(F.WILDERNESS):
        wild.loc[wsel == k, c] = 1
    soil = pd.DataFrame(0, index=range(n), columns=F.SOIL)
    ssel = rng.integers(0, 40, n)
    for k, c in enumerate(F.SOIL):
        soil.loc[ssel == k, c] = 1
    X = pd.concat([cont, wild, soil], axis=1)[F.RAW_COLUMNS]
    # learnable, imbalanced target tied to elevation
    score = (cont["Elevation"].values - 2960) / 280 + rng.normal(0, 1, n)
    y = np.clip(np.round(score * 1.1 + 4).astype(int), 1, 7)
    y[:14] = np.tile(np.arange(1, 8), 2)  # guarantee all 7 classes present
    return X, pd.Series(y, name="Cover_Type")


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def run_eda(X, y):
    log("Exploratory data analysis ...")
    lines = []
    lines.append(f"Instances: {len(X):,}    Features: {X.shape[1]}    Classes: {y.nunique()}")
    counts = y.value_counts().sort_index()
    lines.append("\nClass distribution:")
    for k, v in counts.items():
        lines.append(f"  {k} {F.COVER_TYPES[k]:<20s} {v:>8,}  ({v/len(y):6.2%})")
    lines.append(f"\nImbalance ratio (largest/smallest): {counts.max()/counts.min():.1f} : 1")
    lines.append("\nContinuous feature summary:")
    lines.append(X[F.CONTINUOUS].describe().T.to_string())
    (RES / "eda_summary.txt").write_text("\n".join(str(l) for l in lines))
    print("\n".join(str(l) for l in lines[:14]))

    # Figure 1: class distribution
    plt.figure(figsize=(6.2, 3.4))
    ax = sns.barplot(x=[F.COVER_TYPES[k] for k in counts.index], y=counts.values,
                     color="#3b6ea5")
    ax.set_ylabel("count"); ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right"); plt.title("Cover-type class distribution")
    plt.tight_layout(); plt.savefig(FIG / "class_distribution.png", dpi=150); plt.close()

    # Figure 2: correlation of continuous features
    plt.figure(figsize=(6.2, 5.0))
    corr = X[F.CONTINUOUS].corr()
    sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": .7},
                xticklabels=True, yticklabels=True)
    plt.title("Correlation among continuous features")
    plt.tight_layout(); plt.savefig(FIG / "feature_correlation.png", dpi=150); plt.close()

    # Figure 3: Elevation distribution by class (the strongest single signal)
    plt.figure(figsize=(6.2, 3.4))
    dfp = pd.DataFrame({"Elevation": X["Elevation"].values,
                        "Cover": [F.COVER_TYPES[v] for v in y.values]})
    order = [F.COVER_TYPES[k] for k in counts.index]
    sns.boxplot(data=dfp, x="Cover", y="Elevation", order=order, color="#7aa6c2", fliersize=0)
    plt.xticks(rotation=30, ha="right"); plt.xlabel("")
    plt.title("Elevation separates cover types")
    plt.tight_layout(); plt.savefig(FIG / "elevation_by_class.png", dpi=150); plt.close()
    log("EDA figures saved to figures/")
    return {"n": int(len(X)), "n_features": int(X.shape[1]),
            "class_counts": {int(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": float(counts.max() / counts.min())}


# --------------------------------------------------------------------------- #
# Modelling helpers
# --------------------------------------------------------------------------- #
def evaluate(name, model, Xte, yte, store):
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    f1w = f1_score(yte, pred, average="weighted")
    store[name] = {"accuracy": round(float(acc), 4),
                   "macro_f1": round(float(f1m), 4),
                   "weighted_f1": round(float(f1w), 4)}
    log(f"  {name:<22s} acc={acc:.4f}  macroF1={f1m:.4f}  weightedF1={f1w:.4f}")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run (offline)")
    ap.add_argument("--subsample", type=int, default=0, help="cap number of TRAIN rows")
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        log("SMOKE MODE: synthetic data, tiny models (code test only).")
        X, y = load_smoke()
        max_iter_proposed, sweep, sweep_sub = 40, [20, 40, 80], 4000
    else:
        log("Loading Covertype via scikit-learn (first run downloads ~11 MB) ...")
        try:
            X, y = load_real()
        except Exception as e:
            log(f"Download failed ({e}).")
            log("If you have no internet here, test the code with:  python run_all.py --smoke")
            raise
        max_iter_proposed, sweep, sweep_sub = 300, [50, 100, 200, 400], 120000
    log(f"Loaded {len(X):,} rows x {X.shape[1]} cols.")

    eda = run_eda(X, y)

    # Feature engineering (everything trains on this matrix)
    Xeng = F.add_engineered_features(X)
    feat_cols = Xeng.columns.tolist()

    Xtr, Xte, ytr, yte = train_test_split(
        Xeng, y, test_size=0.30, random_state=SEED, stratify=y)
    if args.subsample and args.subsample < len(Xtr):
        Xtr_b, _, ytr_b, _ = train_test_split(
            Xtr, ytr, train_size=args.subsample, random_state=SEED, stratify=ytr)
    else:
        Xtr_b, ytr_b = Xtr, ytr
    log(f"Train: {len(Xtr_b):,}   Test: {len(Xte):,}")

    metrics = {}

    # ---- Baselines -------------------------------------------------------- #
    log("Training baselines ...")
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=200, n_jobs=-1))])
    lr.fit(Xtr_b, ytr_b); evaluate("LogisticRegression", lr, Xte, yte, metrics)

    nb = GaussianNB()
    nb.fit(Xtr_b, ytr_b); evaluate("GaussianNaiveBayes", nb, Xte, yte, metrics)

    dt = DecisionTreeClassifier(random_state=SEED)
    dt.fit(Xtr_b, ytr_b); evaluate("DecisionTree", dt, Xte, yte, metrics)

    # ---- Proposed model --------------------------------------------------- #
    log("Training proposed model (HistGradientBoosting, imbalance-aware) ...")
    sw = compute_sample_weight("balanced", ytr_b)
    hgb = HistGradientBoostingClassifier(
        max_iter=max_iter_proposed, learning_rate=0.1, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=SEED)
    hgb.fit(Xtr_b, ytr_b, sample_weight=sw)
    pred_hgb = evaluate("ProposedGBDT", hgb, Xte, yte, metrics)

    # Per-class report for the case study
    rep = classification_report(yte, pred_hgb, output_dict=True, zero_division=0)
    per_class = {F.COVER_TYPES[int(k)]: round(rep[k]["f1-score"], 3)
                 for k in rep if k.isdigit()}
    metrics["ProposedGBDT"]["per_class_f1"] = per_class

    # Confusion matrix figure
    labels = sorted(y.unique())
    cm = confusion_matrix(yte, pred_hgb, labels=labels, normalize="true")
    plt.figure(figsize=(5.6, 4.8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=[F.COVER_TYPES[k] for k in labels],
                yticklabels=[F.COVER_TYPES[k] for k in labels])
    plt.ylabel("true"); plt.xlabel("predicted")
    plt.title("Proposed model — row-normalized confusion matrix")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close()

    # ---- Significance: McNemar vs best baseline --------------------------- #
    best_base = max(["LogisticRegression", "GaussianNaiveBayes", "DecisionTree"],
                    key=lambda k: metrics[k]["accuracy"])
    base_model = {"LogisticRegression": lr, "GaussianNaiveBayes": nb, "DecisionTree": dt}[best_base]
    pred_base = base_model.predict(Xte)
    cp = (pred_hgb == yte.values); cb = (pred_base == yte.values)
    b = int(np.sum(cp & ~cb)); c = int(np.sum(~cp & cb))
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p = float(chi2.sf(stat, 1))
    metrics["significance"] = {"best_baseline": best_base, "b_proposed_only": b,
                               "c_baseline_only": c, "mcnemar_chi2": round(stat, 2),
                               "p_value": p}
    log(f"  McNemar vs {best_base}: b={b}, c={c}, chi2={stat:.1f}, p={p:.2e}")

    # ---- Ablation: drop engineered features ------------------------------- #
    log("Ablation: proposed model without engineered features ...")
    raw_only = [c for c in feat_cols if c not in F.ENGINEERED]
    hgb_abl = HistGradientBoostingClassifier(
        max_iter=max_iter_proposed, learning_rate=0.1, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=SEED)
    hgb_abl.fit(Xtr_b[raw_only], ytr_b,
                sample_weight=compute_sample_weight("balanced", ytr_b))
    pa = hgb_abl.predict(Xte[raw_only])
    metrics["ablation_no_engineered"] = {
        "accuracy": round(float(accuracy_score(yte, pa)), 4),
        "macro_f1": round(float(f1_score(yte, pa, average="macro")), 4)}
    log(f"  no-engineered: acc={metrics['ablation_no_engineered']['accuracy']:.4f} "
        f"(full: {metrics['ProposedGBDT']['accuracy']:.4f})")

    # ---- Hyper-parameter sensitivity (max_iter) --------------------------- #
    log("Parameter sensitivity sweep (max_iter) ...")
    if len(Xtr_b) > sweep_sub:
        Xs, _, ys, _ = train_test_split(Xtr_b, ytr_b, train_size=sweep_sub,
                                        random_state=SEED, stratify=ytr_b)
    else:
        Xs, ys = Xtr_b, ytr_b
    sw_s = compute_sample_weight("balanced", ys)
    accs = []
    for m in sweep:
        mdl = HistGradientBoostingClassifier(max_iter=m, learning_rate=0.1,
                                             max_leaf_nodes=63, l2_regularization=1.0,
                                             random_state=SEED)
        mdl.fit(Xs, ys, sample_weight=sw_s)
        accs.append(float(accuracy_score(yte, mdl.predict(Xte))))
    metrics["sensitivity_max_iter"] = {"grid": sweep, "accuracy": [round(a, 4) for a in accs]}
    plt.figure(figsize=(5.4, 3.3))
    plt.plot(sweep, accs, "o-", color="#b5482f")
    plt.xlabel("max_iter (number of boosting rounds)"); plt.ylabel("test accuracy")
    plt.title("Sensitivity to boosting rounds")
    plt.tight_layout(); plt.savefig(FIG / "param_sensitivity.png", dpi=150); plt.close()

    # ---- Save model bundle + metrics -------------------------------------- #
    joblib.dump({"model": hgb, "feature_columns": feat_cols,
                 "cover_types": F.COVER_TYPES}, ROOT / "model.joblib")
    metrics["dataset"] = eda
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Pretty print + paste-ready block --------------------------------- #
    print("\n" + "=" * 64)
    print("RESULTS  (paste these numbers into Table 2 / Table 3 of report.tex)")
    print("=" * 64)
    print(f"{'Model':<24s}{'Accuracy':>10s}{'Macro-F1':>10s}{'Wt-F1':>10s}")
    for k in ["LogisticRegression", "GaussianNaiveBayes", "DecisionTree", "ProposedGBDT"]:
        m = metrics[k]
        print(f"{k:<24s}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['weighted_f1']:>10.4f}")
    print("-" * 64)
    print(f"Ablation (no engineered feats): acc={metrics['ablation_no_engineered']['accuracy']:.4f}"
          f"  macroF1={metrics['ablation_no_engineered']['macro_f1']:.4f}")
    print(f"McNemar vs {metrics['significance']['best_baseline']}: "
          f"chi2={metrics['significance']['mcnemar_chi2']}, p={metrics['significance']['p_value']:.2e}")
    print(f"Per-class F1 (proposed): {metrics['ProposedGBDT']['per_class_f1']}")
    print("=" * 64)
    print(f"Done in {metrics['runtime_seconds']}s. Saved: model.joblib, figures/, results/metrics.json")
    print("Next:  streamlit run app.py")


if __name__ == "__main__":
    main()
