#!/usr/bin/env python3
"""
train_models.py  -  ML layer for the Methanol Reactor Digital Twin
===================================================================
Reads reactor_data.csv (from data_generator.py) and:

  1. EDA            : shape, stats, missing values, correlations
  2. Conversion &
     Yield models   : XGBoost regressors (falls back to sklearn
                      HistGradientBoosting if xgboost is not installed)
  3. RUL model      : regressor on engineered, CAUSAL features
  4. Anomaly model  : Isolation Forest on deviation features
  5. Evaluation     : MSE / RMSE / MAE / R2, classification report, ROC-AUC
  6. Saves          : models/*.joblib, models/metrics.json, model_report.html

Run:  python train_models.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DATA_FILE = "reactor_data.csv"
MODEL_DIR = Path("models")
REPORT_HTML = "model_report.html"
SEED = 42
TEST_SIZE = 0.20                 # fraction of catalyst RUNS held out for testing

# Feature-engineering windows (in HOURS; converted to rows using the sampling step)
BASELINE_H = 96.0                # "recent normal behaviour" window for anomaly features
TREND_H = 80.0                   # smoothing window for the RUL trend features

# Aging index: equivalent hours at a reference temperature (Arrhenius-weighted).
# ED/R matches the sintering energy used in data_generator.py. In a real project
# you would estimate this from lab aging studies.
R_GAS = 8.314
ED_J_MOL = 60_000.0
T_REF_K = 523.15

NEAR_EOL_H = 1000.0              # "near end-of-life" = RUL below this (hours)

# Canonical internal column names (real CSV columns are renamed to these)
TIME, TEMP, PRES, RATIO = "time", "temp", "pres", "ratio"
CONV, YLD, ACT, RUL, ANOM, RUN = "conv", "yld", "act", "rul", "anom", "run"

# Keywords for auto-detecting columns (matched on lowercase, letters+digits only)
KEYWORDS = {
    TIME: ["timeonstream", "time", "hours"],
    TEMP: ["temperature", "temp"],
    PRES: ["pressure", "press"],
    RATIO: ["ratio", "feed"],
    CONV: ["conversion", "conv"],
    YLD: ["yield"],
    ACT: ["activity", "health"],
    RUL: ["rul", "remainingusefullife"],
    ANOM: ["anomaly", "poison", "fault"],
    RUN: ["runid", "run", "batch", "lifecycle"],
}
REQUIRED = [TIME, TEMP, PRES, RATIO, CONV, YLD, ACT, RUL, ANOM]


# =============================================================================
# 1. SMALL HELPERS
# =============================================================================
def banner(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def norm(name):
    """'Temperature_K ' -> 'temperaturek'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(columns):
    """Map canonical names to the real CSV column names by keyword."""
    mapping, used = {}, set()
    for key, kws in KEYWORDS.items():
        for kw in kws:
            hit = next((c for c in columns if c not in used and kw in norm(c)), None)
            if hit is not None:
                mapping[key] = hit
                used.add(hit)
                break
    return mapping


def make_regressor():
    """XGBoost if available, otherwise sklearn's HistGradientBoosting."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            n_jobs=-1, random_state=SEED, verbosity=0), "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print("(xgboost not installed -> using sklearn HistGradientBoostingRegressor)")
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             max_depth=6, random_state=SEED), "HistGradientBoosting"


def reg_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def print_metrics(label, m, unit=""):
    print(f"  {label:<34} MSE={m['MSE']:>10.3f}  RMSE={m['RMSE']:>8.3f}{unit}  "
          f"MAE={m['MAE']:>8.3f}{unit}  R2={m['R2']:>6.3f}")


def show_importance(model, X, y, title, n=2000):
    """Permutation importance: how much does the error grow if a feature is shuffled?"""
    Xs = X.sample(min(n, len(X)), random_state=SEED)
    r = permutation_importance(model, Xs, y.loc[Xs.index], n_repeats=3, random_state=SEED,
                               scoring="neg_mean_squared_error", n_jobs=1)
    imp = pd.Series(np.clip(r.importances_mean, 0, None), index=X.columns)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False) if imp.sum() > 0 else imp
    print(f"  Feature importance - {title} (% of total):")
    for name, val in imp.head(6).items():
        print(f"      {name:<16} {val:5.1f} %  " + "#" * int(val / 3))


# =============================================================================
# 2. LOAD DATA
# =============================================================================
banner("STEP 1 | LOAD DATA")
path = next((p for p in [Path(__file__).resolve().parent / DATA_FILE, Path.cwd() / DATA_FILE]
             if p.exists()), None)
if path is None:
    sys.exit(f"ERROR: {DATA_FILE} not found. Run `python data_generator.py` first.")

raw = pd.read_csv(path)
raw.columns = [str(c).strip() for c in raw.columns]
cmap = detect_columns(list(raw.columns))
missing = [k for k in REQUIRED if k not in cmap]
if missing:
    sys.exit(f"ERROR: could not detect columns for {missing}. Found: {list(raw.columns)}")

df = raw.rename(columns={v: k for k, v in cmap.items()})
keep = [c for c in KEYWORDS if c in df.columns]
df = df[keep].copy()
for c in keep:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=REQUIRED).reset_index(drop=True)

# If there is no run column, rebuild one: a new run starts whenever time resets.
if RUN not in df.columns:
    df[RUN] = (df[TIME].diff() < 0).cumsum()
    print("(no run column found -> inferred run ids from time resets)")

df[ANOM] = (df[ANOM] > 0).astype(int)
df = df.sort_values([RUN, TIME]).reset_index(drop=True)

dt_h = float(df.groupby(RUN)[TIME].diff().median())
if not np.isfinite(dt_h) or dt_h <= 0:
    dt_h = 8.0
print(f"Loaded {path.name}: {len(df):,} rows | {df[RUN].nunique()} runs | sampling step = {dt_h:g} h")
print("Detected columns:", {k: v for k, v in cmap.items()})

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
banner("STEP 2 | EXPLORATORY DATA ANALYSIS")
print("\n-- Basic statistics --")
print(df.drop(columns=[RUN]).describe().T.round(3))
print(f"\n-- Missing values after cleaning: {int(df.isna().sum().sum())}")

per_run = df.groupby(RUN)[TIME].max()
print(f"\n-- Run length (h): min={per_run.min():.0f}  mean={per_run.mean():.0f}  max={per_run.max():.0f}")
print(f"-- Anomaly rows: {df[ANOM].sum():,} ({100 * df[ANOM].mean():.2f} %)  <- class imbalance!")

print("\n-- Correlation matrix (Pearson) --")
corr = df.drop(columns=[RUN]).corr()
print(corr.round(2))
for tgt in (CONV, YLD, RUL):
    print(f"\n-- Correlation with {tgt} (sorted by |r|) --")
    print(corr[tgt].drop(tgt).sort_values(key=np.abs, ascending=False).round(3).to_string())

print("\n-- Mean conversion by temperature band (rough 'volcano' check) --")
band = pd.cut(df[TEMP], bins=6)
print(df.groupby(band, observed=True)[CONV].agg(["count", "mean"]).round(2))

# =============================================================================
# 4. FEATURE ENGINEERING (per run, CAUSAL = uses only past and present rows)
# =============================================================================
banner("STEP 3 | FEATURE ENGINEERING")
base_rows = max(4, int(round(BASELINE_H / dt_h)))
trend_rows = max(3, int(round(TREND_H / dt_h)))
g = df.groupby(RUN)

# ---- Anomaly (deviation) features: "how far is NOW from RECENT NORMAL?" -----
def dev_from_baseline(col):
    """Current value minus the median of the previous `base_rows` rows (shifted so the
    current row is NOT part of its own baseline)."""
    base = g[col].transform(lambda s: s.rolling(base_rows, min_periods=3).median().shift(1))
    return (df[col] - base).fillna(0.0)

df["f_conv_dev"] = dev_from_baseline(CONV)           # sudden conversion drop
df["f_yld_dev"] = dev_from_baseline(YLD)
df["f_temp_dev"] = dev_from_baseline(TEMP)           # lets the model tell process drift from poisoning
df["f_pres_dev"] = dev_from_baseline(PRES)
df["f_conv_diff"] = g[CONV].diff().fillna(0.0)       # step-to-step change
ANOM_FEATS = ["f_conv_dev", "f_yld_dev", "f_temp_dev", "f_pres_dev", "f_conv_diff"]

# ---- RUL (trend) features ---------------------------------------------------
df["f_conv_roll"] = g[CONV].transform(lambda s: s.rolling(trend_rows, min_periods=1).mean())
df["f_yld_roll"] = g[YLD].transform(lambda s: s.rolling(trend_rows, min_periods=1).mean())
# conversion relative to the best level seen so far in this run (1.0 = like new)
df["f_conv_rel"] = df["f_conv_roll"] / df.groupby(RUN)["f_conv_roll"].cummax().clip(lower=1e-6)
# Arrhenius-weighted cumulative thermal exposure = "equivalent hours at 250 degC"
aging_rate = np.exp(-(ED_J_MOL / R_GAS) * (1.0 / df[TEMP] - 1.0 / T_REF_K))
df["f_aging_index"] = (aging_rate * dt_h).groupby(df[RUN]).cumsum()
df["f_temp_mean"] = g[TEMP].transform(lambda s: s.expanding().mean())
RUL_FEATS = [TIME, TEMP, PRES, RATIO, "f_conv_roll", "f_yld_roll",
             "f_conv_rel", "f_aging_index", "f_temp_mean"]

FEATS_A = [TIME, TEMP, PRES, RATIO]       # operating conditions only
FEATS_B = FEATS_A + [ACT]                 # + catalyst health state (for what-if twin)
print(f"Windows: baseline = {base_rows} rows ({base_rows * dt_h:g} h), trend = {trend_rows} rows")
print("Performance features A :", FEATS_A)
print("Performance features B :", FEATS_B)
print("RUL features           :", RUL_FEATS)
print("Anomaly features       :", ANOM_FEATS)

# =============================================================================
# 5. TRAIN / TEST SPLIT BY CATALYST RUN (no leakage between rows of one run)
# =============================================================================
banner("STEP 4 | SPLIT BY RUN")
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, te_idx = next(splitter.split(df, groups=df[RUN]))
train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
assert set(train[RUN]).isdisjoint(set(test[RUN])), "Run leakage between train and test!"
print(f"Train: {train[RUN].nunique()} runs / {len(train):,} rows | "
      f"Test: {test[RUN].nunique()} runs / {len(test):,} rows (whole runs held out)")

all_metrics, bundles = {}, {}

# =============================================================================
# 6. CONVERSION & YIELD MODELS
# =============================================================================
from physics_features import add_physics_features, PHYS_COLS

banner("STEP 5 | CONVERSION & YIELD REGRESSION (physics-informed features)")
print("A = conditions only | B = conditions + catalyst health | nophys = baseline without physics features\n")

# (name, base features, uses measured activity?, add physics features?)
VARIANTS = [
    ("A_baseline_nophys", FEATS_A, False, False),
    ("A_conditions_only", FEATS_A, False, True),
    ("B_baseline_nophys", FEATS_B, True,  False),
    ("B_with_health",     FEATS_B, True,  True),
]
X_test_store = {}
for tname, tcol in (("conversion", CONV), ("yield", YLD)):
    for vname, base_feats, use_act, use_phys in VARIANTS:
        if use_phys:
            cols = base_feats + PHYS_COLS
            Xtr = add_physics_features(train, use_act)[cols]
            Xte = add_physics_features(test, use_act)[cols]
        else:
            Xtr, Xte = train[base_feats], test[base_feats]
        model, algo = make_regressor()
        model.fit(Xtr, train[tcol])
        pred = np.clip(model.predict(Xte), 0, 100)
        key = f"{tname}_{vname}"
        all_metrics[key] = reg_metrics(test[tcol], pred)
        print_metrics(f"{tname} [{vname}]", all_metrics[key], " pp")
        bundles[key] = {"model": model, "features": list(Xtr.columns), "algo": algo,
                        "use_activity": use_act, "uses_physics": use_phys}
        X_test_store[key] = Xte
        if key == "conversion_B_with_health":
            test["pred_conv"] = pred
    print()
show_importance(bundles["conversion_B_with_health"]["model"],
                X_test_store["conversion_B_with_health"], test[CONV], "conversion, B + physics")
print("\n  (pp = percentage points. MSE is in pp^2, so RMSE is the easiest to read.)")

# =============================================================================
# 7. REMAINING USEFUL LIFE (RUL) MODEL
# =============================================================================
banner("STEP 6 | RUL REGRESSION")
rul_model, algo = make_regressor()
rul_model.fit(train[RUL_FEATS], train[RUL])
rul_pred = np.clip(rul_model.predict(test[RUL_FEATS]), 0, None)
test["pred_rul"] = rul_pred

all_metrics["rul_all"] = reg_metrics(test[RUL], rul_pred)
print_metrics("RUL (all test rows)", all_metrics["rul_all"], " h")

near = test[RUL] < NEAR_EOL_H
if near.sum() > 10:
    all_metrics["rul_near_eol"] = reg_metrics(test.loc[near, RUL], rul_pred[near.values])
    print_metrics(f"RUL (true RUL < {NEAR_EOL_H:g} h)", all_metrics["rul_near_eol"], " h")

naive_mae = mean_absolute_error(test[RUL], np.full(len(test), train[RUL].mean()))
print(f"\n  Naive baseline (always predict mean RUL): MAE = {naive_mae:.0f} h")
print(f"  Model MAE = {all_metrics['rul_all']['MAE']:.0f} h  -> "
      f"{100 * (1 - all_metrics['rul_all']['MAE'] / naive_mae):.0f} % better than the baseline")
show_importance(rul_model, test[RUL_FEATS], test[RUL], "RUL")
print("\n  NOTE: RUL depends on FUTURE poisoning events nobody can see, so a perfect "
      "score is impossible by design.")
bundles["rul"] = {"model": rul_model, "features": RUL_FEATS, "algo": algo}

# =============================================================================
# 8. ISOLATION FOREST ANOMALY DETECTOR
# =============================================================================
banner("STEP 7 | ISOLATION FOREST (poisoning detection)")
# Labels are NOT used for training. We only use the training anomaly RATE to tell
# the forest how many points to flag (the 'contamination' parameter).
contamination = float(np.clip(train[ANOM].mean(), 0.005, 0.20))
iso = make_pipeline(StandardScaler(),
                    IsolationForest(n_estimators=300, contamination=contamination,
                                    random_state=SEED, n_jobs=-1))
iso.fit(train[ANOM_FEATS])

y_true = test[ANOM].values
y_pred = (iso.predict(test[ANOM_FEATS]) == -1).astype(int)       # sklearn: -1 = outlier
score = -iso.decision_function(test[ANOM_FEATS])                 # higher = more anomalous
test["anom_score"], test["anom_pred"] = score, y_pred

print(f"Contamination used: {contamination:.3%}\n")
print(classification_report(y_true, y_pred, target_names=["normal", "poisoning"], digits=3))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(cm, index=["true normal", "true poison"], columns=["pred normal", "pred poison"]))
if len(np.unique(y_true)) == 2:
    roc, ap = roc_auc_score(y_true, score), average_precision_score(y_true, score)
    print(f"\nROC-AUC = {roc:.3f} | Average precision (PR-AUC) = {ap:.3f} "
          f"(random guessing would give {y_true.mean():.3f})")
    all_metrics["anomaly"] = {"roc_auc": float(roc), "avg_precision": float(ap)}
print("\n  NOTE: the 48 h anomaly label includes rows where the temporary dip has "
      "mostly recovered,\n  so perfect recall is unrealistic. Reading precision vs recall "
      "is a good viva discussion.")
bundles["anomaly"] = {"model": iso, "features": ANOM_FEATS, "contamination": contamination}

# =============================================================================
# 9. SAVE MODELS + METRICS (for the Streamlit what-if tab later)
# =============================================================================
banner("STEP 8 | SAVE ARTIFACTS")
MODEL_DIR.mkdir(exist_ok=True)
meta = {"dt_h": dt_h, "base_rows": base_rows, "trend_rows": trend_rows,
        "ED_J_MOL": ED_J_MOL, "T_REF_K": T_REF_K, "column_map": cmap}
for name, b in bundles.items():
    b["meta"] = meta
    joblib.dump(b, MODEL_DIR / f"{name}.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
print(f"Saved {len(bundles)} models + metrics.json to ./{MODEL_DIR}/")

# =============================================================================
# 10. INTERACTIVE VALIDATION REPORT (Plotly -> HTML)
# =============================================================================
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rid = int(test.groupby(RUN)[ANOM].sum().idxmax())     # test run with the most poisoning
    one = test[test[RUN] == rid].sort_values(TIME)
    s1 = test.sample(min(3000, len(test)), random_state=SEED)

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=(
        "Conversion: predicted vs actual (%)", "RUL: predicted vs actual (h)",
        f"Anomaly score over time - test run {rid}"))
    fig.add_trace(go.Scattergl(x=s1[CONV], y=s1["pred_conv"], mode="markers",
                               marker=dict(size=4, opacity=0.5, color="#22d3ee"), name="Conversion"), 1, 1)
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="white"),
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scattergl(x=s1[RUL], y=s1["pred_rul"], mode="markers",
                               marker=dict(size=4, opacity=0.5, color="#c084fc"), name="RUL"), 1, 2)
    top = float(max(s1[RUL].max(), s1["pred_rul"].max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color="white"),
                             showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=one[TIME], y=one["anom_score"], mode="lines",
                             line=dict(color="#38bdf8"), name="Anomaly score"), 1, 3)
    hit = one[one[ANOM] == 1]
    fig.add_trace(go.Scatter(x=hit[TIME], y=hit["anom_score"], mode="markers",
                             marker=dict(color="#ef4444", symbol="x", size=9), name="True poisoning"), 1, 3)
    flagged = one[one["anom_pred"] == 1]
    fig.add_trace(go.Scatter(x=flagged[TIME], y=flagged["anom_score"], mode="markers",
                             marker=dict(color="#fbbf24", symbol="circle-open", size=11),
                             name="Flagged by model"), 1, 3)
    fig.update_xaxes(title_text="Actual", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Actual", row=1, col=2)
    fig.update_yaxes(title_text="Predicted", row=1, col=2)
    fig.update_xaxes(title_text="Time on stream (h)", row=1, col=3)
    fig.update_layout(template="plotly_dark", height=520, title="Model validation on held-out catalyst runs")
    fig.write_html(REPORT_HTML, include_plotlyjs=True)
    print(f"Validation report written to ./{REPORT_HTML} (double-click to open)")
except Exception as exc:
    print(f"(Report skipped: {exc})")

# =============================================================================
# 11. FINAL SCOREBOARD
# =============================================================================
banner("SCOREBOARD (held-out catalyst runs)")
for k in ("conversion_A_conditions_only", "conversion_B_with_health",
          "yield_A_conditions_only", "yield_B_with_health"):
    print(f"  {k:<32} R2 = {all_metrics[k]['R2']:.3f} | RMSE = {all_metrics[k]['RMSE']:.2f} pp")
print(f"  {'rul_all':<32} R2 = {all_metrics['rul_all']['R2']:.3f} | MAE  = {all_metrics['rul_all']['MAE']:.0f} h")
if "anomaly" in all_metrics:
    print(f"  {'anomaly (Isolation Forest)':<32} ROC-AUC = {all_metrics['anomaly']['roc_auc']:.3f}")
print("\nDone. Next: load models/*.joblib inside app.py for the What-If simulator.")