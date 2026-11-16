"""
pages/1_What_If_Simulator.py  -  Digital-twin "What-If" page
============================================================
Loads the models trained by train_models.py (./models/*.joblib) and lets you:
  * drag T / P / H2:CO / time-on-stream / catalyst activity and see live predictions
  * view conversion, yield and an ESTIMATED remaining useful life on gauges
  * see a temperature sweep and a Temperature x Pressure conversion map
  * test the Isolation Forest poisoning detector with a simulated conversion drop

Streamlit shows this automatically in the sidebar when you run:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))   # lets pages/ import physics_features.py
from physics_features import add_physics_features

# 0. PAGE SETUP + STYLE

st.set_page_config(page_title="What-If Simulator | Reactor Twin", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {background: linear-gradient(135deg, #0b1f33 0%, #12304d 55%, #0f4c5c 100%);
           border: 1px solid #1f3b57; border-radius: 14px; padding: 20px 28px; margin-bottom: 16px;}
    .hero h1 {margin: 0; font-size: 1.8rem; color: #e8f1f8;}
    .hero p  {margin: 4px 0 0 0; color: #8fb3cf; font-size: 0.95rem;}
    .pill {display: inline-block; padding: 5px 16px; border-radius: 999px;
           font-size: 0.85rem; font-weight: 700; letter-spacing: 0.8px;}
    .pill.ok   {background: #0e3b2a; color: #4ade80; border: 1px solid #1f6b4a;}
    .pill.crit {background: #3b1215; color: #f87171; border: 1px solid #7a2229;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🧪 What-If Simulator</h1>
<p>Change operating conditions and catalyst state, then watch the trained ML models respond in real time.</p>
</div>""", unsafe_allow_html=True)


# 1. SLIDER LIMITS  (keep these inside the range your data covers!)
#    Tree models cannot extrapolate. If you widen the temperature range in
#    data_generator.py and retrain, update T_MIN / T_MAX here too.

T_MIN, T_MAX = 485.0, 590.0        # K
P_MIN, P_MAX = 50.0, 100.0         # bar
R_MIN, R_MAX = 1.6, 3.2            # H2/CO
TOS_MAX = 8000                     # hours
ACT_MIN = 0.20
R_GAS = 8.314

MODEL_NAMES = ["conversion_A_conditions_only", "conversion_B_with_health",
               "yield_A_conditions_only", "yield_B_with_health", "rul", "anomaly"]



# 2. LOAD MODELS (cached, with friendly errors instead of tracebacks)

def find_model_dir():
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        d = base / "models"
        if d.is_dir():
            return d
    return None


@st.cache_resource(show_spinner="Loading trained models...")
def load_bundles(model_dir: str):
    return {n: joblib.load(Path(model_dir) / f"{n}.joblib") for n in MODEL_NAMES}


model_dir = find_model_dir()
if model_dir is None:
    st.error("❌ No `models/` folder found. Run `python train_models.py` in the project folder first.")
    st.stop()
try:
    B = load_bundles(str(model_dir))
except FileNotFoundError as exc:
    st.error(f"❌ A model file is missing: `{exc.filename}`. Re-run `python train_models.py`.")
    st.stop()
except ModuleNotFoundError as exc:
    st.error(f"❌ A library needed to load the models is missing: `{exc.name}`. "
             f"Run `pip install {exc.name}` and refresh.")
    st.stop()
except Exception as exc:
    st.error(f"❌ Could not load models: {exc}")
    st.stop()


# 3. PREDICTION HELPERS

def predict_df(bundle, **cols):
    """Vectorised prediction. Inputs may be scalars or arrays (broadcast together)."""
    arrs = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in cols.values()])
    X = pd.DataFrame({k: a.ravel() for k, a in zip(cols.keys(), arrs)})
    if bundle.get("uses_physics"):                       # build the same physics features as training
        X = add_physics_features(X, bundle["use_activity"])
    X = X[bundle["features"]]
    return np.clip(bundle["model"].predict(X), 0, 100), arrs[0].shape


def estimate_rul(T, P, r, t, conv, yld, conv_fresh):
    """Steady-state RUL estimate: assume the reactor has run at constant conditions for t hours.
    The RUL model needs history-based features, so we reconstruct them under that assumption."""
    meta = B["rul"]["meta"]
    aging = np.exp(-(meta["ED_J_MOL"] / R_GAS) * (1.0 / T - 1.0 / meta["T_REF_K"])) * (t + meta["dt_h"])
    row = {"time": t, "temp": T, "pres": P, "ratio": r,
           "f_conv_roll": conv, "f_yld_roll": yld,
           "f_conv_rel": min(conv / max(conv_fresh, 1e-6), 1.0),
           "f_aging_index": aging, "f_temp_mean": T}
    X = pd.DataFrame([row])[B["rul"]["features"]]
    return float(max(B["rul"]["model"].predict(X)[0], 0.0))


def gauge(value, title, vmax, color, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "valueformat": ".1f"},
        title={"text": title},
        gauge={"axis": {"range": [0, vmax]}, "bar": {"color": color},
               "bgcolor": "rgba(255,255,255,0.05)", "borderwidth": 0}))
    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=25, r=25, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def style(fig, title, xt, yt, height=450):
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top", font=dict(size=16)),
        height=height, font=dict(size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,27,42,0.6)",
        margin=dict(l=70, r=40, t=110, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"))
    ax = dict(gridcolor="#1c2b3d", zeroline=False, automargin=True, title_standoff=14, tickangle=0)
    fig.update_xaxes(title_text=xt, **ax)
    fig.update_yaxes(title_text=yt, **ax)
    return fig


# 4. SIDEBAR CONTROLS

st.sidebar.markdown("## 🎛️ Scenario")
T = st.sidebar.slider("Temperature (K)", T_MIN, T_MAX, 520.0, 0.5)
P = st.sidebar.slider("Pressure (bar)", P_MIN, P_MAX, 77.0, 0.5)
R = st.sidebar.slider("Feed ratio H2/CO", R_MIN, R_MAX, 2.5, 0.05)
TOS = st.sidebar.slider("Time on stream (h)", 0, TOS_MAX, 2000, 50)

st.sidebar.markdown("### Catalyst state")
use_health = st.sidebar.checkbox("Catalyst health is known (model B)", value=True,
                                 help="On: uses activity as an input. Off: conditions-only model A.")
ACT = st.sidebar.slider("Catalyst activity (0-1)", ACT_MIN, 1.0, 0.65, 0.01, disabled=not use_health)
st.sidebar.caption("Time and activity are correlated in real operation. Extreme mismatches "
                   "(e.g. 0 h with activity 0.3) lie outside what the models have seen.")

conv_model = B["conversion_B_with_health"] if use_health else B["conversion_A_conditions_only"]
yld_model = B["yield_B_with_health"] if use_health else B["yield_A_conditions_only"]


# 5. CORE PREDICTIONS

args = dict(time=TOS, temp=T, pres=P, ratio=R, act=ACT)
conv = float(predict_df(conv_model, **args)[0][0])
yld = float(predict_df(yld_model, **args)[0][0])
# Reference: same conditions with a brand-new catalyst
conv_fresh = float(predict_df(B["conversion_B_with_health"], time=0, temp=T, pres=P, ratio=R, act=1.0)[0][0])
rul = estimate_rul(T, P, R, TOS, conv, yld, conv_fresh)

tab1, tab2, tab3 = st.tabs(["⚙️ Performance what-if", "☣️ Poisoning detector", "📊 Model accuracy"])


# TAB 1: PERFORMANCE

with tab1:
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge(conv, "Predicted conversion", 100, "#22d3ee", " %"))
    g2.plotly_chart(gauge(yld, "Predicted yield", 100, "#a3e635", " %"))
    g3.plotly_chart(gauge(rul, "Estimated RUL (steady-state)", 20000, "#c084fc", " h"))

    delta = conv - conv_fresh
    st.caption(f"Model used: **{'B (conditions + catalyst health)' if use_health else 'A (conditions only)'}** | "
               f"vs a fresh catalyst at the same conditions: **{delta:+.1f} pp** conversion. "
               "RUL assumes the reactor has run at these constant conditions since start-up, and it cannot "
               "foresee future poisoning events.")

    left, right = st.columns(2)

    # ---- Temperature sweep ---------------------------------------------------
    T_grid = np.linspace(T_MIN, T_MAX, 71)
    fig = go.Figure()
    if use_health:
        curves = [("Fresh catalyst (1.00)", 1.0, "#4ade80"),
                  (f"Current setting ({ACT:.2f})", ACT, "#22d3ee"),
                  ("Aged catalyst (0.40)", 0.4, "#f87171")]
        for label, a_val, colr in curves:
            y, _ = predict_df(conv_model, time=TOS, temp=T_grid, pres=P, ratio=R, act=a_val)
            fig.add_trace(go.Scatter(x=T_grid, y=y, mode="lines", name=label,
                                     line=dict(color=colr, width=2.6)))
    else:
        y, _ = predict_df(conv_model, time=TOS, temp=T_grid, pres=P, ratio=R, act=ACT)
        fig.add_trace(go.Scatter(x=T_grid, y=y, mode="lines", name="Conditions-only model",
                                 line=dict(color="#22d3ee", width=2.6)))
    fig.add_trace(go.Scatter(x=[T], y=[conv], mode="markers", name="Operating point",
                             marker=dict(size=13, color="white", line=dict(width=2, color="#0b1f33"))))

    y_cur, _ = predict_df(conv_model, time=TOS, temp=T_grid, pres=P, ratio=R, act=ACT)
    T_opt = float(T_grid[int(np.argmax(y_cur))])
    fig.add_vline(
        x=T_opt,
        line_dash="dot",
        line_color="#fbbf24",
        annotation_text=f"Optimal: {T_opt:.0f} K",
        annotation_position="top right",
        annotation_font_color="#fbbf24",
    )
    style(
        fig,
        "Conversion vs temperature (other inputs held fixed)",
        "Temperature (K)",
        "Conversion (%)",
    )
    left.plotly_chart(fig)

    # Clean, dark-theme metric cards for Optimal Temperature
    max_conv = float(np.max(y_cur))
    temp_shift = T_opt - T
    c1, c2 = left.columns(2)
    c1.metric(
        label="Optimal Temperature",
        value=f"{T_opt:.0f} K",
        delta=f"{temp_shift:+.0f} K from current",
        delta_color="off",
    )
    c2.metric(
        label="Peak Conversion at Optimal T",
        value=f"{max_conv:.1f} %",
        delta=f"{max_conv - conv:+.1f} % gain",
    )
    
    # ---- Temperature x Pressure map -------------------------------------------
    Tg, Pg = np.linspace(T_MIN, T_MAX, 36), np.linspace(P_MIN, P_MAX, 26)
    TT, PP = np.meshgrid(Tg, Pg)
    z, shape = predict_df(conv_model, time=TOS, temp=TT, pres=PP, ratio=R, act=ACT)
    fig2 = go.Figure(go.Contour(x=Tg, y=Pg, z=z.reshape(shape), colorscale="Turbo",
                                contours=dict(showlabels=True), colorbar=dict(title="Conv %")))
    fig2.add_trace(go.Scatter(x=[T], y=[P], mode="markers", name="Operating point",
                              marker=dict(size=13, color="white", line=dict(width=2, color="black"))))
    style(fig2, "Conversion map: temperature x pressure", "Temperature (K)", "Pressure (bar)")
    right.plotly_chart(fig2)

    
# TAB 2: POISONING DETECTOR (Isolation Forest)

with tab2:
    st.markdown("Simulate what the sensors would show if a **sulfur-poisoning event** hit. The detector "
                "looks at how far *now* is from *recent normal behaviour*, so inputs are **deviations**.")
    d1, d2, d3 = st.columns(3)
    drop = d1.slider("Sudden conversion drop (pp)", 0.0, 25.0, 0.0, 0.5)
    t_dev = d2.slider("Temperature deviation (K)", -8.0, 8.0, 0.0, 0.5)
    p_dev = d3.slider("Pressure deviation (bar)", -4.0, 4.0, 0.0, 0.5)

    def det_frame(drops):
        drops = np.asarray(drops, dtype=float)
        n = drops.size
        return pd.DataFrame({"f_conv_dev": -drops, "f_yld_dev": -0.97 * drops,
                             "f_temp_dev": np.full(n, t_dev), "f_pres_dev": np.full(n, p_dev),
                             "f_conv_diff": -drops})[B["anomaly"]["features"]]

    iso = B["anomaly"]["model"]
    now = det_frame([drop])
    flagged = int(iso.predict(now)[0] == -1)
    score = float(-iso.decision_function(now)[0])   # > 0 means flagged

    cls, txt = ("crit", "● ANOMALY FLAGGED - POSSIBLE POISONING") if flagged else ("ok", "● NORMAL OPERATION")
    st.markdown(f'<span class="pill {cls}">{txt}</span>', unsafe_allow_html=True)
    st.write("")

    sweep = np.linspace(0, 25, 126)
    sc = -iso.decision_function(det_frame(sweep))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sweep, y=sc, mode="lines", name="Anomaly score",
                              line=dict(color="#38bdf8", width=2.6)))
    fig3.add_hline(y=0, line_dash="dash", line_color="#fbbf24", annotation_text="Alarm threshold")
    fig3.add_trace(go.Scatter(x=[drop], y=[score], mode="markers", name="Current scenario",
                              marker=dict(size=14, color="#ef4444" if flagged else "#4ade80",
                                          line=dict(width=2, color="white"))))
    style(fig3, "Anomaly score vs size of conversion drop", "Sudden conversion drop (pp)",
          "Anomaly score (higher = more abnormal)")
    st.plotly_chart(fig3)
    st.caption("Unsupervised Isolation Forest: it never saw poisoning labels during training. "
               "On held-out runs it reached ROC-AUC about 0.93 but only moderate precision/recall, "
               "so treat alerts as 'inspect the catalyst', not as proof.")


# TAB 3: METRICS FROM TRAINING

MODEL_NAMES = {
    "conversion_A_baseline_nophys": (
        "Conversion — Model A (Pure ML, No Physics: Conditions Only)"
    ),
    "conversion_A_conditions_only": (
        "Conversion — Model A (Physics-Informed: Conditions Only)"
    ),
    "conversion_B_baseline_nophys": (
        "Conversion — Model B (Pure ML, No Physics: Conditions + Health)"
    ),
    "conversion_B_with_health": (
        "Conversion — Model B (Physics-Informed Digital Twin: Conditions +"
        " Health)"
    ),
    "yield_A_baseline_nophys": (
        "Product Yield — Model A (Pure ML, No Physics: Conditions Only)"
    ),
    "yield_A_conditions_only": (
        "Product Yield — Model A (Physics-Informed: Conditions Only)"
    ),
    "yield_B_baseline_nophys": (
        "Product Yield — Model B (Pure ML, No Physics: Conditions + Health)"
    ),
    "yield_B_with_health": (
        "Product Yield — Model B (Physics-Informed Digital Twin: Conditions +"
        " Health)"
    ),
    "rul_all": "Remaining Useful Life (RUL — Full Operating Range)",
    "rul_near_eol": "Remaining Useful Life (RUL — Near End-of-Life)",
}

with tab3:
    mpath = model_dir / "metrics.json"
    if mpath.exists():
        try:
            m = json.loads(mpath.read_text())
            rows = [
                {
                    "Model": MODEL_NAMES.get(k, k),
                    "R2": round(v["R2"], 3),
                    "RMSE": round(v["RMSE"], 2),
                    "MAE": round(v["MAE"], 2),
                }
                for k, v in m.items()
                if "R2" in v
            ]
            st.markdown(
                "**Regression accuracy on held-out catalyst runs** (pp for"
                " conversion/yield, hours for RUL)"
            )
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, use_container_width=True
            )
            if "anomaly" in m:
                st.markdown(
                    "**Isolation Forest (Poisoning Detector):** ROC-AUC ="
                    f" {m['anomaly']['roc_auc']:.3f}, Average Precision ="
                    f" {m['anomaly']['avg_precision']:.3f}"
                )
        except Exception as exc:
            st.warning(f"Could not read metrics.json: {exc}")
    else:
        st.info("metrics.json not found. Re-run `python train_models.py`.")