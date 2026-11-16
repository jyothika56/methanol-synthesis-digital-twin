"""
app.py  -  Methanol Synthesis Reactor Digital Twin  |  Control-Room Dashboard
=============================================================================
Run with:   streamlit run app.py

Reads `reactor_data.csv` (produced by data_generator.py) and shows:
  * KPI cards         : average conversion, average yield, minimum catalyst health, ...
  * Sidebar filters   : temperature + pressure ranges (auto-detected from the CSV)
  * Plotly charts     : conversion/yield vs time, catalyst health with poisoning
                        highlights, and a temperature "volcano" scatter
  * Robust handling   : missing file, empty file, bad columns, empty filter result

LAYOUT NOTES (v2)
  All charts share ONE styling function (style_fig) that guarantees:
    - title on its own line at top-left, legend on a separate line below it
    - automargin + axis-title standoff so labels never collide with tick numbers
    - horizontal tick labels, capped tick counts, padded axis ranges
"""

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. PAGE CONFIG + STYLING  (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Reactor Digital Twin | Control Room",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid #1f2a3a;}

    /* ---- Header banner ---- */
    .hero {
        background: linear-gradient(135deg, #0b1f33 0%, #12304d 55%, #0f4c5c 100%);
        border: 1px solid #1f3b57; border-radius: 14px;
        padding: 20px 28px; margin-bottom: 18px;
    }
    .hero h1 {margin: 0; font-size: 1.85rem; color: #e8f1f8; letter-spacing: 0.4px; line-height: 1.25;}
    .hero p  {margin: 6px 0 0 0; color: #8fb3cf; font-size: 0.95rem; line-height: 1.5;}

    /* ---- KPI cards (fixed min-height + wrapping so text never overlaps) ---- */
    .kpi {
        background: #0f1b2a; border: 1px solid #1f2f44; border-left: 5px solid var(--accent);
        border-radius: 12px; padding: 14px 18px; min-height: 118px; height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35); overflow: hidden;
    }
    .kpi .label {color: #8aa0b8; font-size: 0.72rem; text-transform: uppercase;
                 letter-spacing: 1.2px; overflow-wrap: anywhere;}
    .kpi .value {color: #f2f7fb; font-size: clamp(1.45rem, 2.1vw, 2.0rem); font-weight: 700;
                 line-height: 1.3; white-space: nowrap;}
    .kpi .sub   {color: #6f8aa6; font-size: 0.78rem; overflow-wrap: anywhere;}

    /* ---- Status pill ---- */
    .pill {display: inline-block; padding: 4px 14px; border-radius: 999px;
           font-size: 0.8rem; font-weight: 700; letter-spacing: 0.8px;}
    .pill.ok   {background: #0e3b2a; color: #4ade80; border: 1px solid #1f6b4a;}
    .pill.warn {background: #3b2f0e; color: #fbbf24; border: 1px solid #7a5f17;}
    .pill.crit {background: #3b1215; color: #f87171; border: 1px solid #7a2229;}

    h3 {color: #d6e4f0; margin-top: 0.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# 2. CONSTANTS
# =============================================================================
DATA_FILE = "reactor_data.csv"
EOL_THRESHOLD = 0.30          # catalyst end-of-life activity (same as data_generator.py)
FLEET_BIN_H = 200             # hours per bin in "fleet average" view
MAX_SCATTER_POINTS = 8000     # keeps the volcano plot fast

# Keywords used to auto-detect columns (checked in order, after normalising names).
COLUMN_KEYWORDS = {
    "time":        ["timeonstream", "time", "hours"],
    "temperature": ["temperature", "temp"],
    "pressure":    ["pressure", "press"],
    "conversion":  ["conversion", "conv"],
    "yield":       ["yield"],
    "activity":    ["activity", "health"],
    "anomaly":     ["anomaly", "poison", "fault"],
    "run":         ["runid", "run", "batch", "lifecycle"],
    "rul":         ["rul", "remainingusefullife"],
}
REQUIRED_KEYS = ["time", "temperature", "pressure", "conversion", "yield", "activity"]

# Chart colours
C_CONV, C_YIELD, C_HEALTH, C_ALARM, C_EOL = "#22d3ee", "#a3e635", "#38bdf8", "#ef4444", "#fbbf24"

# Unit suffixes recognised in column names, e.g. 'temperature_K' -> 'Temperature (K)'
UNIT_MAP = {"k": "K", "bar": "bar", "h": "h", "hr": "h", "hours": "h", "pct": "%", "percent": "%"}


# =============================================================================
# 3. DATA LOADING + COLUMN DETECTION
# =============================================================================
def normalise(name: str) -> str:
    """'Temperature_K ' -> 'temperaturek'  (lowercase, letters/digits only)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def pretty(name: str) -> str:
    """Readable axis/slider label that PRESERVES case and turns unit suffixes into (unit).
    'time_on_stream_h' -> 'Time on stream (h)',  'temperature_K' -> 'Temperature (K)'."""
    tokens = [t for t in str(name).replace("-", "_").split("_") if t]
    if not tokens:
        return str(name)
    unit = None
    if len(tokens) > 1 and tokens[-1].lower() in UNIT_MAP:
        unit = UNIT_MAP[tokens[-1].lower()]
        tokens = tokens[:-1]
    label = " ".join(tokens)
    label = label[:1].upper() + label[1:]
    return f"{label} ({unit})" if unit else label


def detect_columns(columns) -> dict:
    """Map logical names ('temperature', ...) to the real CSV column names."""
    mapping, used = {}, set()
    for key, keywords in COLUMN_KEYWORDS.items():
        for kw in keywords:
            hit = next((c for c in columns if c not in used and kw in normalise(c)), None)
            if hit is not None:
                mapping[key] = hit
                used.add(hit)
                break
    return mapping


@st.cache_data(show_spinner="Loading reactor data...")
def read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    """Cached CSV read. `mtime` makes the cache refresh when the file changes."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_dataframe():
    """Return (DataFrame or None, source_label). Shows friendly errors, never a traceback."""
    candidates = [Path(__file__).resolve().parent / DATA_FILE, Path.cwd() / DATA_FILE]
    path = next((p for p in candidates if p.exists()), None)

    if path is not None:
        try:
            return read_csv_cached(str(path), path.stat().st_mtime), path.name
        except pd.errors.EmptyDataError:
            st.error(f"❌ `{path.name}` is empty. Re-run `python data_generator.py`.")
        except pd.errors.ParserError as exc:
            st.error(f"❌ `{path.name}` could not be parsed as CSV.\n\n{exc}")
        except PermissionError:
            st.error(f"❌ No permission to read `{path.name}`. Close it in Excel and retry.")
        except Exception as exc:  # last-resort safety net
            st.error(f"❌ Unexpected error while reading `{path.name}`: {exc}")
    else:
        st.error(
            f"❌ **`{DATA_FILE}` was not found.**\n\n"
            "1. Run `python data_generator.py` in this folder, **or**\n"
            "2. Make sure `app.py` and the CSV are in the same folder, **or**\n"
            "3. Upload the CSV below."
        )

    # Fallback: manual upload so the demo is never dead in front of the professor
    uploaded = st.file_uploader("Upload reactor_data.csv", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            df.columns = [str(c).strip() for c in df.columns]
            return df, uploaded.name
        except Exception as exc:
            st.error(f"❌ Could not read the uploaded file: {exc}")
    return None, ""


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================
# st.plotly_chart changed its sizing argument across Streamlit versions
# (use_container_width -> width="stretch"). Detect which one is available.
_HAS_WIDTH_ARG = "width" in inspect.signature(st.plotly_chart).parameters


def show(container, fig, key):
    """Render a Plotly figure full-width in a column/container, on any Streamlit version."""
    if _HAS_WIDTH_ARG:
        container.plotly_chart(fig, width="stretch", key=key)
    else:
        container.plotly_chart(fig, use_container_width=True, key=key)


def kpi_card(col, label, value, sub, accent):
    """Render one styled KPI card inside a Streamlit column."""
    col.markdown(
        f"""<div class="kpi" style="--accent:{accent}">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def range_slider(label, series, key):
    """Sidebar range slider built from the data's own min/max (handles constant columns)."""
    lo, hi = float(np.floor(series.min())), float(np.ceil(series.max()))
    if lo >= hi:
        st.sidebar.caption(f"{label}: constant ({lo:g}), no filter needed.")
        return lo, hi
    step = float(max(round((hi - lo) / 200, 2), 0.01))
    return st.sidebar.slider(label, lo, hi, (lo, hi), step=step, key=key)


def anomaly_windows(t: np.ndarray, flag: np.ndarray, max_windows: int = 40):
    """Return (start, end) time pairs of contiguous anomaly blocks (for shaded bands)."""
    flag = flag.astype(int)
    edges = np.diff(np.concatenate(([0], flag, [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    return [(t[s], t[e]) for s, e in zip(starts, ends)][:max_windows]


def style_fig(fig, title, x_title, y_title, height=450, legend=True):
    """
    ONE layout recipe for every chart, designed so nothing can overlap:

      +----------------------------------------+
      |  Title (top-left, own line)            |  <- y=0.97, anchored top
      |  Legend (horizontal, own line)         |  <- sits just above the plot area
      |  +----------------------------------+  |
      |  |            plot area             |  |
      |  +----------------------------------+  |
      |     tick labels / axis title           |  <- automargin + title_standoff
      +----------------------------------------+
    """
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top",
                   font=dict(size=16), pad=dict(l=4)),
        height=height,
        font=dict(size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,27,42,0.6)",
        margin=dict(l=70, r=30, t=110, b=70),          # generous minimums; automargin grows them
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(font_size=12),
    )
    axis_common = dict(
        gridcolor="#1c2b3d", zeroline=False,
        automargin=True,                 # auto-grow margin to fit tick labels + axis title
        title_standoff=14,               # gap between tick labels and the axis title
        tickangle=0,                     # horizontal tick labels (readable)
        ticks="outside", tickcolor="#2a3b52", ticklen=5,
        showline=True, linecolor="#2a3b52",
    )
    fig.update_xaxes(title_text=x_title, nticks=8, **axis_common)
    fig.update_yaxes(title_text=y_title, nticks=7, **axis_common)
    return fig


# =============================================================================
# 5. LOAD DATA + VALIDATE COLUMNS
# =============================================================================
st.markdown(
    """<div class="hero">
         <h1>⚗️ Methanol Synthesis Reactor - Digital Twin Control Room</h1>
         <p>Fixed-bed Cu/ZnO/Al₂O₃ reactor &nbsp;|&nbsp; CO + 2H₂ ⇌ CH₃OH &nbsp;|&nbsp;
            Catalyst health monitoring &amp; poisoning detection</p>
       </div>""",
    unsafe_allow_html=True,
)

raw, source_name = get_dataframe()
if raw is None:
    st.stop()
if raw.empty:
    st.error("❌ The dataset has headers but no rows. Re-run `python data_generator.py`.")
    st.stop()

cols = detect_columns(list(raw.columns))
missing = [k for k in REQUIRED_KEYS if k not in cols]
if missing:
    st.error(
        f"❌ Could not find column(s) for: **{', '.join(missing)}**.\n\n"
        f"Columns in your file: `{list(raw.columns)}`"
    )
    st.stop()

# Work on a clean copy: force numeric types, drop unusable rows
df = raw.copy()
numeric_keys = [k for k in cols if k != "run"]
for k in numeric_keys:
    df[cols[k]] = pd.to_numeric(df[cols[k]], errors="coerce")
df = df.dropna(subset=[cols[k] for k in REQUIRED_KEYS]).reset_index(drop=True)
if df.empty:
    st.error("❌ No valid numeric rows remain after cleaning. Check the CSV content.")
    st.stop()

# Standard anomaly flag (0/1) even if the column is missing
if "anomaly" in cols:
    df["_anom"] = (df[cols["anomaly"]].fillna(0) > 0).astype(int)
else:
    df["_anom"] = 0
    st.warning("⚠️ No anomaly column detected, so poisoning highlights are disabled.")

# Health scale: fraction (0-1) or percent (0-100)?
HEALTH_SCALE = 100.0 if df[cols["activity"]].max() > 1.5 else 1.0
EOL_LINE = EOL_THRESHOLD * HEALTH_SCALE

# =============================================================================
# 6. SIDEBAR CONTROLS
# =============================================================================
st.sidebar.markdown("## 🎛️ Control Panel")
st.sidebar.caption(f"Data source: `{source_name}`")

st.sidebar.markdown("### Operating window")
t_lo, t_hi = range_slider(pretty(cols["temperature"]), df[cols["temperature"]], "temp_slider")
p_lo, p_hi = range_slider(pretty(cols["pressure"]), df[cols["pressure"]], "press_slider")

mask = df[cols["temperature"]].between(t_lo, t_hi) & df[cols["pressure"]].between(p_lo, p_hi)
fdf = df[mask].copy()

st.sidebar.markdown("### Time-series view")
has_run = "run" in cols
view_mode = st.sidebar.radio(
    "Display mode",
    ["Single catalyst run", "Fleet average"] if has_run else ["Fleet average"],
    help="Single run = one catalyst lifecycle. Fleet average = mean of all runs per time bin.",
)

st.sidebar.divider()
st.sidebar.metric("Rows in selected window", f"{len(fdf):,}", f"of {len(df):,} total", delta_color="off")

if fdf.empty:
    st.warning("⚠️ No data points fall inside the selected temperature/pressure window. Widen the sliders.")
    st.stop()

# =============================================================================
# 7. KPI CARDS
# =============================================================================
avg_conv = fdf[cols["conversion"]].mean()
avg_yield = fdf[cols["yield"]].mean()
min_health = fdf[cols["activity"]].min()
health_pct = min_health / HEALTH_SCALE * 100
n_anom = int(fdf["_anom"].sum())

# Overall status for the pill
if health_pct <= EOL_THRESHOLD * 100 + 5:
    status_cls, status_txt = "crit", "● CATALYST REPLACEMENT DUE"
elif health_pct <= 50 or n_anom > 0:
    status_cls, status_txt = "warn", "● MONITOR - DEGRADATION / POISONING SEEN"
else:
    status_cls, status_txt = "ok", "● NOMINAL OPERATION"
st.markdown(f'<span class="pill {status_cls}">{status_txt}</span>', unsafe_allow_html=True)
st.write("")

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Average conversion", f"{avg_conv:.1f} %",
         f"dataset avg {df[cols['conversion']].mean():.1f} %", "#22d3ee")
kpi_card(k2, "Average yield", f"{avg_yield:.1f} %",
         f"dataset avg {df[cols['yield']].mean():.1f} %", "#a3e635")
health_accent = "#ef4444" if health_pct < 40 else "#fbbf24" if health_pct < 65 else "#4ade80"
kpi_card(k3, "Min catalyst health", f"{health_pct:.1f} %",
         f"end-of-life at {EOL_THRESHOLD * 100:.0f} %", health_accent)
kpi_card(k4, "Poisoning-flagged rows", f"{n_anom:,}",
         f"{100 * n_anom / len(fdf):.2f} % of selected rows", "#ef4444" if n_anom else "#4ade80")
st.write("")

# =============================================================================
# 8. BUILD THE TIME-SERIES FRAME (single run OR fleet average)
# =============================================================================
tcol = cols["time"]

if view_mode == "Single catalyst run" and has_run:
    run_options = sorted(fdf[cols["run"]].dropna().unique().tolist())
    default_run = fdf[cols["run"]].value_counts().idxmax()  # run with most rows in window
    run_sel = st.sidebar.selectbox("Catalyst run", run_options, index=run_options.index(default_run))
    ts = fdf[fdf[cols["run"]] == run_sel].sort_values(tcol)
    plot_df = pd.DataFrame({
        "t": ts[tcol].values,
        "conv": ts[cols["conversion"]].values,
        "yld": ts[cols["yield"]].values,
        "act": ts[cols["activity"]].values,
        "anom": ts["_anom"].values.astype(bool),
    })
    view_label = f"Run {run_sel}"
    show_bands = True
else:
    binned = fdf.assign(_bin=(fdf[tcol] // FLEET_BIN_H) * FLEET_BIN_H).groupby("_bin")
    plot_df = pd.DataFrame({
        "t": binned[tcol].mean().index.values,
        "conv": binned[cols["conversion"]].mean().values,
        "yld": binned[cols["yield"]].mean().values,
        "act": binned[cols["activity"]].mean().values,
        "anom": (binned["_anom"].max().values > 0),
    })
    view_label = f"Fleet average ({FLEET_BIN_H} h bins)"
    show_bands = False

x_title_time = pretty(tcol)
bad = plot_df[plot_df["anom"]]

# =============================================================================
# 9. CHART ROW 1: CONVERSION & YIELD  |  CATALYST HEALTH
# =============================================================================
st.markdown(f"### 📈 Reactor performance - {view_label}")
left, right = st.columns(2)

# ---- 9a. Conversion + Yield vs time-on-stream -------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=plot_df["t"], y=plot_df["conv"], mode="lines", name="Conversion (%)",
                          line=dict(color=C_CONV, width=2.2)))
fig1.add_trace(go.Scatter(x=plot_df["t"], y=plot_df["yld"], mode="lines", name="Yield (%)",
                          line=dict(color=C_YIELD, width=2.2)))
if not bad.empty:
    fig1.add_trace(go.Scatter(x=bad["t"], y=bad["conv"], mode="markers", name="Poisoning event",
                              cliponaxis=False,
                              marker=dict(color=C_ALARM, symbol="x", size=9,
                                          line=dict(width=1, color="white"))))
style_fig(fig1, "Conversion & yield vs time-on-stream", x_title_time, "Percent (%)")
# Pad the y-range so lines and markers never touch the plot frame
y_all = pd.concat([plot_df["conv"], plot_df["yld"]]).dropna()
if not y_all.empty:
    pad = max((y_all.max() - y_all.min()) * 0.08, 1.0)
    fig1.update_yaxes(range=[max(0.0, y_all.min() - pad), min(100.0, y_all.max() + pad)])
show(left, fig1, "fig_conv_yield")

# ---- 9b. Catalyst health with anomaly highlights ----------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=plot_df["t"], y=plot_df["act"], mode="lines", name="Catalyst activity",
                          line=dict(color=C_HEALTH, width=2.4),
                          fill="tozeroy", fillcolor="rgba(56,189,248,0.10)"))
if not bad.empty:
    fig2.add_trace(go.Scatter(x=bad["t"], y=bad["act"], mode="markers", name="Poisoning event",
                              cliponaxis=False,
                              marker=dict(color=C_ALARM, size=8, symbol="circle",
                                          line=dict(width=1, color="white"))))
if show_bands:  # shaded red bands marking each poisoning window
    for x0, x1 in anomaly_windows(plot_df["t"].values, plot_df["anom"].values):
        fig2.add_vrect(x0=x0, x1=x1, fillcolor=C_ALARM, opacity=0.18, line_width=0)
# End-of-life threshold drawn as a LEGEND ENTRY (no floating text to collide with data)
fig2.add_trace(go.Scatter(x=[plot_df["t"].min(), plot_df["t"].max()], y=[EOL_LINE, EOL_LINE],
                          mode="lines", name=f"End-of-life ({EOL_THRESHOLD * 100:.0f} %)",
                          line=dict(color=C_EOL, width=1.6, dash="dash"), hoverinfo="skip"))
style_fig(fig2, "Catalyst health & poisoning events", x_title_time,
          "Activity" + (" (%)" if HEALTH_SCALE > 1 else " (0-1)"))
fig2.update_yaxes(range=[0, max(plot_df["act"].max() * 1.08, EOL_LINE * 1.3)])
show(right, fig2, "fig_health")

# =============================================================================
# 10. CHART ROW 2: TEMPERATURE "VOLCANO" + RUL
# =============================================================================
st.markdown("### 🔬 Process insight")
c1, c2 = st.columns(2)

# ---- 10a. Conversion vs temperature, coloured by catalyst health ------------
sample = fdf.sample(min(len(fdf), MAX_SCATTER_POINTS), random_state=0)
fig3 = go.Figure(go.Scattergl(
    x=sample[cols["temperature"]], y=sample[cols["conversion"]], mode="markers",
    marker=dict(size=5, color=sample[cols["activity"]], colorscale="Turbo", opacity=0.75,
                colorbar=dict(title=dict(text="Activity", side="right"),
                              thickness=14, len=0.85, outlinewidth=0)),
    hovertemplate="T=%{x:.1f}<br>Conv=%{y:.1f} %<br>Activity=%{marker.color:.2f}<extra></extra>",
))
style_fig(fig3, "Conversion vs temperature", pretty(cols["temperature"]), "Conversion (%)", legend=False)
fig3.update_layout(hovermode="closest", margin=dict(l=70, r=80, t=110, b=70))
show(c1, fig3, "fig_volcano")

# ---- 10b. Remaining useful life (if available) or a distribution fallback ---
if "rul" in cols and has_run and view_mode == "Single catalyst run":
    ts_rul = fdf[fdf[cols["run"]] == run_sel].sort_values(tcol)
    fig4 = go.Figure(go.Scatter(x=ts_rul[tcol], y=ts_rul[cols["rul"]], mode="lines", name="RUL",
                                line=dict(color="#c084fc", width=2.4),
                                fill="tozeroy", fillcolor="rgba(192,132,252,0.12)"))
    style_fig(fig4, f"Remaining useful life - run {run_sel}", x_title_time, "RUL (hours)", legend=False)
else:
    fig4 = go.Figure(go.Histogram(x=fdf[cols["conversion"]], nbinsx=40, marker_color=C_CONV, opacity=0.85))
    style_fig(fig4, "Conversion distribution (selected window)", "Conversion (%)", "Count", legend=False)
    fig4.update_layout(hovermode="closest")
show(c2, fig4, "fig_rul_or_hist")

# =============================================================================
# 11. RAW DATA + DOWNLOAD
# =============================================================================
with st.expander("📋 Inspect filtered data / detected columns"):
    st.caption("Detected column mapping: " + ", ".join(f"**{k}** → `{v}`" for k, v in cols.items()))
    st.dataframe(fdf.drop(columns=["_anom"]).head(500))
    st.download_button("⬇️ Download filtered CSV",
                       fdf.drop(columns=["_anom"]).to_csv(index=False).encode("utf-8"),
                       file_name="reactor_filtered.csv", mime="text/csv")

st.caption("Digital-twin demo | synthetic data generated with Arrhenius kinetics, "
           "thermal deactivation and sulfur-poisoning events.")