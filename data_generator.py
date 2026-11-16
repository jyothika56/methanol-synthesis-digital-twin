
"""
data_generator.py
=================
Synthetic digital-twin data for an industrial fixed-bed METHANOL SYNTHESIS reactor
    CO + 2 H2  <=>  CH3OH      (exothermic, Cu/ZnO/Al2O3 catalyst)

The script simulates many catalyst "lifecycles". In each one a fresh catalyst is
loaded, the reactor runs with slowly drifting operating conditions, the catalyst
deactivates thermally, and random sulfur-poisoning events hit it. Output is a
tidy table (reactor_data.csv) suitable for supervised ML, RUL estimation and
anomaly detection.

MODEL SUMMARY (a reduced-order, lumped model - NO PDEs, NO PID):
  1. Arrhenius kinetics:      k(T) = A * exp(-Ea / (R*T))
  2. Equilibrium limit:       X_eq(T, P, H2/CO) from log10(Kp) = 5139/T - 12.621
  3. Conversion:              X = X_eq * (1 - exp(-Da)),  Da = k(T) * activity * g(P) * f(H2/CO)
  4. Thermal deactivation:    activity decays exponentially, faster at high T
  5. Sulfur poisoning:        random events -> permanent step loss + temporary dip
  6. Yield:                   Yield = Conversion * Selectivity(T, H2/CO)
  7. Sensor noise:            Gaussian noise added to all *measured* signals

Known simplifications (mention these in your report - they show you understand
the model's limits):
  * CO-only carbon source (no CO2 route), ideal-gas equilibrium (no fugacity).
  * Lumped pseudo-first-order approach-to-equilibrium instead of a full
    Langmuir-Hinshelwood rate law with axial profiles.
  * Deactivation is ACCELERATED vs. real plants (real Cu/ZnO lasts 2-4 years)
    so that the 0-8000 h window shows interesting catalyst-life behaviour.

Run:  python data_generator.py
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter   # used to generate smooth drifting signals


# 1. CONFIGURATION  (edit these to change the dataset - everything is in one place)


SEED = 42                     # fixed seed -> reproducible dataset
N_RUNS = 80                   # number of catalyst lifecycles (fresh loads) to simulate
DT_H = 8.0                    # sampling interval in hours (one row every 8 h)
MAX_TOS_H = 8000.0            # maximum time-on-stream per run (hours)
HORIZON_H = 40000.0           # internal look-ahead horizon used ONLY to find the
                              # true end-of-life time (needed for RUL labels)
TRUNCATE_AT_EOL = True        # True: run stops when catalyst hits end-of-life
                              #       (realistic: operators replace the catalyst)
OUTPUT_CSV = "reactor_data.csv"
MAKE_PREVIEW_PLOT = True      # write a quick Plotly sanity-check HTML file
PREVIEW_HTML = "reactor_preview.html"

# ---- Physical constants -----------------------------------------------------
R_GAS = 8.314                 # J/(mol*K)
BAR_TO_ATM = 1.0 / 1.01325    # unit conversion (equilibrium correlation uses atm)

# ---- Realistic industrial operating windows ---------------------------------
# Each run picks a random SETPOINT inside these ranges; the actual conditions then
# wander around the setpoint (feed variations, ambient effects, operator tweaks).
T_SETPOINT_RANGE = (490.0, 580.0)    # K    (~222-272 degC; typical is 220-280 degC)
P_SETPOINT_RANGE = (55.0, 95.0)      # bar  (typical industrial loop: 50-100 bar)
RATIO_SETPOINT_RANGE = (1.8, 3.0)    # H2/CO molar feed ratio (stoichiometric = 2.0)

# Hard physical limits after fluctuations are added (safety clipping)
T_LIMITS = (470.0, 605.0)            # K
P_LIMITS = (40.0, 110.0)             # bar
RATIO_LIMITS = (1.5, 3.5)            # -

# Process fluctuation (true plant variability, NOT sensor noise)
T_FLUCT_SIGMA = 3.0                  # K
P_FLUCT_SIGMA = 1.5                  # bar
RATIO_FLUCT_SIGMA = 0.05             # -
FLUCT_PERSISTENCE = 0.98             # AR(1) coefficient: closer to 1 = slower drift

# ---- Arrhenius kinetics ------------------------------------------------------
EA_J_MOL = 90_000.0           # activation energy, J/mol (typical Cu/ZnO range 80-100 kJ/mol)
A_PRE = 2.2e9                 # lumped pre-exponential factor (includes fixed catalyst
                              # loading and space-time, so k(T) is dimensionless "Da" scale)
P_REF_BAR = 80.0              # reference pressure for the rate pressure-dependence
P_ORDER = 0.6                 # rate ~ (P/P_ref)^0.6
RATIO_OPT = 2.2               # H2/CO ratio where the lumped rate is highest
RATIO_WIDTH = 1.0             # how fast the rate falls off away from the optimum

# ---- Catalyst deactivation ---------------------------------------------------
KD_REF_PER_H = 1.5e-4         # thermal decay rate constant at the reference temperature (1/h)
T_REF_DEACT_K = 523.15        # reference temperature for deactivation (250 degC)
ED_J_MOL = 60_000.0           # "activation energy" of sintering -> hotter = faster aging
BATCH_VARIABILITY = 0.15      # lognormal sigma: each catalyst batch ages slightly differently
A_EOL = 0.30                  # end-of-life: replace catalyst when PERMANENT activity <= 0.30

# ---- Sulfur poisoning events -------------------------------------------------
POISON_MEAN_INTERVAL_H = 3000.0       # on average one event per 8000 h of operation
POISON_STEP_RANGE = (0.85, 0.95)      # permanent activity multiplier (lose 5-15 %)
POISON_TRANSIENT_RANGE = (0.15, 0.35) # extra TEMPORARY activity dip (15-35 %)
TAU_RECOVERY_H = 24.0                 # transient dip recovers with this time constant
ANOMALY_WINDOW_H = 48.0               # rows within 48 h of an event get is_anomaly = 1
ANOMALY_WINDOW_STEPS = int(np.ceil(ANOMALY_WINDOW_H / DT_H))

# ---- Selectivity (how much converted CO becomes methanol, not CH4/DME/etc.) --
SEL_BASE = 0.99               # selectivity at 490 K
SEL_T_SLOPE = 0.0007          # selectivity lost per K above 490 K (side reactions)
SEL_LEAN_PENALTY = 0.04       # extra loss per unit of H2/CO below 2.0 (H2-starved)

# ---- Sensor noise (Gaussian, 1-sigma) ---------------------------------------
NOISE_T_K = 1.0               # thermocouple
NOISE_P_BAR = 0.4             # pressure transmitter
NOISE_RATIO = 0.03            # flow-ratio computation
NOISE_CONV_PCT = 0.8          # online gas chromatograph, percentage points
NOISE_YIELD_PCT = 0.8         # online gas chromatograph, percentage points



# 2. PHYSICS / CHEMISTRY FUNCTIONS


def arrhenius_k(T):
    """Arrhenius rate constant k(T) = A * exp(-Ea / (R*T)).  T in kelvin."""
    return A_PRE * np.exp(-EA_J_MOL / (R_GAS * T))


def deactivation_rate(T):
    """
    Thermal deactivation rate kd(T) in 1/h, Arrhenius-type around a reference T:
        kd = kd_ref * exp( -(Ed/R) * (1/T - 1/T_ref) )
    Hotter operation -> faster sintering of Cu crystallites -> faster aging.
    """
    return KD_REF_PER_H * np.exp(-(ED_J_MOL / R_GAS) * (1.0 / T - 1.0 / T_REF_DEACT_K))


def equilibrium_conversion(T, P_bar, ratio):
    """
    Equilibrium CO conversion for CO + 2H2 <=> CH3OH.

    Basis: 1 mol CO and `ratio` mol H2 fed. With CO conversion X:
        n_CO = 1 - X,  n_H2 = ratio - 2X,  n_MeOH = X,  n_total = 1 + ratio - 2X
    Equilibrium constant (Kp in atm^-2, widely used correlation):
        log10(Kp) = 5139/T - 12.621
        Kp = y_MeOH / (y_CO * y_H2^2 * P^2)
    =>  n_MeOH * n_tot^2 / (n_CO * n_H2^2) = Kp * P^2     (solve for X)

    The left side rises monotonically from 0 to infinity as X goes from 0 to X_max,
    so a vectorised bisection (60 iterations) solves every row at once, fast.
    """
    T = np.asarray(T, dtype=float)
    P_atm = np.asarray(P_bar, dtype=float) * BAR_TO_ATM
    ratio = np.asarray(ratio, dtype=float)

    Kp = 10.0 ** (5139.0 / T - 12.621)
    target = Kp * P_atm ** 2

    # Max possible conversion: limited by CO (X<=1) or by H2 (X <= ratio/2)
    lo = np.zeros_like(T)
    hi = np.minimum(1.0, ratio / 2.0) - 1e-9

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        n_co = 1.0 - mid
        n_h2 = ratio - 2.0 * mid
        n_tot = 1.0 + ratio - 2.0 * mid
        q = mid * n_tot ** 2 / (n_co * n_h2 ** 2)
        too_high = q > target              # overshoot equilibrium -> move upper bound down
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def methanol_selectivity(T, ratio):
    """
    Fraction of converted CO that ends up as methanol (rest -> CH4, DME, higher
    alcohols). Falls with temperature and when the feed is H2-lean (ratio < 2).
    """
    sel = SEL_BASE - SEL_T_SLOPE * (T - 490.0)
    sel = sel - SEL_LEAN_PENALTY * np.maximum(2.0 - ratio, 0.0)
    return np.clip(sel, 0.80, 0.995)


# 3. SIGNAL HELPERS


def drifting_signal(rng, n, sigma, phi=FLUCT_PERSISTENCE):
    """
    Zero-mean AR(1) process: x[k] = phi*x[k-1] + noise. Produces SMOOTH slow drifts
    (like real feed/ambient variations) with stationary std = sigma.
    Uses scipy's lfilter so there is no slow Python loop.
    """
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - phi ** 2), n)
    eps[0] = rng.normal(0.0, sigma)                 # start from the stationary distribution
    return lfilter([1.0], [1.0, -phi], eps)


# 4. SIMULATE ONE CATALYST LIFECYCLE


def simulate_one_run(run_id, rng):
    """Simulate one fresh catalyst load and return its rows as a DataFrame."""

    # Total internal time grid (long horizon, so we can find true end-of-life)
    n_total = int(HORIZON_H / DT_H) + 1
    n_out_max = int(MAX_TOS_H / DT_H) + 1
    t = np.arange(n_total) * DT_H                   # time-on-stream, hours

    # ---- 4a. True operating conditions: setpoint + slow drift ---------------
    T_set = rng.uniform(*T_SETPOINT_RANGE)
    P_set = rng.uniform(*P_SETPOINT_RANGE)
    r_set = rng.uniform(*RATIO_SETPOINT_RANGE)

    T_true = np.clip(T_set + drifting_signal(rng, n_total, T_FLUCT_SIGMA), *T_LIMITS)
    P_true = np.clip(P_set + drifting_signal(rng, n_total, P_FLUCT_SIGMA), *P_LIMITS)
    r_true = np.clip(r_set + drifting_signal(rng, n_total, RATIO_FLUCT_SIGMA), *RATIO_LIMITS)

    # ---- 4b. Thermal (exponential) deactivation -----------------------------
    # a_thermal(t) = exp( - integral of kd(T(t)) dt ). Because T wanders, we
    # integrate the instantaneous rate with a cumulative sum.
    batch_factor = rng.lognormal(mean=0.0, sigma=BATCH_VARIABILITY)   # batch-to-batch variation
    kd = deactivation_rate(T_true) * batch_factor
    cum_decay = np.concatenate(([0.0], np.cumsum(kd[:-1] * DT_H)))
    a_thermal = np.exp(-cum_decay)

    # ---- 4c. Sulfur-poisoning events (Poisson process) ----------------------
    # Each event causes:
    #   (i)  a PERMANENT multiplicative loss of activity (sulfur bonds to Cu sites)
    #   (ii) a TEMPORARY extra dip that recovers exponentially (loosely held S species)
    # Rows within ANOMALY_WINDOW_H after the event are labelled is_anomaly = 1.
    n_events = rng.poisson(HORIZON_H / POISON_MEAN_INTERVAL_H)
    event_idx = np.sort(rng.integers(1, n_total, size=n_events))

    perm_mult = np.ones(n_total)          # cumulative permanent loss factor
    transient_mult = np.ones(n_total)     # temporary dip factor
    is_anomaly = np.zeros(n_total, dtype=int)

    for idx in event_idx:
        step = rng.uniform(*POISON_STEP_RANGE)
        dip = rng.uniform(*POISON_TRANSIENT_RANGE)
        elapsed = t[idx:] - t[idx]
        perm_mult[idx:] *= step
        transient_mult[idx:] *= 1.0 - dip * np.exp(-elapsed / TAU_RECOVERY_H)
        is_anomaly[idx: idx + ANOMALY_WINDOW_STEPS] = 1

    # Permanent activity decides end-of-life; effective activity (incl. the
    # temporary dip) is what the reactor "feels" and what drives conversion.
    a_perm = a_thermal * perm_mult
    a_eff = a_perm * transient_mult

    # ---- 4d. True end-of-life and Remaining Useful Life ---------------------
    below = np.where(a_perm <= A_EOL)[0]
    eol_idx = int(below[0]) if below.size > 0 else n_total - 1   # fallback: censored at horizon
    eol_time = t[eol_idx]

    n_out = n_out_max
    if TRUNCATE_AT_EOL:
        n_out = min(n_out_max, eol_idx + 1)    # keep the EOL row (RUL = 0), drop the rest

    # Slice everything to the rows we actually output
    sl = slice(0, n_out)
    t_o, T_o, P_o, r_o = t[sl], T_true[sl], P_true[sl], r_true[sl]
    a_o, anom_o = a_eff[sl], is_anomaly[sl]
    rul_o = np.maximum(eol_time - t_o, 0.0)

    # ---- 4e. Reactor performance (true, noise-free) -------------------------
    k = arrhenius_k(T_o)
    pressure_factor = (P_o / P_REF_BAR) ** P_ORDER
    ratio_factor = np.exp(-0.5 * ((r_o - RATIO_OPT) / RATIO_WIDTH) ** 2)

    damkohler = k * a_o * pressure_factor * ratio_factor       # dimensionless reaction "strength"
    x_eq = equilibrium_conversion(T_o, P_o, r_o)               # thermodynamic ceiling
    conversion = x_eq * (1.0 - np.exp(-damkohler))             # approach-to-equilibrium
    yield_true = conversion * methanol_selectivity(T_o, r_o)

    # ---- 4f. Sensor noise on all MEASURED signals ---------------------------
    T_meas = T_o + rng.normal(0.0, NOISE_T_K, n_out)
    P_meas = P_o + rng.normal(0.0, NOISE_P_BAR, n_out)
    r_meas = r_o + rng.normal(0.0, NOISE_RATIO, n_out)
    conv_meas = np.clip(conversion * 100.0 + rng.normal(0.0, NOISE_CONV_PCT, n_out), 0.0, 100.0)
    yield_meas = np.clip(yield_true * 100.0 + rng.normal(0.0, NOISE_YIELD_PCT, n_out), 0.0, None)
    yield_meas = np.minimum(yield_meas, conv_meas)              # yield cannot exceed conversion

    return pd.DataFrame({
        "run_id": run_id,                        # which catalyst lifecycle (use for train/test split!)
        "time_on_stream_h": t_o,
        "temperature_K": np.round(T_meas, 2),    # measured (noisy)
        "pressure_bar": np.round(P_meas, 2),     # measured (noisy)
        "feed_ratio_H2_CO": np.round(r_meas, 3), # measured (noisy)
        "conversion_pct": np.round(conv_meas, 2),   # measured (noisy)
        "yield_pct": np.round(yield_meas, 2),       # measured (noisy)
        "catalyst_activity": np.round(a_o, 4),      # ground-truth label (0-1, noise-free)
        "rul_hours": np.round(rul_o, 1),            # ground-truth label (noise-free)
        "is_anomaly": anom_o,                       # 1 = poisoning event window, 0 = normal
    })


# 5. OPTIONAL PREVIEW PLOT (quick visual sanity check)


def make_preview(df, path):
    """Writes an interactive Plotly HTML showing conversion + activity for 3 runs."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Measured CO conversion (%) - red X = poisoning window",
                        "True catalyst activity (0-1)"),
    )
    for rid in df["run_id"].unique()[:3]:
        run = df[df["run_id"] == rid]
        bad = run[run["is_anomaly"] == 1]
        fig.add_trace(go.Scatter(x=run["time_on_stream_h"], y=run["conversion_pct"],
                                 mode="lines", name=f"Run {rid}", legendgroup=f"r{rid}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bad["time_on_stream_h"], y=bad["conversion_pct"],
                                 mode="markers", marker=dict(color="red", symbol="x", size=9),
                                 name=f"Run {rid} poisoning", legendgroup=f"r{rid}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=run["time_on_stream_h"], y=run["catalyst_activity"],
                                 mode="lines", name=f"Run {rid} activity",
                                 legendgroup=f"r{rid}", showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="Time on stream (h)", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=700, title="Methanol reactor - data preview")
    fig.write_html(path, include_plotlyjs=True)   # embedded JS -> works offline
    print(f"Preview plot written to: {path}  (double-click it to open in your browser)")


# 6. MAIN


def main():
    rng = np.random.default_rng(SEED)

    frames = [simulate_one_run(run_id=i + 1, rng=rng) for i in range(N_RUNS)]
    df = pd.concat(frames, ignore_index=True)

    assert not df.isna().any().any(), "Generated data contains NaNs - check parameters."
    df.to_csv(OUTPUT_CSV, index=False)

    # ---- Summary printed to the terminal ------------------------------------
    run_lengths = df.groupby("run_id")["time_on_stream_h"].max()
    print("=" * 62)
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns  ->  {OUTPUT_CSV}")
    print(f"Catalyst lifecycles (runs)      : {N_RUNS}")
    print(f"Run length (h)  min/mean/max    : {run_lengths.min():.0f} / "
          f"{run_lengths.mean():.0f} / {run_lengths.max():.0f}")
    print(f"Anomaly rows (poisoning)        : {df['is_anomaly'].sum():,} "
          f"({100 * df['is_anomaly'].mean():.2f} %)")
    print(f"Temperature range (K)           : {df['temperature_K'].min():.1f} - {df['temperature_K'].max():.1f}")
    print(f"Pressure range (bar)            : {df['pressure_bar'].min():.1f} - {df['pressure_bar'].max():.1f}")
    print(f"H2/CO range                     : {df['feed_ratio_H2_CO'].min():.2f} - {df['feed_ratio_H2_CO'].max():.2f}")
    print(f"Conversion range (%)            : {df['conversion_pct'].min():.1f} - {df['conversion_pct'].max():.1f}")
    print(f"Max RUL (h)                     : {df['rul_hours'].max():.0f}")
    print("=" * 62)

    if MAKE_PREVIEW_PLOT:
        try:
            make_preview(df, PREVIEW_HTML)
        except Exception as exc:    # plotting is optional; never lose the CSV because of it
            print(f"(Preview plot skipped: {exc})")


if __name__ == "__main__":
    main()