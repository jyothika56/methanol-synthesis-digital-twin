"""
physics_features.py  -  Thermodynamic + kinetic feature engineering ("grey-box" ML)
====================================================================================
Shared by train_models.py (training) and pages/1_What_If_Simulator.py (inference).
Keeping ONE copy guarantees the model sees identical features in both places.

Input DataFrame must contain the canonical columns:
    time [h], temp [K], pres [bar], ratio [H2/CO]   (+ act [0-1] if use_activity=True)

Features added (see PHYS_COLS):
    x_eq       equilibrium CO conversion for CO + 2H2 <=> CH3OH  (thermodynamic ceiling, 0-1)
    k_arr      Arrhenius rate constant  k = A * exp(-Ea / (R*T))
    da_proxy   Damkohler-like number    k * activity * (P/Pref)^n   ("how hard the catalyst works")
    x_phys     physics-only estimate    x_eq * (1 - exp(-da_proxy))  (approach-to-equilibrium)
    gap_to_eq  x_eq - x_phys            (how far the reactor sits below its equilibrium ceiling)

WHY COMPOSITE FEATURES?  Tree models (XGBoost etc.) are invariant to monotonic transforms
of a SINGLE column, so 1/T, ln(k) or k(T) alone add nothing. What trees struggle to learn
is the INTERACTION of T, P, ratio and activity - which is exactly what x_eq and x_phys encode.
"""

import numpy as np
import pandas as pd

R_GAS = 8.314                    # J/(mol K)
BAR_TO_ATM = 1.0 / 1.01325

# ---- Kinetic constants (in a real project: fit these to lab/plant data) ------------------
EA_J_MOL = 90_000.0              # activation energy, J/mol (Cu/ZnO literature range ~80-100 kJ/mol)
A_PRE = 2.2e9                    # lumped pre-exponential factor
P_REF_BAR = 80.0                 # reference pressure
P_ORDER = 0.5                    # assumed pressure order of the rate

# ---- Crude deactivation model, used ONLY when real catalyst health is unknown (variant A) --
KD_REF_PER_H = 1.5e-4            # decay constant at T_REF_K, 1/h
ED_J_MOL = 60_000.0              # sintering "activation energy"
T_REF_K = 523.15

PHYS_COLS = ["x_eq", "k_arr", "da_proxy", "x_phys", "gap_to_eq"]


def equilibrium_conversion(T, P_bar, ratio):
    """Equilibrium CO conversion (fraction) for CO + 2H2 <=> CH3OH.
    log10(Kp) = 5139/T - 12.621 (Kp in atm^-2). Basis: 1 mol CO + `ratio` mol H2.
    Solves  X*(1+r-2X)^2 / ((1-X)*(r-2X)^2) = Kp*P^2  by vectorised bisection."""
    T = np.asarray(T, dtype=float)
    P_atm = np.asarray(P_bar, dtype=float) * BAR_TO_ATM
    r = np.asarray(ratio, dtype=float)
    target = 10.0 ** (5139.0 / T - 12.621) * P_atm ** 2
    lo = np.zeros_like(T)
    hi = np.minimum(1.0, r / 2.0) - 1e-9
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        q = mid * (1.0 + r - 2.0 * mid) ** 2 / ((1.0 - mid) * (r - 2.0 * mid) ** 2)
        too_high = q > target
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def add_physics_features(df, use_activity):
    """Return a COPY of df with the physics columns added.
    use_activity=True  -> use the measured/known catalyst activity column 'act' (variant B)
    use_activity=False -> estimate activity from time-on-stream and temperature (variant A)"""
    out = df.copy()
    T = out["temp"].to_numpy(dtype=float)
    P = out["pres"].to_numpy(dtype=float)
    r = out["ratio"].to_numpy(dtype=float)

    x_eq = equilibrium_conversion(T, P, r)
    k = A_PRE * np.exp(-EA_J_MOL / (R_GAS * T))

    if use_activity:
        act = out["act"].to_numpy(dtype=float)
    else:
        kd = KD_REF_PER_H * np.exp(-(ED_J_MOL / R_GAS) * (1.0 / T - 1.0 / T_REF_K))
        act = np.exp(-kd * out["time"].to_numpy(dtype=float))

    da = k * act * (P / P_REF_BAR) ** P_ORDER
    x_phys = x_eq * (1.0 - np.exp(-da))

    out["x_eq"], out["k_arr"], out["da_proxy"] = x_eq, k, da
    out["x_phys"], out["gap_to_eq"] = x_phys, x_eq - x_phys
    return out


if __name__ == "__main__":      # quick self-test:  python physics_features.py
    demo = pd.DataFrame({"time": [0, 0, 0], "temp": [500.0, 525.0, 580.0], "pres": [77.0] * 3,
                         "ratio": [2.2] * 3, "act": [1.0] * 3})
    print(add_physics_features(demo, True)[["temp", *PHYS_COLS]].round(3).to_string(index=False))