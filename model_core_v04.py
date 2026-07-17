"""
Shared core for the endo-lysosomal drug accumulation model -- v0.4.

Mirror of model_core.py but with the v0.4 Shipman et al. (2022) trafficking rate
constants and the v0.4 refit results. Kept as a separate module so the v0.2/v0.3
backend (model_core.py) stays intact for comparison.

NOTE: app.py currently imports model_core (the v0.3 backend) and is deliberately
NOT rewired to this module yet -- per instruction, the Streamlit app is left
unchanged in this version. This file exists to (a) record the canonical v0.4
fitted constants and (b) make the eventual app switch a one-line import change.

v0.4 change (prompts/model_v04_updated_instructions.md): fixed trafficking rates
updated to proximal-tubule-specific values from Shipman et al. (2022), a
5-compartment OK-cell megalin-trafficking model (AEE=EE, AV/Rab7=LE, Lys=LY,
Rab11a-DAT recycling -> our k_rec efflux):

    k_mat  0.04  -> 0.048  /min   (Shipman k_m,1)
    k_fuse 0.02  -> 0.0094 /min   (Shipman k_m,2)
    k_rec  0.02  -> 0.02 nominal, uncertainty band 0.02-0.046 /min (upper = k_DAT,f)
    k_esc  kept 0.0002 (Gilleron)  -- receptor doesn't escape; cargo committed to LY
    k_deg  still drug-fitted        -- cargo chemistry, not a compartment property

Model equations (unchanged from v0.3):
    dC_ee/dt = V_max*C_ext/(K_m+C_ext) - (k_mat + k_rec) * C_ee
    dC_le/dt = k_mat * C_ee            - (k_fuse + k_esc) * C_le
    dC_ly/dt = k_fuse * C_le           - k_deg * C_ly
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Fixed trafficking parameters (cell machinery, not drug) -- v0.4 Shipman rates
# ---------------------------------------------------------------------------
K_REC_NOMINAL = 0.02      # /min  band lower bound (= old v0.2 value)
K_REC_UPPER   = 0.046     # /min  Shipman k_DAT,f fast EE recycling (band upper bound)
FIXED_DEFAULT = dict(k_mat=0.048, k_rec=K_REC_NOMINAL, k_fuse=0.0094, k_esc=0.0002)

# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------
V_CELL = {
    "RPTEC/TERT1": 1.99e-12,  # L, d ~ 15.6 um
    "NRK-52E":     1.21e-12,  # L, d ~ 13.2 um
}

# ---------------------------------------------------------------------------
# Frozen v0.4 fitted parameters (linear-regime k_uptake, k_deg) from
# endo_lyso_model_v04.py Steps 1 (RPTEC) and 2A (NRK, k_deg fixed), at the
# nominal k_rec = 0.02. The k_rec band 0.02-0.046 maps k_uptake_RPTEC onto
# 2.635e-07 -- 3.597e-07 (see K_UPTAKE_RPTEC_BAND).
# ---------------------------------------------------------------------------
FITTED = {
    "k_uptake_RPTEC": 2.6350e-07,  # fmol/cell/min/uM  (Step 1, nominal k_rec)
    "k_uptake_NRK":   3.3730e-08,  # fmol/cell/min/uM  (Step 2A, k_deg fixed)
    "k_deg_PB":       2.4450e-04,  # /min  (Step 1, shared with Step 2A)
    "k_deg_NRK_free": 1.6280e-03,  # /min  (Step 2B, NRK k_deg free -- reference only)
}
K_UPTAKE_RPTEC_BAND = (2.6350e-07, 3.5970e-07)  # k_rec 0.02 -> 0.046

KM_DEFAULT = 200.0        # uM, v0.3 mid-range default
CAL_CONC   = 34.0         # uM, calibration concentration

# ---------------------------------------------------------------------------
# Digitised calibration data (Jarzina et al. 2022 Fig 6A), v0.2 replacement
# ---------------------------------------------------------------------------
T_ALL_H = np.array([0.0, 3.53, 6.44, 24.0, 48.0])

RPTEC_CENTRAL = np.array([0.0, 636.0, 1377.0, 3558.0, 2792.0])
RPTEC_LOWER   = np.array([0.0, 442.0,  870.0, 1974.0, 1494.0])
RPTEC_UPPER   = np.array([0.0, 883.0, 1909.0, 5156.0, 4065.0])

NRK_CENTRAL   = np.array([0.0, 299.0, 662.0, 831.0, 104.0])


def v_max(k_uptake, K_m):
    """Back-calculate V_max so that at C_ext << K_m, uptake ~= k_uptake * C_ext."""
    return k_uptake * K_m


def odes(t, y, p, c_ext):
    """RHS of the 3-compartment system. y = [C_ee, C_le, C_ly] in fmol/cell."""
    C_ee, C_le, C_ly = y
    uptake = p["V_max"] * c_ext / (p["K_m"] + c_ext)
    dC_ee = uptake            - (p["k_mat"] + p["k_rec"]) * C_ee
    dC_le = p["k_mat"] * C_ee - (p["k_fuse"] + p["k_esc"]) * C_le
    dC_ly = p["k_fuse"] * C_le - p["k_deg"] * C_ly
    return [dC_ee, dC_le, dC_ly]


def make_params(k_uptake, K_m, k_deg, fixed=None):
    p = dict(FIXED_DEFAULT if fixed is None else fixed)
    p["V_max"] = v_max(k_uptake, K_m)
    p["K_m"] = K_m
    p["k_deg"] = k_deg
    return p


def simulate(p, V_cell, c_ext_um, t_eval_h, t_washout_h=24.0, sim_end_h=50.0):
    """
    Simulate exposure (0 -> t_washout_h at c_ext_um) then washout
    (t_washout_h -> sim_end_h at C_ext=0). Returns (t_eval_h, total_nM, compartments_nM).
    """
    t_eval_h = np.atleast_1d(np.asarray(t_eval_h, dtype=float))
    t_eval_min = t_eval_h * 60.0
    t_washout_min = t_washout_h * 60.0
    sim_end_min = sim_end_h * 60.0

    mask1 = t_eval_min <= t_washout_min
    mask2 = ~mask1

    sol1 = solve_ivp(odes, (0.0, t_washout_min), [0.0, 0.0, 0.0],
                      args=(p, c_ext_um), method="BDF", dense_output=True,
                      rtol=1e-8, atol=1e-12)
    Y = np.zeros((3, t_eval_min.size))
    if mask1.any():
        Y[:, mask1] = sol1.sol(t_eval_min[mask1])

    if mask2.any():
        sol2 = solve_ivp(odes, (t_washout_min, sim_end_min), sol1.y[:, -1],
                          args=(p, 0.0), method="BDF", dense_output=True,
                          rtol=1e-8, atol=1e-12)
        Y[:, mask2] = sol2.sol(t_eval_min[mask2])

    Y = np.clip(Y, 0.0, None)
    comp_nM = Y * (1e-6 / V_cell)          # fmol/cell -> nM
    total_nM = comp_nM.sum(axis=0)
    compartments = dict(C_ee=comp_nM[0], C_le=comp_nM[1], C_ly=comp_nM[2])
    return t_eval_h, total_nM, compartments


def first_crossing_h(t, y, thr):
    above = np.where(y >= thr)[0]
    return float(t[above[0]]) if above.size else None
