"""
Shared core for the endo-lysosomal drug accumulation model -- v0.4.

Mirror of model_core.py but with the v0.4 Shipman et al. (2022) trafficking rate
constants and the v0.4 refit results. Kept as a separate module so the v0.2/v0.3
backend (model_core.py) stays intact for comparison.

This module is the live backend: app.py and the v0.4 analysis scripts import it,
and it holds the canonical frozen v0.4 fitted constants. model_core.py is the
retained v0.3 backend, kept for version-to-version comparison only.

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
    "k_uptake_RPTEC":    2.6350e-07,  # fmol/cell/min/uM  (Step 1, nominal k_rec)
    "k_uptake_NRK":      3.3730e-08,  # fmol/cell/min/uM  (Step 2A, k_deg fixed = k_deg_PB)
    "k_deg_PB":          2.4450e-04,  # /min  (Step 1, shared with Step 2A)
    "k_uptake_NRK_free": 7.8950e-08,  # fmol/cell/min/uM  (Step 2B, NRK k_deg free)
    "k_deg_NRK_free":    1.6280e-03,  # /min  (Step 2B, NRK k_deg free)
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


def v_max(k_uptake, K_m, c_cal=CAL_CONC):
    """
    Back-calculate V_max so that saturating uptake reproduces the fitted linear
    uptake AT THE CALIBRATION CONCENTRATION c_cal, for any K_m.

    k_uptake is fitted in the linear regime (uptake = k_uptake * C_ext) against the
    single-dose (34 uM) Jarzina data, so the calibration is anchored at c_cal, not at
    C_ext -> 0. Setting V_max = k_uptake * (K_m + c_cal) makes
        uptake(c_cal) = V_max * c_cal / (K_m + c_cal) = k_uptake * c_cal
    hold exactly for every K_m. The earlier V_max = k_uptake * K_m instead pinned the
    C_ext -> 0 tangent, which undershot the calibration by K_m/(K_m + c_cal) (~15% at
    K_m = 200 uM) because 34 uM is not << K_m. K_m still bends the dose-response above
    c_cal; only the anchor point moved from 0 to the calibration dose.
    """
    return k_uptake * (K_m + c_cal)


def odes(t, y, p, c_ext):
    """
    RHS of the 3-compartment system. y = [C_ee, C_le, C_ly] in fmol/cell.

    t is in MINUTES (the integration time base, matching the /min rate constants).
    c_ext is either a scalar (uM) or a callable taking time in HOURS -> uM, so that
    time-varying exposures (e.g. a PBPK-derived C_ext(t)) can drive the model.
    """
    C_ee, C_le, C_ly = y
    c = c_ext(t / 60.0) if callable(c_ext) else c_ext
    uptake = p["V_max"] * c / (p["K_m"] + c)
    dC_ee = uptake            - (p["k_mat"] + p["k_rec"]) * C_ee
    dC_le = p["k_mat"] * C_ee - (p["k_fuse"] + p["k_esc"]) * C_le
    dC_ly = p["k_fuse"] * C_le - p["k_deg"] * C_ly
    return [dC_ee, dC_le, dC_ly]


def _simulate_segments(p, V_cell, segments, t_eval_h):
    """
    Integrate the system piecewise over `segments`, carrying state across boundaries.

    segments: list of (t0_h, t1_h, c_ext), where c_ext is a scalar (uM) or a callable
    of time-in-hours. Splitting at segment edges is what keeps the stiff solver from
    stepping straight over a discontinuity in the exposure (e.g. a washout or a
    dosing event) -- BDF will happily miss a jump it never evaluates at.

    Returns (t_eval_h, total_nM, compartments_nM).
    """
    t_eval_h = np.atleast_1d(np.asarray(t_eval_h, dtype=float))
    t_eval_min = t_eval_h * 60.0
    Y = np.zeros((3, t_eval_min.size))
    y0 = np.array([0.0, 0.0, 0.0])
    n_seg = len(segments)

    for i, (t0_h, t1_h, c_ext) in enumerate(segments):
        t0, t1 = t0_h * 60.0, t1_h * 60.0
        if t1 <= t0:
            continue
        sol = solve_ivp(odes, (t0, t1), y0, args=(p, c_ext), method="BDF",
                        dense_output=True, rtol=1e-8, atol=1e-12)
        # First segment owns its left edge; later segments take (t0, t1]. The final
        # segment has no upper bound, so it also covers any t_eval at sim_end.
        lo_ok = (t_eval_min >= t0) if i == 0 else (t_eval_min > t0)
        hi_ok = np.ones_like(lo_ok) if i == n_seg - 1 else (t_eval_min <= t1)
        mask = lo_ok & hi_ok
        if mask.any():
            Y[:, mask] = sol.sol(t_eval_min[mask])
        y0 = sol.y[:, -1]

    Y = np.clip(Y, 0.0, None)
    comp_nM = Y * (1e-6 / V_cell)          # fmol/cell -> nM
    total_nM = comp_nM.sum(axis=0)
    return t_eval_h, total_nM, dict(C_ee=comp_nM[0], C_le=comp_nM[1], C_ly=comp_nM[2])


def make_params(k_uptake, K_m, k_deg, fixed=None):
    p = dict(FIXED_DEFAULT if fixed is None else fixed)
    p["V_max"] = v_max(k_uptake, K_m)
    p["K_m"] = K_m
    p["k_deg"] = k_deg
    return p


def simulate(p, V_cell, c_ext_um, t_eval_h, t_washout_h=24.0, sim_end_h=50.0):
    """
    Step-exposure protocol: constant c_ext_um over (0 -> t_washout_h), then washout
    (t_washout_h -> sim_end_h at C_ext=0). This is the Jarzina calibration protocol
    and a special case of simulate_profile(); it passes a scalar per segment, so its
    numerics are identical to the pre-refactor implementation.

    Returns (t_eval_h, total_nM, compartments_nM).
    """
    segments = [(0.0, t_washout_h, c_ext_um),
                (t_washout_h, sim_end_h, 0.0)]
    return _simulate_segments(p, V_cell, segments, t_eval_h)


def simulate_profile(p, V_cell, c_ext_fn, t_eval_h, sim_end_h=None, breakpoints=()):
    """
    Simulate an arbitrary time-varying extracellular exposure.

    c_ext_fn   : callable, time in HOURS -> extracellular concentration in uM.
                 e.g. a PBPK trace wrapped in scipy.interpolate.PchipInterpolator,
                 or an analytic PK profile.
    breakpoints: times (h) at which c_ext_fn is discontinuous (dosing events, washout).
                 Integration is split there so the solver cannot step over the jump.
                 Smooth profiles need none.

    Nothing about the calibration changes here: k_uptake and k_deg are properties of
    the cell and drug, not of the exposure profile, so no refit is required.
    """
    t_eval_h = np.atleast_1d(np.asarray(t_eval_h, dtype=float))
    if sim_end_h is None:
        sim_end_h = float(t_eval_h.max())
    edges = [0.0] + sorted(float(b) for b in breakpoints if 0.0 < b < sim_end_h) + [sim_end_h]
    segments = [(edges[i], edges[i + 1], c_ext_fn) for i in range(len(edges) - 1)]
    return _simulate_segments(p, V_cell, segments, t_eval_h)


# ---------------------------------------------------------------------------
# Analytic extracellular-exposure profiles.
#
# Each factory returns a callable C_ext(t) giving concentration in uM for a time
# (or array of times) in HOURS -- exactly the c_ext_fn contract simulate_profile()
# expects. The returned callable also carries a `.breakpoints` tuple listing the
# times (h) where the profile is non-smooth (dose events, infusion stop, washout);
# pass it straight through as simulate_profile(..., breakpoints=fn.breakpoints) so
# the stiff BDF solver is forced to evaluate the kink instead of stepping over it.
# Smooth profiles carry an empty tuple.
#
# No refit is needed to use any of these: k_uptake and k_deg are properties of the
# cell and drug, not of the exposure (see simulate_profile docstring).
# ---------------------------------------------------------------------------
def pk_step(C0, t_off_h, t_on_h=0.0):
    """
    Square pulse: constant C0 (uM) on [t_on, t_off), zero outside. This is the
    profile form of the Jarzina step-exposure protocol -- pk_step(C0, t_washout)
    driven through simulate_profile() reproduces simulate(C0, t_washout_h=...).
    """
    C0 = float(C0); t_on = float(t_on_h); t_off = float(t_off_h)

    def fn(t_h):
        t = np.asarray(t_h, dtype=float)
        return np.where((t >= t_on) & (t < t_off), C0, 0.0)

    fn.breakpoints = tuple(b for b in (t_on, t_off) if b > 0.0)
    return fn


def pk_bolus(C0, k_e):
    """
    IV-bolus exposure: instantaneous rise to C0 (uM) at t=0, then first-order decay
    at rate k_e (/h):  C_ext(t) = C0 * exp(-k_e * t). Smooth for t > 0, so no
    breakpoints. AUC over [0, inf) = C0 / k_e; k_e = 0 recovers a constant C0.
    """
    C0 = float(C0); k_e = float(k_e)

    def fn(t_h):
        return C0 * np.exp(-k_e * np.asarray(t_h, dtype=float))

    fn.breakpoints = ()
    return fn


def pk_bolus_train(C0, k_e, tau_h, n_doses):
    """
    Repeated IV boluses: dose C0 (uM) every tau_h hours, n_doses times, each decaying
    at k_e (/h). C_ext is the superposition of the doses given so far. The dose at
    t=0 sets the initial spike; interior dose times (tau, 2*tau, ...) are breakpoints.
    """
    C0 = float(C0); k_e = float(k_e); tau = float(tau_h); n = int(n_doses)

    def fn(t_h):
        t = np.asarray(t_h, dtype=float)
        out = np.zeros_like(t)
        for i in range(n):
            ti = i * tau
            out = out + np.where(t >= ti, C0 * np.exp(-k_e * (t - ti)), 0.0)
        return out

    fn.breakpoints = tuple(i * tau for i in range(1, n))
    return fn


def pk_infusion(C_ss, k_e, t_off_h, t_on_h=0.0):
    """
    Zero-order infusion to plateau C_ss (uM), then first-order washout. On
    [t_on, t_off]:  C = C_ss * (1 - exp(-k_e * (t - t_on))). After t_off it decays
    from the value reached at t_off at rate k_e (/h). Breakpoints at t_off (and t_on
    if positive) -- the infusion-stop kink is exactly where the solver must not skip.
    """
    C_ss = float(C_ss); k_e = float(k_e); t_on = float(t_on_h); t_off = float(t_off_h)
    c_off = C_ss * (1.0 - np.exp(-k_e * (t_off - t_on)))

    def fn(t_h):
        t = np.asarray(t_h, dtype=float)
        rising = C_ss * (1.0 - np.exp(-k_e * np.clip(t - t_on, 0.0, None)))
        decaying = c_off * np.exp(-k_e * np.clip(t - t_off, 0.0, None))
        return np.where(t < t_on, 0.0, np.where(t <= t_off, rising, decaying))

    fn.breakpoints = tuple(b for b in (t_on, t_off) if b > 0.0)
    return fn


def first_crossing_h(t, y, thr):
    above = np.where(y >= thr)[0]
    return float(t[above[0]]) if above.size else None
