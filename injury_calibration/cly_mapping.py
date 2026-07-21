"""
Map nominal extracellular concentration -> lysosomal load C_ly, via the frozen v0.4
accumulation model.

This is the core step of the injury calibration: it moves the dose-response off the
nominal-concentration axis and onto the lysosomal-load axis, so the injury function
can be fitted as f(C_ly) rather than f(C_ext).

Protocol matches Jarzina's readouts: CONSTANT extracellular exposure for 24 h, single
timepoint. Units: C_ext in uM (model input), C_ly in nM (model output).

Each cell line is mapped through its OWN uptake parameterisation. Never map rat data
through human uptake or vice versa -- the human/rat sensitivity gap is uptake-driven,
so mixing them would corrupt the collapse test that is the point of the exercise.
"""

import functools

import numpy as np
from scipy.interpolate import PchipInterpolator

import model_core_v04 as mc

EXPOSURE_H = 24.0        # Jarzina readouts are all 24 h single-timepoint
N_TIME = 241             # 6-min resolution over the exposure window (for AUC/peak)

# Which frozen fit each cell line uses. NRK has two published fits; "2A" (k_deg shared
# with RPTEC) is the default because it keeps "uptake alone explains the species gap"
# as the testable hypothesis behind the collapse test. "2B" (k_deg free) fits the NRK
# washout better but attributes part of the gap to degradation instead. In practice the
# choice rescales C_ly by a near-constant ~1.12x at every dose and K_m, so it moves
# C_ly50_NRK ~12% -- reported as a sensitivity row, not a separate headline.
NRK_FITS = {
    "2A": dict(k_uptake="k_uptake_NRK",      k_deg="k_deg_PB"),
    "2B": dict(k_uptake="k_uptake_NRK_free", k_deg="k_deg_NRK_free"),
}
DEFAULT_NRK_FIT = "2A"

# Regression guard: total intracellular conc (nM) predicted for RPTEC at the 34 uM
# calibration dose, 24 h, K_m = 200, nominal k_rec. Recomputed from the frozen v0.4
# fit -- if an import breaks or a constant drifts, this trips.
#
# NOTE this pins TOTAL intracellular, not C_ly. C_ly itself was never measured (Jarzina
# assayed total by LC-MS/MS), so there is no "calibration C_ly" to check against; a
# guard on C_ly would only be pinning the model to itself.
CAL_TOTAL_NM_RPTEC = 3907.0
CAL_TOTAL_TOL = 0.01     # relative


def _params(cell_line, K_m, nrk_fit=DEFAULT_NRK_FIT):
    """Frozen v0.4 parameters for a cell line at a given K_m (V_max re-anchored)."""
    if cell_line == "RPTEC/TERT1":
        k_uptake, k_deg = mc.FITTED["k_uptake_RPTEC"], mc.FITTED["k_deg_PB"]
    elif cell_line == "NRK-52E":
        spec = NRK_FITS[nrk_fit]
        k_uptake, k_deg = mc.FITTED[spec["k_uptake"]], mc.FITTED[spec["k_deg"]]
    else:
        raise ValueError(f"unknown cell line: {cell_line!r}")
    # mc.make_params applies V_max = k_uptake * (K_m + 34), keeping the 34 uM
    # calibration exact for every K_m.
    return mc.make_params(k_uptake, K_m, k_deg)


# ---------------------------------------------------------------------------
# C_ly summary statistics -- swappable. Each takes the 0-24 h trajectory.
# ---------------------------------------------------------------------------
def cly_at_24h(t_h, cly):
    """Instantaneous lysosomal load at the 24 h readout. THE DEFAULT."""
    return float(cly[-1])


def cly_peak(t_h, cly):
    """
    Peak lysosomal load over the exposure window. Under a constant exposure C_ly rises
    monotonically, so this is degenerate with cly_at_24h -- it differs only once the
    protocol includes a washout. Kept for interface symmetry.
    """
    return float(np.max(cly))


def cly_auc(t_h, cly):
    """Cumulative lysosomal exposure, integral of C_ly dt over 0-24 h (nM.h)."""
    return float(np.trapezoid(cly, t_h))


SUMMARY_STATS = {"at_24h": cly_at_24h, "peak": cly_peak, "auc": cly_auc}
DEFAULT_SUMMARY = "at_24h"


@functools.lru_cache(maxsize=8192)
def cly_for_dose(cell_line, dose_uM, K_m, summary=DEFAULT_SUMMARY,
                 nrk_fit=DEFAULT_NRK_FIT):
    """
    Lysosomal load (nM) after a constant `dose_uM` exposure for 24 h.

    Cached because the K_m sweep, the ratio root-find, and the bootstrap all revisit
    the same (cell_line, dose, K_m) combinations many times over.
    """
    p = _params(cell_line, K_m, nrk_fit=nrk_fit)
    t_eval = np.linspace(0.0, EXPOSURE_H, N_TIME)
    # sim_end == t_washout means the whole simulated window is the exposure phase.
    _, _, comps = mc.simulate(p, mc.V_CELL[cell_line], dose_uM, t_eval,
                              t_washout_h=EXPOSURE_H, sim_end_h=EXPOSURE_H)
    return SUMMARY_STATS[summary](t_eval, comps["C_ly"])


def map_doses(cell_line, doses_uM, K_m, summary=DEFAULT_SUMMARY,
              nrk_fit=DEFAULT_NRK_FIT):
    """Vector form of cly_for_dose. Returns C_ly (nM) for each nominal dose."""
    return np.array([cly_for_dose(cell_line, float(d), float(K_m), summary, nrk_fit)
                     for d in doses_uM])


# ---------------------------------------------------------------------------
# Sanity checks (Step 2 of the spec)
# ---------------------------------------------------------------------------
def check_calibration_guard(K_m=mc.KM_DEFAULT):
    """Assert the imported model still reproduces the frozen 34 uM RPTEC total."""
    p = _params("RPTEC/TERT1", K_m)
    _, total, _ = mc.simulate(p, mc.V_CELL["RPTEC/TERT1"], mc.CAL_CONC, [EXPOSURE_H],
                              t_washout_h=EXPOSURE_H, sim_end_h=EXPOSURE_H)
    got = float(total[0])
    rel = abs(got - CAL_TOTAL_NM_RPTEC) / CAL_TOTAL_NM_RPTEC
    assert rel < CAL_TOTAL_TOL, (
        f"calibration guard FAILED: RPTEC total at {mc.CAL_CONC} uM / 24 h = {got:.1f} nM, "
        f"expected ~{CAL_TOTAL_NM_RPTEC:.0f} nM ({rel:.1%} off). The accumulation model "
        f"or its frozen parameters have changed.")
    return got


def check_monotonic(cell_line, doses_uM, K_m, summary=DEFAULT_SUMMARY,
                    nrk_fit=DEFAULT_NRK_FIT):
    """Assert C_ly is monotonic non-decreasing in C_ext (required for the C_ly axis)."""
    cly = map_doses(cell_line, doses_uM, K_m, summary, nrk_fit)
    d = np.diff(cly)
    assert np.all(d >= -1e-9), (
        f"C_ly not monotonic in C_ext for {cell_line} at K_m={K_m}: {cly}")
    return cly


# ---------------------------------------------------------------------------
# Fast C_ly(K_m) interpolation, for the continuous ratio root-find
# ---------------------------------------------------------------------------
def build_cly_interpolator(cell_line, doses_uM, K_m_grid, summary=DEFAULT_SUMMARY,
                           nrk_fit=DEFAULT_NRK_FIT):
    """
    Precompute C_ly on a dense K_m grid and return a callable K_m -> C_ly vector.

    The crossing-K_m root-find needs C_ly at arbitrary (non-grid) K_m. Solving the ODEs
    inside every root-find iteration is wasteful, so instead we solve once per grid K_m
    and interpolate in log-K_m with a shape-preserving PCHIP (monotone, no overshoot).
    """
    K_m_grid = np.asarray(K_m_grid, dtype=float)
    table = np.array([map_doses(cell_line, doses_uM, km, summary, nrk_fit)
                      for km in K_m_grid])           # (n_Km, n_doses)
    interp = PchipInterpolator(np.log10(K_m_grid), table, axis=0, extrapolate=False)

    def cly_at(K_m):
        out = interp(np.log10(float(K_m)))
        if np.any(np.isnan(out)):
            raise ValueError(f"K_m={K_m} outside interpolation grid "
                             f"[{K_m_grid.min()}, {K_m_grid.max()}]")
        return np.asarray(out, dtype=float)

    cly_at.K_m_grid = K_m_grid
    cly_at.table = table
    return cly_at


def saturation_decades(cly, viability, lo=10.0, hi=90.0):
    """
    Saturation diagnostic: how many decades of C_ly the viability transition spans.

    Takes the points whose viability lies in (lo, hi) -- the transition region -- plus
    the points immediately bracketing it, and returns log10(max/min) of their C_ly. A
    small value means many different viabilities collapse onto near-identical lysosomal
    loads, i.e. C_ly50 is poorly identified and non-lysosomal killing mechanisms are
    likely contributing at high dose. Expected to be worst at low K_m.
    """
    cly = np.asarray(cly, dtype=float)
    viability = np.asarray(viability, dtype=float)
    inside = np.where((viability > lo) & (viability < hi))[0]
    if inside.size == 0:
        # No point lands inside the transition; fall back to the bracketing pair.
        below = np.where(viability <= lo)[0]
        above = np.where(viability >= hi)[0]
        if below.size == 0 or above.size == 0:
            return float("nan")
        idx = np.array([above[-1], below[0]])
    else:
        idx = np.arange(max(inside[0] - 1, 0), min(inside[-1] + 2, cly.size))
    span = cly[idx]
    if span.min() <= 0:
        return float("nan")
    return float(np.log10(span.max() / span.min()))
