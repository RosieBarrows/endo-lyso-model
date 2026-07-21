"""
Tests for the injury calibration (injury_calibration/).

The load-bearing tests are:
  * test_calibration_guard          -- the imported accumulation model still reproduces
                                       the frozen 34 uM RPTEC total (catches import drift)
  * test_hill_recovers_known_params -- the fitter recovers parameters from synthetic data,
                                       so a wrong C_ly50 can be blamed on the data, not
                                       the optimiser
  * test_nominal_axis_recovers_jarzina_ec50 -- fitting on the nominal axis reproduces
                                       Jarzina's published EC50s, the end-to-end check
                                       that the machinery is sound before the C_ly-axis
                                       results are trusted
  * test_abs_midpoint_matches_numeric -- the closed-form C_ly50_abs agrees with a numeric
                                       solve, including with a non-zero floor

No pytest required:

    .venv/bin/python test_injury_calibration.py

Functions are named test_* so pytest will also collect them if installed.
"""

import numpy as np
from scipy.optimize import brentq

import model_core_v04 as mc
from injury_calibration import cly_mapping as cm, data_io, hill


# ---------------------------------------------------------------------------
# Accumulation-model interface
# ---------------------------------------------------------------------------
def test_calibration_guard():
    """The imported v0.4 model still reproduces the frozen 34 uM / 24 h RPTEC total."""
    total = cm.check_calibration_guard()
    assert abs(total - cm.CAL_TOTAL_NM_RPTEC) / cm.CAL_TOTAL_NM_RPTEC < cm.CAL_TOTAL_TOL


def test_vmax_anchoring_holds_at_calibration_dose():
    """
    V_max = k_uptake*(K_m+34) must keep uptake at 34 uM identical for every K_m.
    This is the property the whole K_m sweep rests on: if it broke, sweeping K_m would
    silently move the calibration rather than only the extrapolation.
    """
    ku = mc.FITTED["k_uptake_RPTEC"]
    uptakes = []
    for K_m in (50.0, 200.0, 2000.0):
        p = mc.make_params(ku, K_m, mc.FITTED["k_deg_PB"])
        uptakes.append(p["V_max"] * mc.CAL_CONC / (p["K_m"] + mc.CAL_CONC))
    assert np.allclose(uptakes, uptakes[0], rtol=1e-12), \
        f"uptake at the calibration dose drifts with K_m: {uptakes}"


def test_cly_monotonic_in_dose():
    """C_ly must be non-decreasing in C_ext for the C_ly axis to be well-ordered."""
    doses = [10.0, 34.0, 62.5, 125.0, 250.0, 500.0, 1000.0]
    for cl in ("RPTEC/TERT1", "NRK-52E"):
        for K_m in (50.0, 200.0, 2000.0):
            cly = cm.map_doses(cl, doses, K_m)
            assert np.all(np.diff(cly) >= -1e-9), f"{cl} K_m={K_m}: {cly}"


def test_summary_stats_ordering():
    """AUC >> instantaneous load, and peak == 24 h value under a monotonic exposure."""
    d, K_m = 125.0, 200.0
    at24 = cm.cly_for_dose("RPTEC/TERT1", d, K_m, summary="at_24h")
    peak = cm.cly_for_dose("RPTEC/TERT1", d, K_m, summary="peak")
    auc = cm.cly_for_dose("RPTEC/TERT1", d, K_m, summary="auc")
    assert np.isclose(at24, peak, rtol=1e-9), "peak should equal the 24 h value"
    assert 0 < auc < at24 * 24.0, "AUC must lie below the rectangle at24 * 24 h"


def test_cly_interpolator_matches_direct_solve():
    """The K_m interpolator backing the root-find must agree with solving directly."""
    doses = np.array([62.5, 250.0, 1000.0])
    grid = np.logspace(np.log10(20.0), np.log10(5000.0), 40)
    interp = cm.build_cly_interpolator("RPTEC/TERT1", doses, grid)
    for K_m in (77.0, 340.0, 1234.0):          # deliberately off-grid
        direct = cm.map_doses("RPTEC/TERT1", doses, K_m)
        assert np.allclose(interp(K_m), direct, rtol=2e-3), \
            f"interpolation error at K_m={K_m}: {interp(K_m)} vs {direct}"


# ---------------------------------------------------------------------------
# Hill fitting
# ---------------------------------------------------------------------------
def test_hill_recovers_known_params():
    """Fit synthetic noiseless data and recover the generating C50 and n."""
    C50_true, n_true = 4200.0, 3.4
    cly = np.array([500.0, 1500.0, 3000.0, 5000.0, 9000.0, 20000.0])
    y = hill.hill(cly, C50_true, n_true)
    f = hill.fit_viability(cly, y, sd=np.ones_like(y))
    assert abs(f["C_ly50_hill"] - C50_true) / C50_true < 1e-3, f["C_ly50_hill"]
    assert abs(f["n"] - n_true) / n_true < 1e-3, f["n"]
    assert np.isclose(f["C_ly50_abs"], f["C_ly50_hill"]), "no floor -> abs == hill"


def test_free_floor_recovers_known_floor():
    """With a genuine non-zero floor in the data, the free-bottom fit finds it."""
    C50_true, n_true, b_true = 3000.0, 2.5, 12.0
    cly = np.array([300.0, 1000.0, 2500.0, 5000.0, 12000.0, 40000.0])
    y = hill.hill(cly, C50_true, n_true, bottom=b_true)
    f = hill.fit_viability(cly, y, sd=np.ones_like(y), free_bottom=True)
    assert abs(f["bottom"] - b_true) < 0.5, f["bottom"]
    assert abs(f["C_ly50_hill"] - C50_true) / C50_true < 1e-2, f["C_ly50_hill"]
    # With a floor at 12, the Hill midpoint sits at 56% viability, NOT 50% -- so the
    # absolute midpoint must be strictly larger than the Hill parameter.
    assert f["C_ly50_abs"] > f["C_ly50_hill"], \
        "a non-zero floor must push the 50%-absolute load above the Hill midpoint"


def test_abs_midpoint_matches_numeric():
    """The closed-form abs_midpoint agrees with numerically solving viability = 50."""
    for C50, n, b in [(4000.0, 3.0, 0.0), (2500.0, 1.8, 15.0), (900.0, 6.0, 5.0)]:
        analytic = hill.abs_midpoint(C50, n, b)
        numeric = brentq(lambda c: hill.hill(c, C50, n, b) - 50.0, 1e-3, 1e9)
        assert abs(analytic - numeric) / numeric < 1e-6, (C50, n, b, analytic, numeric)


def test_abs_midpoint_undefined_above_floor():
    """If the fitted floor sits above 50%, there is no 50% crossing -- must be NaN."""
    assert np.isnan(hill.abs_midpoint(1000.0, 2.0, bottom=60.0))


def test_weighting_follows_sd():
    """Variance weighting must pull the fit toward the low-SD points."""
    cly = np.array([1000.0, 3000.0, 6000.0, 12000.0])
    y = np.array([95.0, 60.0, 25.0, 5.0])
    tight = hill.fit_viability(cly, y, sd=np.array([20.0, 0.5, 20.0, 20.0]))
    pred = hill.hill(cly, tight["C_ly50_hill"], tight["n"], tight["bottom"])
    err = np.abs(pred - y)
    assert err[1] == err.min(), \
        f"the 1/sd^2-weighted point should be fit best, got residuals {err}"


def test_unweighted_when_sds_missing():
    """An arm with NaN SDs must fall back to an unweighted fit rather than crashing."""
    cly = np.array([1000.0, 3000.0, 6000.0])
    y = np.array([90.0, 55.0, 20.0])
    f = hill.fit_viability(cly, y, sd=np.array([np.nan, np.nan, np.nan]))
    assert f["weighted"] is False
    assert np.isfinite(f["C_ly50_abs"])


def test_saturation_diagnostic_detects_collapse():
    """A transition spanning little C_ly must report few decades; a wide one, more."""
    viab = np.array([95.0, 60.0, 30.0, 5.0])
    tight = cm.saturation_decades(np.array([1000.0, 1100.0, 1200.0, 1300.0]), viab)
    wide = cm.saturation_decades(np.array([100.0, 1000.0, 10000.0, 100000.0]), viab)
    assert tight < 0.3, tight
    assert wide > 2.0, wide
    assert tight < wide


# ---------------------------------------------------------------------------
# End-to-end: the machinery must reproduce Jarzina's published EC50s
# ---------------------------------------------------------------------------
def test_nominal_axis_recovers_jarzina_ec50():
    """
    Fitting the same Hill on the NOMINAL axis must land near Jarzina's reported EC50s
    (~57 uM RPTEC, ~575 uM NRK). This is the check that validates the fitter end to end
    on real data before any C_ly-axis number is believed.
    """
    arms = data_io.load_all()
    expected = {"RPTEC/TERT1": 57.0, "NRK-52E": 575.0}
    for cl, ec50_ref in expected.items():
        arm = arms.get(("ke3_viability", cl))
        if arm is None:
            continue
        free = cl == "NRK-52E"
        f = hill.fit_viability(arm.conc_uM, arm.mean, sd=arm.sd, free_bottom=free)
        rel = abs(f["C_ly50_abs"] - ec50_ref) / ec50_ref
        assert rel < 0.15, \
            f"{cl}: fitted EC50 {f['C_ly50_abs']:.1f} uM vs Jarzina ~{ec50_ref} uM ({rel:.0%})"


def test_data_arms_load_and_are_clamped():
    """Viability arms load, are sorted, and carry no out-of-range values."""
    arms = data_io.load_all()
    assert ("ke3_viability", "RPTEC/TERT1") in arms, "primary human arm missing"
    for (ke, cl), arm in arms.items():
        assert np.all(np.diff(arm.conc_uM) > 0), f"{cl} {ke}: concentrations unsorted"
        if ke == "ke3_viability":
            assert arm.mean.min() >= 0.0 and arm.mean.max() <= 100.0, \
                f"{cl}: viability outside [0, 100] -- clamping failed"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
