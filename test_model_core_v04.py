"""
Regression + sanity tests for model_core_v04.

The load-bearing test is `test_step_matches_profile`: it pins the claim that the
new time-varying path (simulate_profile driven by a piecewise-constant pk_step)
reproduces the original constant-exposure simulate() bit-for-bit. That is the
guarantee that the v0.4 refactor -- which introduced callable c_ext and segment
splitting -- did not perturb the calibrated behaviour underneath everything else.

No pytest required: run directly with the project venv,

    .venv/bin/python test_model_core_v04.py

Functions are named test_* so pytest will also collect them if it is installed.
"""

import numpy as np

import model_core_v04 as mc


# Calibrated RPTEC parameters, straight from the frozen v0.4 fit (no refit here).
P = mc.make_params(mc.FITTED["k_uptake_RPTEC"], mc.KM_DEFAULT, mc.FITTED["k_deg_PB"])
V = mc.V_CELL["RPTEC/TERT1"]

# A dense grid that lands points before, across, and after the washout edge.
T_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 24.0, 400),
    np.linspace(24.0, 50.0, 300),
    [24.0],                       # exactly on the discontinuity
]))


def test_step_matches_profile():
    """simulate(step + washout) == simulate_profile(pk_step) to solver tolerance."""
    dose, t_washout, sim_end = 34.0, 24.0, 50.0

    t_a, tot_a, comp_a = mc.simulate(P, V, dose, T_GRID,
                                     t_washout_h=t_washout, sim_end_h=sim_end)

    step = mc.pk_step(dose, t_washout)
    t_b, tot_b, comp_b = mc.simulate_profile(P, V, step, T_GRID,
                                             sim_end_h=sim_end,
                                             breakpoints=step.breakpoints)

    # Absolute nM tolerance scaled to the signal size; both paths use the same
    # BDF settings, so agreement should be essentially to round-off.
    scale = max(tot_a.max(), 1.0)
    assert np.allclose(tot_a, tot_b, rtol=0, atol=1e-6 * scale), \
        f"total mismatch: max |d| = {np.abs(tot_a - tot_b).max():.3e} nM"
    for key in ("C_ee", "C_le", "C_ly"):
        assert np.allclose(comp_a[key], comp_b[key], rtol=0, atol=1e-6 * scale), \
            f"{key} mismatch: max |d| = {np.abs(comp_a[key] - comp_b[key]).max():.3e} nM"


def test_bolus_zero_ke_is_constant():
    """pk_bolus(C0, k_e=0) is a flat C0 exposure with no washout."""
    dose, sim_end = 34.0, 30.0
    bolus = mc.pk_bolus(dose, 0.0)
    assert bolus.breakpoints == ()
    ts = np.linspace(0.0, sim_end, 50)
    assert np.allclose(bolus(ts), dose)


def test_bolus_auc_matches_analytic():
    """Numeric AUC of pk_bolus over a long window ~= analytic C0 / k_e."""
    C0, k_e = 100.0, 0.2               # /h ; half-life ~3.5 h
    t = np.linspace(0.0, 200.0, 200_001)   # long enough that the tail is negligible
    auc_num = np.trapezoid(mc.pk_bolus(C0, k_e)(t), t)
    auc_exact = C0 / k_e
    assert abs(auc_num - auc_exact) / auc_exact < 1e-4, \
        f"AUC {auc_num:.4f} vs analytic {auc_exact:.4f}"


def test_infusion_shape_and_breakpoints():
    """pk_infusion rises to below C_ss, peaks at t_off, then decays; kink flagged."""
    C_ss, k_e, t_off = 50.0, 0.3, 12.0
    inf = mc.pk_infusion(C_ss, k_e, t_off)
    assert inf.breakpoints == (t_off,)
    assert inf(0.0) == 0.0 or np.isclose(inf(0.0), 0.0)
    peak = float(inf(t_off))
    assert 0.0 < peak < C_ss                      # plateau not fully reached
    assert inf(t_off + 5.0) < peak                # decaying after stop
    assert inf(t_off) >= inf(t_off - 1.0)         # still rising up to the stop


def test_bolus_train_breakpoints_and_superposition():
    """Dose train places interior breakpoints at dose times and sums doses."""
    C0, k_e, tau, n = 20.0, 0.1, 6.0, 4
    train = mc.pk_bolus_train(C0, k_e, tau, n)
    assert train.breakpoints == (6.0, 12.0, 18.0)
    # Just after the 2nd dose (t=6+), exposure = fresh dose + decayed 1st dose.
    t = 6.0 + 1e-6
    expected = C0 + C0 * np.exp(-k_e * t)
    assert np.isclose(float(train(t)), expected, rtol=1e-4)


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
