"""
Hill fitting for the injury function, on the C_ly axis.

Two conventions, chosen per arm by what the data actually reaches:

  RPTEC (viability):  top FIXED 100, bottom FIXED 0.
      The floor points (250/500/1000 uM -> 0.8, 0.0, 0.0 % of control) genuinely reach
      zero, so fixing the bottom is reading the data, not assuming it.

  NRK (viability):    top FIXED 100, bottom FREE in [0, 20].
      NRK's lowest observed viability is 23% at the highest dose tested -- the curve has
      NOT reached its floor in range. Fixing bottom = 0 would extrapolate below every
      observation and would bias this arm's C_ly50 and n. The [0, 20] bound keeps the
      free floor below the lowest observed point so it cannot run somewhere unphysical.
      This matters because C_ly50(NRK) is the load-bearing number in the human/rat
      collapse test, so its uncertainty must be honest rather than artificially tight.

Top is fixed at 100 in both cases because the readout is % of control: the foot of the
curve is fixed by construction, not something to be fitted.

TWO MIDPOINTS ARE REPORTED, and the distinction matters:
  * `C_ly50_hill` -- the fitted Hill midpoint parameter (what the optimiser estimates).
    With a free floor b, this is the load at viability (100 + b)/2, NOT at 50%.
  * `C_ly50_abs`  -- the load at 50% ABSOLUTE viability, in closed form from the fit.
    Identical to C_ly50_hill when b = 0.
Because the two arms use different floor conventions, only `C_ly50_abs` is the same
physical quantity in both, so it -- not the Hill parameter -- is what the human/rat
ratio, the crossing-K_m root-find, and the AOP ordering check are built on.
"""

import numpy as np
from scipy.optimize import least_squares

# n is bounded rather than free: where C_ly saturates across the viability transition
# (low K_m especially) the curve is near-vertical on the C_ly axis and n runs away.
# The bound keeps the optimiser well-posed; n at or near N_MAX is a saturation artefact
# and is flagged as such, never read as a cooperativity estimate.
N_MIN, N_MAX = 0.1, 50.0
N_SATURATION_FLAG = 10.0     # above this, report as saturation-driven

BOTTOM_MAX_NRK = 20.0        # free-floor upper bound (below NRK's lowest point, 23%)

BOOTSTRAP_B = 2000
RNG_SEED = 20260719


def hill(cly, C50, n, bottom=0.0, top=100.0):
    """Descending Hill: top at C_ly -> 0, `bottom` at C_ly -> inf."""
    cly = np.asarray(cly, dtype=float)
    # Guard the power: C_ly is strictly positive here, but keep it safe under bootstrap.
    ratio = np.clip(cly, 1e-12, None) / C50
    return bottom + (top - bottom) / (1.0 + ratio ** n)


def abs_midpoint(C50, n, bottom=0.0, top=100.0, level=50.0):
    """
    C_ly at a given ABSOLUTE viability `level`, closed form from the fitted Hill.

    Solving level = bottom + (top-bottom)/(1+x) for x = (C/C50)^n gives
        x = (top - level) / (level - bottom)
    so C = C50 * x**(1/n). Reduces to C50 exactly when bottom = 0 and level = 50.
    Returns NaN if the fitted curve never reaches `level` (floor above it).
    """
    if not (bottom < level < top):
        return float("nan")
    x = (top - level) / (level - bottom)
    return float(C50 * x ** (1.0 / n))


def _residuals(theta, cly, y, w, free_bottom):
    C50 = 10.0 ** theta[0]
    n = theta[1]
    bottom = theta[2] if free_bottom else 0.0
    return (hill(cly, C50, n, bottom) - y) * w


def _weights(sd, n_pts):
    """
    Variance weights ~ 1/sd. (Least squares minimises sum of squared residuals, so
    weighting the RESIDUAL by 1/sd yields the 1/sd^2 variance weighting on the square.)
    Arms with no usable SDs are fit unweighted.
    """
    if sd is None or not np.isfinite(sd).all() or np.any(sd <= 0):
        return np.ones(n_pts), False
    return 1.0 / sd, True


def fit_viability(cly, viability, sd=None, free_bottom=False, bottom_max=BOTTOM_MAX_NRK):
    """
    Fit the fixed-top Hill on the C_ly axis. Returns a dict of fitted quantities.

    `free_bottom=True` frees the floor into [0, bottom_max] (the NRK convention);
    otherwise the floor is pinned at 0 (the RPTEC convention).
    """
    cly = np.asarray(cly, dtype=float)
    y = np.asarray(viability, dtype=float)
    w, weighted = _weights(sd, y.size)

    # Start C50 at the geometric centre of the data, n at 2 (mild cooperativity),
    # floor at a small positive value so the optimiser can move it either way.
    x0 = [np.log10(np.sqrt(cly.min() * cly.max())), 2.0]
    lo = [np.log10(cly.min()) - 3.0, N_MIN]
    hi = [np.log10(cly.max()) + 3.0, N_MAX]
    if free_bottom:
        x0.append(min(5.0, bottom_max))
        lo.append(0.0)
        hi.append(bottom_max)

    res = least_squares(_residuals, x0=x0, bounds=(lo, hi), method="trf",
                        args=(cly, y, w, free_bottom))

    C50 = 10.0 ** res.x[0]
    n = float(res.x[1])
    bottom = float(res.x[2]) if free_bottom else 0.0

    return dict(
        C_ly50_hill=float(C50),
        n=n,
        bottom=bottom,
        C_ly50_abs=abs_midpoint(C50, n, bottom),
        sse=float(np.sum(res.fun ** 2)),
        weighted=weighted,
        free_bottom=free_bottom,
        n_at_bound=bool(n >= N_MAX * 0.999),
        n_saturation_flag=bool(n >= N_SATURATION_FLAG),
        bottom_at_bound=bool(free_bottom and bottom >= bottom_max * 0.999),
        # A free floor that collapses to 0 means the data provided no evidence for a
        # non-zero floor -- worth surfacing, since freeing it was a deliberate choice.
        bottom_collapsed_to_zero=bool(free_bottom and bottom <= 1e-6),
        n_points=int(y.size),
        result=res,
    )


def ci_covariance(fit, cly, viability, sd=None, level=0.95):
    """
    Asymptotic CI on C_ly50_abs from the fit covariance (Jacobian at the optimum).

    WITH 4-6 POINTS AND A NEAR-VERTICAL HILL THIS IS OPTIMISTIC. It is reported
    alongside the bootstrap precisely so the two can be compared: a large disagreement
    is itself the diagnostic that the fit is poorly constrained.
    """
    from scipy.stats import norm

    res = fit["result"]
    y = np.asarray(viability, dtype=float)
    n_par = res.x.size
    dof = y.size - n_par
    if dof <= 0:
        return (float("nan"), float("nan"))

    # Covariance from the Gauss-Newton approximation, scaled by residual variance for
    # unweighted fits (for weighted fits the residuals are already in sigma units).
    try:
        _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
        tol = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > tol]
        VT = VT[:s.size]
        cov = VT.T @ np.diag(1.0 / s ** 2) @ VT
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"))
    if not fit["weighted"]:
        cov = cov * (fit["sse"] / dof)

    # Delta method on log10(C50) -> C50, then carry to C_ly50_abs by the same factor
    # (abs_midpoint is proportional to C50 at fixed n, bottom).
    sd_log10 = np.sqrt(max(cov[0, 0], 0.0))
    z = norm.ppf(0.5 + level / 2.0)
    scale = fit["C_ly50_abs"] / fit["C_ly50_hill"] if fit["C_ly50_hill"] > 0 else 1.0
    lo = 10.0 ** (np.log10(fit["C_ly50_hill"]) - z * sd_log10) * scale
    hi = 10.0 ** (np.log10(fit["C_ly50_hill"]) + z * sd_log10) * scale
    return (float(lo), float(hi))


def ci_bootstrap(cly, viability, sd=None, free_bottom=False, bottom_max=BOTTOM_MAX_NRK,
                 B=BOOTSTRAP_B, level=0.95, seed=RNG_SEED):
    """
    Nonparametric CI on C_ly50_abs by RESIDUAL bootstrap.

    Resampling the points themselves would be unstable here: with only 4-6 doses per arm,
    many resamples would drop the single point that anchors the transition, and the fit
    would be meaningless rather than merely uncertain. Instead we resample the fit
    residuals and add them back to the fitted curve, which preserves the dose design
    while propagating the observed scatter. Percentile interval.
    """
    cly = np.asarray(cly, dtype=float)
    y = np.asarray(viability, dtype=float)
    base = fit_viability(cly, y, sd=sd, free_bottom=free_bottom, bottom_max=bottom_max)
    pred = hill(cly, base["C_ly50_hill"], base["n"], base["bottom"])
    resid = y - pred

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        y_star = pred + rng.choice(resid, size=resid.size, replace=True)
        y_star = np.clip(y_star, 0.0, 100.0)     # viability stays a bounded readout
        try:
            f = fit_viability(cly, y_star, sd=sd, free_bottom=free_bottom,
                              bottom_max=bottom_max)
        except Exception:
            continue
        v = f["C_ly50_abs"]
        if np.isfinite(v):
            draws.append(v)

    if len(draws) < B // 10:
        return (float("nan"), float("nan")), np.array(draws)
    draws = np.array(draws)
    a = (1.0 - level) / 2.0
    return (float(np.quantile(draws, a)), float(np.quantile(draws, 1.0 - a))), draws


# ---------------------------------------------------------------------------
# KE1 / LAMP corroboration fit -- free top, no upper plateau in range
# ---------------------------------------------------------------------------
LAMP_ONSET_LEVEL = 200.0     # 2x control = "lysosomal disturbance onset"

# Upper bound on the free LAMP plateau. The highest LAMP value observed is ~1041% of
# control; a fitted plateau far above that is pure extrapolation from 3-4 points with no
# upper asymptote in range. Left unbounded the optimiser runs the plateau to infinity and
# the model degenerates into a power law -- the fit still "converges" but its plateau is
# meaningless. Bounding it keeps the fit interpretable; A_at_bound flags when it binds.
LAMP_A_MAX = 5000.0


def lamp_model(cly, A, C50L, h):
    """
    Rising response with a hard 100% intercept (readout is % of control, so the foot is
    fixed by construction): 100 + A * C^h / (C50L^h + C^h). Free top (100 + A).
    """
    cly = np.clip(np.asarray(cly, dtype=float), 1e-12, None)
    return 100.0 + A * cly ** h / (C50L ** h + cly ** h)


def fit_lamp(cly, lamp, sd=None):
    """
    Fit the LAMP arm and extract the C_ly at which it crosses 2x control.

    The functional form is deliberately not over-invested in: this arm exists only for
    the AOP ordering check in Step 4b, not as a quantitative injury function.
    """
    cly = np.asarray(cly, dtype=float)
    y = np.asarray(lamp, dtype=float)
    w, weighted = _weights(sd, y.size)

    def resid(theta):
        A, C50L, h = 10.0 ** theta[0], 10.0 ** theta[1], theta[2]
        return (lamp_model(cly, A, C50L, h) - y) * w

    x0 = [np.log10(max(y.max() - 100.0, 10.0)),
          np.log10(np.sqrt(cly.min() * cly.max())), 2.0]
    lo = [np.log10(1.0), np.log10(cly.min()) - 3.0, 0.1]
    hi = [np.log10(LAMP_A_MAX), np.log10(cly.max()) + 3.0, 20.0]

    res = least_squares(resid, x0=x0, bounds=(lo, hi), method="trf")
    A, C50L, h = 10.0 ** res.x[0], 10.0 ** res.x[1], float(res.x[2])

    # Crossing of LAMP = 200: solve 100 + A*x/(1+x) = 200 with x = (C/C50L)^h
    # -> x = 100 / (A - 100), valid only if the fitted plateau exceeds 2x control.
    if A > LAMP_ONSET_LEVEL - 100.0:
        onset = float(C50L * (100.0 / (A - 100.0)) ** (1.0 / h))
    else:
        onset = float("nan")

    return dict(A=float(A), C50L=float(C50L), h=h, onset_cly=onset,
                sse=float(np.sum(res.fun ** 2)), weighted=weighted,
                A_at_bound=bool(A >= LAMP_A_MAX * 0.999),
                n_points=int(y.size), result=res)


def lamp_onset_sensitivity(cly, lamp, sd=None, a_maxes=(1500.0, 5000.0, 20000.0)):
    """
    Re-fit the LAMP arm under several plateau bounds and return the onset each implies.

    The plateau is unidentified (LAMP has no upper asymptote in range), so the onset is
    the only quantity worth reporting -- and it is only worth reporting if it survives
    the arbitrariness of the bound. Spread across these refits is the honest uncertainty
    on the AOP-ordering check.
    """
    out = {}
    for a_max in a_maxes:
        saved = globals()["LAMP_A_MAX"]
        globals()["LAMP_A_MAX"] = a_max
        try:
            out[a_max] = fit_lamp(cly, lamp, sd=sd)["onset_cly"]
        finally:
            globals()["LAMP_A_MAX"] = saved
    return out
