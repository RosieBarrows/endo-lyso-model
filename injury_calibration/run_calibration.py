"""
Driver: calibrate the lysosomal-load -> injury function f(C_ly) and characterise it.

Run with:  .venv/bin/python -m injury_calibration.run_calibration

Produces injury_calibration/results/*.csv + SUMMARY.md, and figures under figures/injury/.
Imports the frozen v0.4 accumulation model; never modifies it.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import model_core_v04 as mc
from injury_calibration import MODEL_VERSION, cly_mapping as cm, data_io, hill

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "figures", "injury")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# K_m sweep. {50, 100, 200, 500} is the PLAUSIBLE subset (matches
# concentration_sweep_mm_v04.py and prior work). 1000 and 2000 are limiting-behaviour
# probes, NOT equally-weighted candidates: as K_m greatly exceeds the dose range uptake
# becomes ~linear and C_ly-indexing collapses back toward nominal-dose indexing, and a
# K_m of 2000 uM implies implausibly weak polymyxin-megalin binding. See SUMMARY.md.
K_M_PLAUSIBLE = [50.0, 100.0, 200.0, 500.0]
K_M_PROBE = [1000.0, 2000.0]
K_M_SWEEP = K_M_PLAUSIBLE + K_M_PROBE

# Dense grid backing the continuous crossing-K_m root-find (log-spaced).
K_M_INTERP_GRID = np.logspace(np.log10(20.0), np.log10(5000.0), 40)

CELL_LINES = ["RPTEC/TERT1", "NRK-52E"]
SHORT = {"RPTEC/TERT1": "RPTEC", "NRK-52E": "NRK"}
COLORS = {"RPTEC/TERT1": "#1f77b4", "NRK-52E": "#d62728"}

# RPTEC fixes its floor at 0 (its data reaches zero); NRK frees it in [0, 20]
# (its curve bottoms out at 23% and never reaches a floor in range).
FREE_BOTTOM = {"RPTEC/TERT1": False, "NRK-52E": True}


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Step 2 sanity checks
# ---------------------------------------------------------------------------
def run_sanity_checks(arms):
    banner("SANITY CHECKS")
    total = cm.check_calibration_guard()
    print(f"  Calibration guard: RPTEC total at {mc.CAL_CONC} uM / 24 h = {total:.1f} nM "
          f"(expected ~{cm.CAL_TOTAL_NM_RPTEC:.0f}) -- PASS")
    print("    (pins TOTAL intracellular; C_ly was never measured, so there is no "
          "observed C_ly to guard against)")
    for (ke, cl), arm in sorted(arms.items()):
        if ke != "ke3_viability":
            continue
        for K_m in K_M_SWEEP:
            cm.check_monotonic(cl, arm.conc_uM, K_m)
        print(f"  C_ly monotonic in C_ext for {SHORT[cl]} across all K_m -- PASS")


# ---------------------------------------------------------------------------
# Step 3 -- sweep K_m, fit viability on the C_ly axis
# ---------------------------------------------------------------------------
def sweep_viability(arms):
    banner("STEP 3: K_m SWEEP -- viability Hill fits on the C_ly axis")
    rows, fits = [], {}
    for K_m in K_M_SWEEP:
        for cl in CELL_LINES:
            arm = arms.get(("ke3_viability", cl))
            if arm is None:
                continue
            cly = cm.map_doses(cl, arm.conc_uM, K_m)
            fb = FREE_BOTTOM[cl]
            f = hill.fit_viability(cly, arm.mean, sd=arm.sd, free_bottom=fb)
            cov_lo, cov_hi = hill.ci_covariance(f, cly, arm.mean, sd=arm.sd)
            (bs_lo, bs_hi), _ = hill.ci_bootstrap(cly, arm.mean, sd=arm.sd, free_bottom=fb)
            decades = cm.saturation_decades(cly, arm.mean)

            fits[(K_m, cl)] = dict(fit=f, cly=cly, arm=arm)
            rows.append(dict(
                K_m=K_m, K_m_class="plausible" if K_m in K_M_PLAUSIBLE else "probe",
                cell_line=SHORT[cl],
                C_ly50_abs=f["C_ly50_abs"], C_ly50_hill=f["C_ly50_hill"],
                n=f["n"], bottom=f["bottom"],
                ci_cov_lo=cov_lo, ci_cov_hi=cov_hi,
                ci_boot_lo=bs_lo, ci_boot_hi=bs_hi,
                sat_decades=decades, sse=f["sse"],
                n_saturation_flag=f["n_saturation_flag"],
                n_at_bound=f["n_at_bound"], bottom_at_bound=f["bottom_at_bound"],
                free_bottom=fb,
                bottom_collapsed_to_zero=f.get("bottom_collapsed_to_zero", False),
            ))
            print(f"  K_m={K_m:6.0f}  {SHORT[cl]:5s}  C_ly50_abs={f['C_ly50_abs']:9.0f} nM"
                  f"  n={f['n']:6.2f}  floor={f['bottom']:5.1f}"
                  f"  boot CI=[{bs_lo:8.0f},{bs_hi:9.0f}]"
                  f"  cov CI=[{cov_lo:8.0f},{cov_hi:9.0f}]"
                  f"  sat={decades:.2f} dec")
    return pd.DataFrame(rows), fits


def add_ratio(df):
    """Human/rat C_ly50_abs ratio at every swept K_m (result 4a), on the ABS midpoint."""
    piv = df.pivot_table(index="K_m", columns="cell_line", values="C_ly50_abs")
    if not {"RPTEC", "NRK"} <= set(piv.columns):
        df["ratio_NRK_over_RPTEC"] = np.nan
        return df
    ratio = (piv["NRK"] / piv["RPTEC"]).rename("ratio_NRK_over_RPTEC")
    return df.merge(ratio, left_on="K_m", right_index=True, how="left")


# ---------------------------------------------------------------------------
# Crossing K_m: solve ratio(K_m) = 1 -- the primary result
# ---------------------------------------------------------------------------
def _ratio_at(K_m, interps, arms):
    """C_ly50_abs(NRK) / C_ly50_abs(RPTEC) at an arbitrary K_m, via interpolated C_ly."""
    vals = {}
    for cl in CELL_LINES:
        arm = arms[("ke3_viability", cl)]
        cly = interps[cl](K_m)
        f = hill.fit_viability(cly, arm.mean, sd=arm.sd, free_bottom=FREE_BOTTOM[cl])
        vals[cl] = f["C_ly50_abs"]
    return vals["NRK-52E"] / vals["RPTEC/TERT1"]


def find_crossing(arms, n_boot=400, seed=hill.RNG_SEED):
    """
    Solve for the K_m at which C_ly50_abs(NRK) == C_ly50_abs(RPTEC), i.e. the lysosomal
    K_m the human/rat collapse test implies. Bootstrap the crossing for a CI.

    C_ly is precomputed on a dense K_m grid and interpolated, so the root-find and the
    bootstrap never re-solve the ODEs.
    """
    banner("PRIMARY RESULT: crossing K_m where the human/rat C_ly50 gap closes")
    interps = {cl: cm.build_cly_interpolator(cl, arms[("ke3_viability", cl)].conc_uM,
                                             K_M_INTERP_GRID)
               for cl in CELL_LINES}

    lo, hi = K_M_INTERP_GRID[0], K_M_INTERP_GRID[-1]
    f_lo, f_hi = _ratio_at(lo, interps, arms) - 1.0, _ratio_at(hi, interps, arms) - 1.0
    print(f"  ratio-1 at K_m={lo:.0f}: {f_lo:+.3f}   at K_m={hi:.0f}: {f_hi:+.3f}")
    if f_lo * f_hi > 0:
        print("  No sign change across the grid -- ratio never reaches 1. "
              "Reporting NaN crossing.")
        return dict(K_m_cross=float("nan"), ci=(float("nan"), float("nan")),
                    draws=np.array([]), interps=interps, bracketed=False)

    K_cross = brentq(lambda km: _ratio_at(km, interps, arms) - 1.0, lo, hi, xtol=1e-3)
    print(f"  Crossing K_m = {K_cross:.0f} uM")

    # Bootstrap: resample residuals per arm, refit, re-solve the crossing.
    rng = np.random.default_rng(seed)
    base = {}
    for cl in CELL_LINES:
        arm = arms[("ke3_viability", cl)]
        cly0 = interps[cl](K_cross)
        f0 = hill.fit_viability(cly0, arm.mean, sd=arm.sd, free_bottom=FREE_BOTTOM[cl])
        pred = hill.hill(cly0, f0["C_ly50_hill"], f0["n"], f0["bottom"])
        base[cl] = (arm, pred, arm.mean - pred)

    draws = []
    for _ in range(n_boot):
        y_star = {}
        for cl in CELL_LINES:
            arm, pred, resid = base[cl]
            y_star[cl] = np.clip(pred + rng.choice(resid, size=resid.size, replace=True),
                                 0.0, 100.0)

        def ratio_star(km):
            vals = {}
            for cl in CELL_LINES:
                arm = arms[("ke3_viability", cl)]
                f = hill.fit_viability(interps[cl](km), y_star[cl], sd=arm.sd,
                                       free_bottom=FREE_BOTTOM[cl])
                vals[cl] = f["C_ly50_abs"]
            return vals["NRK-52E"] / vals["RPTEC/TERT1"] - 1.0

        try:
            if ratio_star(lo) * ratio_star(hi) > 0:
                continue
            draws.append(brentq(ratio_star, lo, hi, xtol=1e-2))
        except Exception:
            continue

    draws = np.array(draws)
    if draws.size >= n_boot // 10:
        ci = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
        cv = float(np.std(draws) / np.mean(draws))
        print(f"  Bootstrap 95% CI = [{ci[0]:.0f}, {ci[1]:.0f}] uM   "
              f"CV = {cv:.2f}   ({draws.size}/{n_boot} replicates bracketed)")
    else:
        ci, cv = (float("nan"), float("nan")), float("nan")
        print(f"  Bootstrap failed to bracket ({draws.size}/{n_boot}) -- CI unavailable")

    return dict(K_m_cross=K_cross, ci=ci, cv=cv, draws=draws, interps=interps,
                bracketed=True)


# ---------------------------------------------------------------------------
# Step 3 corroboration + Step 4b -- LAMP
# ---------------------------------------------------------------------------
def fit_lamp_arms(arms):
    banner("STEP 4b: LAMP (KE1) corroboration -- lysosomal-disturbance onset")
    rows, fits = [], {}
    for K_m in K_M_SWEEP:
        for cl in CELL_LINES:
            arm = arms.get(("ke1_lamp", cl))
            if arm is None:
                continue
            cly = cm.map_doses(cl, arm.conc_uM, K_m)
            f = hill.fit_lamp(cly, arm.mean, sd=arm.sd)
            # The LAMP plateau is unidentified (no upper asymptote in range), so the
            # onset is only trustworthy if it survives the arbitrariness of that bound.
            sens = hill.lamp_onset_sensitivity(cly, arm.mean, sd=arm.sd)
            svals = np.array([v for v in sens.values() if np.isfinite(v)])
            spread = float(svals.max() / svals.min()) if svals.size else float("nan")

            fits[(K_m, cl)] = dict(fit=f, cly=cly, arm=arm, onset_sens=sens)
            rows.append(dict(K_m=K_m, cell_line=SHORT[cl],
                             lamp_onset_cly=f["onset_cly"],
                             onset_lo=float(svals.min()) if svals.size else np.nan,
                             onset_hi=float(svals.max()) if svals.size else np.nan,
                             onset_spread=spread, A=f["A"], h=f["h"],
                             A_at_bound=f["A_at_bound"], weighted=f["weighted"]))
            print(f"  K_m={K_m:6.0f}  {SHORT[cl]:5s}  LAMP 2x-control onset "
                  f"C_ly={f['onset_cly']:8.0f} nM  "
                  f"(bound-sensitivity {svals.min():.0f}-{svals.max():.0f}, "
                  f"{spread:.2f}x; plateau {100 + f['A']:.0f}%"
                  f"{' AT BOUND' if f['A_at_bound'] else ''}, "
                  f"{'weighted' if f['weighted'] else 'unweighted'})")
    return pd.DataFrame(rows), fits


def check_aop_ordering(df_viab, df_lamp):
    """AOP ordering: LAMP onset (KE1) must sit BELOW viability C_ly50 (KE3)."""
    banner("STEP 4b: AOP ordering check -- KE1 onset below KE3 C_ly50?")
    if df_lamp.empty:
        print("  No LAMP arms available -- ordering check skipped.")
        return pd.DataFrame()
    m = df_viab.merge(df_lamp, on=["K_m", "cell_line"], how="inner")
    m["aop_ok"] = m["lamp_onset_cly"] < m["C_ly50_abs"]
    # Robust only if the ordering survives the WORST-CASE onset under the plateau-bound
    # sensitivity -- i.e. even the highest onset the bound choice permits stays below
    # C_ly50. Otherwise the verdict is an artefact of an arbitrary bound.
    m["aop_robust"] = m["onset_hi"] < m["C_ly50_abs"]
    for _, r in m.iterrows():
        verdict = "PASS" if r["aop_ok"] else "FAIL"
        rob = "robust" if r["aop_robust"] else "NOT robust to plateau bound"
        print(f"  K_m={r['K_m']:6.0f}  {r['cell_line']:5s}  "
              f"KE1 onset {r['lamp_onset_cly']:8.0f} (worst {r['onset_hi']:8.0f}) "
              f"vs KE3 C_ly50 {r['C_ly50_abs']:8.0f} nM  -> {verdict} ({rob})")
    return m[["K_m", "cell_line", "lamp_onset_cly", "onset_lo", "onset_hi",
              "C_ly50_abs", "aop_ok", "aop_robust"]]


# ---------------------------------------------------------------------------
# Control fit: same Hill on the NOMINAL axis (should recover Jarzina's EC50s)
# ---------------------------------------------------------------------------
def nominal_control_fit(arms):
    """
    Fit viability against nominal concentration, as a machinery check. Recovering
    Jarzina's reported ~57 uM (RPTEC) / ~575 uM (NRK) EC50s validates the fitting code
    before its output on the C_ly axis is trusted.
    """
    banner("CONTROL: same Hill fitted on the NOMINAL concentration axis")
    rows = []
    for cl in CELL_LINES:
        arm = arms.get(("ke3_viability", cl))
        if arm is None:
            continue
        f = hill.fit_viability(arm.conc_uM, arm.mean, sd=arm.sd,
                               free_bottom=FREE_BOTTOM[cl])
        rows.append(dict(cell_line=SHORT[cl], EC50_abs_uM=f["C_ly50_abs"],
                         EC50_hill_uM=f["C_ly50_hill"], n=f["n"], bottom=f["bottom"]))
        print(f"  {SHORT[cl]:5s}  EC50(abs) = {f['C_ly50_abs']:7.1f} uM   "
              f"n = {f['n']:5.2f}   floor = {f['bottom']:4.1f}")
    print("  Jarzina report ~57 uM (RPTEC) and ~575 uM (NRK) -- compare above.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_viability_vs_cly(fits):
    n = len(K_M_SWEEP)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    for ax, K_m in zip(axes.ravel(), K_M_SWEEP):
        for cl in CELL_LINES:
            e = fits.get((K_m, cl))
            if e is None:
                continue
            f, cly, arm = e["fit"], e["cly"], e["arm"]
            ax.errorbar(cly, arm.mean, yerr=arm.sd, fmt="o", color=COLORS[cl],
                        ms=6, capsize=3, label=f"{SHORT[cl]} data")
            grid = np.logspace(np.log10(cly.min() / 3), np.log10(cly.max() * 3), 300)
            ax.plot(grid, hill.hill(grid, f["C_ly50_hill"], f["n"], f["bottom"]),
                    "-", color=COLORS[cl], lw=2,
                    label=f"{SHORT[cl]} fit (C$_{{ly}}$50={f['C_ly50_abs']:.0f})")
            ax.axvline(f["C_ly50_abs"], color=COLORS[cl], ls=":", lw=1)
        ax.axhline(50, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("C$_{ly}$ at 24 h (nM)")
        ax.set_ylabel("viability (% of control)")
        cls = "plausible" if K_m in K_M_PLAUSIBLE else "limiting probe"
        ax.set_title(f"K$_m$ = {K_m:.0f} µM ({cls})")
        ax.set_ylim(-5, 115)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Viability vs lysosomal load, fitted Hill per K$_m$", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "viability_vs_cly.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_cly50_band(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cl in CELL_LINES:
        s = df[df["cell_line"] == SHORT[cl]].sort_values("K_m")
        if s.empty:
            continue
        lo = s["C_ly50_abs"] - s["ci_boot_lo"]
        hi = s["ci_boot_hi"] - s["C_ly50_abs"]
        ax.errorbar(s["K_m"], s["C_ly50_abs"], yerr=[lo.clip(lower=0), hi.clip(lower=0)],
                    fmt="o-", color=COLORS[cl], capsize=4, lw=2, ms=7,
                    label=f"{SHORT[cl]} (bootstrap 95% CI)")
    ax.axvspan(min(K_M_PROBE), max(K_M_PROBE) * 1.05, color="grey", alpha=0.12)
    ax.text(min(K_M_PROBE) * 1.05, ax.get_ylim()[1] * 0.95, "limiting probes\n(not equally weighted)",
            fontsize=8, va="top", color="dimgrey")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K$_m$ (µM)  —  unidentified from the single-dose calibration")
    ax.set_ylabel("C$_{ly}$50 at 50% absolute viability (nM)")
    ax.set_title("The headline uncertainty: C$_{ly}$50 is K$_m$-conditional")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "cly50_vs_km_band.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_ratio_and_crossing(df, cross):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    s = df.drop_duplicates("K_m").sort_values("K_m")
    ax.plot(s["K_m"], s["ratio_NRK_over_RPTEC"], "o-", color="#6a3d9a", lw=2, ms=7,
            label="C$_{ly}$50(NRK) / C$_{ly}$50(RPTEC)")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="ratio = 1 (gap fully collapsed)")
    if np.isfinite(cross["K_m_cross"]):
        ax.axvline(cross["K_m_cross"], color="crimson", ls="-", lw=1.5,
                   label=f"crossing K$_m$ = {cross['K_m_cross']:.0f} µM")
        lo, hi = cross["ci"]
        if np.isfinite(lo):
            ax.axvspan(lo, hi, color="crimson", alpha=0.12, label="crossing 95% CI")
    ax.axvspan(min(K_M_PROBE), max(K_M_PROBE) * 1.05, color="grey", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("K$_m$ (µM)")
    ax.set_ylabel("C$_{ly}$50 ratio, rat / human")
    ax.set_title("Does the human/rat sensitivity gap collapse on the C$_{ly}$ axis?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "cly50_ratio_crossing.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_saturation_diagnostic(fits, arms):
    n = len(K_M_SWEEP)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    for ax, K_m in zip(axes.ravel(), K_M_SWEEP):
        for cl in CELL_LINES:
            e = fits.get((K_m, cl))
            if e is None:
                continue
            arm, cly = e["arm"], e["cly"]
            ax.plot(arm.conc_uM, cly, "o-", color=COLORS[cl], lw=2, ms=6,
                    label=f"{SHORT[cl]}")
            trans = (arm.mean > 10) & (arm.mean < 90)
            if trans.any():
                ax.plot(arm.conc_uM[trans], cly[trans], "s", ms=11, mfc="none",
                        mec=COLORS[cl], mew=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("nominal C$_{ext}$ (µM)")
        ax.set_ylabel("C$_{ly}$ at 24 h (nM)")
        d = {SHORT[cl]: cm.saturation_decades(fits[(K_m, cl)]["cly"],
                                              fits[(K_m, cl)]["arm"].mean)
             for cl in CELL_LINES if (K_m, cl) in fits}
        dtxt = "  ".join(f"{k}:{v:.2f}dec" for k, v in d.items())
        ax.set_title(f"K$_m$ = {K_m:.0f} µM\ntransition spans {dtxt}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Saturation diagnostic: C$_{ly}$ vs nominal dose "
                 "(squares = viability transition points)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "saturation_diagnostic.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_lamp(lamp_fits, viab_df):
    ks = [k for k in K_M_SWEEP if any((k, cl) in lamp_fits for cl in CELL_LINES)]
    if not ks:
        return None
    ncol = 3
    nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    for ax, K_m in zip(axes.ravel(), ks):
        for cl in CELL_LINES:
            e = lamp_fits.get((K_m, cl))
            if e is None:
                continue
            f, cly, arm = e["fit"], e["cly"], e["arm"]
            yerr = arm.sd if np.isfinite(arm.sd).all() else None
            ax.errorbar(cly, arm.mean, yerr=yerr, fmt="o", color=COLORS[cl], ms=6,
                        capsize=3, label=f"{SHORT[cl]} LAMP")
            grid = np.logspace(np.log10(cly.min() / 3), np.log10(cly.max() * 3), 300)
            ax.plot(grid, hill.lamp_model(grid, f["A"], f["C50L"], f["h"]),
                    "-", color=COLORS[cl], lw=2)
            if np.isfinite(f["onset_cly"]):
                ax.axvline(f["onset_cly"], color=COLORS[cl], ls=":", lw=1.5)
            sub = viab_df[(viab_df["K_m"] == K_m) & (viab_df["cell_line"] == SHORT[cl])]
            if not sub.empty:
                ax.axvline(float(sub["C_ly50_abs"].iloc[0]), color=COLORS[cl],
                           ls="--", lw=1.5, alpha=0.6)
        ax.axhline(hill.LAMP_ONSET_LEVEL, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("C$_{ly}$ at 24 h (nM)")
        ax.set_ylabel("LAMP-1 (% of control)")
        ax.set_title(f"K$_m$ = {K_m:.0f} µM\ndotted = KE1 onset, dashed = KE3 C$_{{ly}}$50",
                     fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(ks):]:
        ax.axis("off")
    fig.suptitle("KE1 (LAMP-1) corroboration and AOP ordering on the C$_{ly}$ axis",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "lamp_vs_cly.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    banner(f"INJURY CALIBRATION -- f(C_ly) from Jarzina (2022), accumulation model {MODEL_VERSION}")
    arms = data_io.load_all()
    if ("ke3_viability", "RPTEC/TERT1") not in arms:
        raise SystemExit("No RPTEC viability arm -- nothing to calibrate.")

    run_sanity_checks(arms)
    nominal_df = nominal_control_fit(arms)

    viab_df, viab_fits = sweep_viability(arms)
    viab_df = add_ratio(viab_df)

    cross = find_crossing(arms)
    lamp_df, lamp_fits = fit_lamp_arms(arms)
    aop_df = check_aop_ordering(viab_df, lamp_df)

    payasi = data_io.load_payasi()
    if payasi is None:
        print("\n  [Payasi cross-check] data/payasi_viability_rptec.csv not present "
              "-- Step 5 skipped (expected).")

    banner("WRITING OUTPUTS")
    viab_df.to_csv(os.path.join(RESULTS_DIR, "cly50_by_km.csv"), index=False)
    nominal_df.to_csv(os.path.join(RESULTS_DIR, "nominal_axis_control.csv"), index=False)
    if not lamp_df.empty:
        lamp_df.to_csv(os.path.join(RESULTS_DIR, "lamp_onset_by_km.csv"), index=False)
    if not aop_df.empty:
        aop_df.to_csv(os.path.join(RESULTS_DIR, "aop_ordering.csv"), index=False)
    for f in ["cly50_by_km.csv", "nominal_axis_control.csv"]:
        print(f"  results/{f}")

    figs = [fig_viability_vs_cly(viab_fits), fig_cly50_band(viab_df),
            fig_ratio_and_crossing(viab_df, cross), fig_saturation_diagnostic(viab_fits, arms),
            fig_lamp(lamp_fits, viab_df)]
    for p in figs:
        if p:
            print(f"  {os.path.relpath(p, os.path.dirname(RESULTS_DIR))}")

    write_summary(viab_df, nominal_df, lamp_df, aop_df, cross, arms)
    print(f"  results/SUMMARY.md")
    return viab_df, cross


def write_summary(viab_df, nominal_df, lamp_df, aop_df, cross, arms):
    """Write results/SUMMARY.md -- see the module docstring for what it must lead with."""
    from injury_calibration.summary import render
    text = render(viab_df, nominal_df, lamp_df, aop_df, cross, arms,
                  K_M_PLAUSIBLE, K_M_PROBE, MODEL_VERSION)
    with open(os.path.join(RESULTS_DIR, "SUMMARY.md"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
