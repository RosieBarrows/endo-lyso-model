"""
v0.4 -- Michaelis-Menten uptake + K_m sweep, with Shipman (2022) trafficking rates.

Identical to the v0.3 sweep (concentration_sweep_mm.py) except the fixed
trafficking rate constants are the v0.4 Shipman OK-cell values:
    k_mat = 0.048, k_fuse = 0.0094 /min  (k_rec at its nominal 0.02, k_esc 0.0002).
The slower k_fuse (0.02 -> 0.0094) lengthens lysosomal filling, so threshold
crossing times are later than v0.3. k_deg and the linear k_uptake calibration are
re-derived internally from the 34 uM data (Step 1 / Step 2A) under these rates.

Replaces the linear uptake term (k_uptake * C_ext) with the full saturating form:

    uptake = V_max * C_ext / (K_m + C_ext),   V_max = k_uptake_fitted * (K_m + CAL_CONC)

The (K_m + CAL_CONC) back-calculation anchors the saturating curve to the fitted
linear uptake AT the 34 uM calibration dose, so the fit is preserved at 34 uM for
every K_m (not only in the C_ext << K_m limit); above the calibration dose K_m
bends the dose-response and uptake saturates at V_max. (An earlier V_max =
k_uptake * K_m pinned the C_ext -> 0 tangent and undershot the 34 uM data by
K_m/(K_m + 34) -- ~15% at K_m = 200. See model_core_v04.v_max.)

Sweep K_m in {50, 100, 200, 500} uM over doses {34, 125, 250, 500, 1000, 2000} uM,
24h exposure + 24h washout. Figures -> ./mm_sweep_v04/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

OUTDIR = os.path.join("figures", "mm_sweep_v04")   # all generated figures live under figures/ (gitignored)
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants / fixed parameters (v0.4 -- Shipman trafficking rates)
# ---------------------------------------------------------------------------
T_WASHOUT_MIN = 24.0 * 60.0
SIM_END_MIN   = 50.0 * 60.0
V_CELL_RPTEC  = 1.99e-12
V_CELL_NRK    = 1.21e-12
FIXED = dict(k_mat=0.048, k_rec=0.02, k_fuse=0.0094, k_esc=0.0002)

CAL_CONC   = 34.0
DOSES      = [34.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
KM_VALUES  = [50.0, 100.0, 200.0, 500.0]


# ---------------------------------------------------------------------------
# Model with Michaelis-Menten uptake
# ---------------------------------------------------------------------------
def odes(t, y, p, c_ext):
    C_ee, C_le, C_ly = y
    uptake = p['V_max'] * c_ext / (p['K_m'] + c_ext)     # MM uptake (fmol/cell/min)
    dC_ee = uptake             - (p['k_mat'] + p['k_rec']) * C_ee
    dC_le = p['k_mat'] * C_ee  - (p['k_fuse'] + p['k_esc']) * C_le
    dC_ly = p['k_fuse'] * C_le - p['k_deg'] * C_ly
    return [dC_ee, dC_le, dC_ly]


def simulate(p, V_cell, c_ext_um, t_eval_h, return_compartments=False):
    t_eval_h = np.atleast_1d(np.asarray(t_eval_h, dtype=float))
    t_eval_min = t_eval_h * 60.0
    mask1 = t_eval_min <= T_WASHOUT_MIN
    mask2 = ~mask1
    sol1 = solve_ivp(odes, (0.0, T_WASHOUT_MIN), [0.0, 0.0, 0.0],
                     args=(p, c_ext_um), method='BDF', dense_output=True,
                     rtol=1e-8, atol=1e-12)
    sol2 = solve_ivp(odes, (T_WASHOUT_MIN, SIM_END_MIN), sol1.y[:, -1],
                     args=(p, 0.0), method='BDF', dense_output=True,
                     rtol=1e-8, atol=1e-12)
    Y = np.zeros((3, t_eval_min.size))
    if mask1.any():
        Y[:, mask1] = sol1.sol(t_eval_min[mask1])
    if mask2.any():
        Y[:, mask2] = sol2.sol(t_eval_min[mask2])
    Y = np.clip(Y, 0.0, None)
    comp_nM = Y * (1e-6 / V_cell)
    total = comp_nM.sum(axis=0)
    if return_compartments:
        return t_eval_h, total, dict(C_ee=comp_nM[0], C_le=comp_nM[1], C_ly=comp_nM[2])
    return t_eval_h, total


def make_params(V_max, K_m, k_deg):
    p = dict(FIXED); p['V_max'] = V_max; p['K_m'] = K_m; p['k_deg'] = k_deg
    return p


# ---------------------------------------------------------------------------
# Re-derive linear k_uptake / k_deg fits (Step 1 + Step 2A) at 34 uM under the
# v0.4 fixed rates (fit in the linear regime: uptake = k_uptake*C_ext == MM with huge K_m)
# ---------------------------------------------------------------------------
T_FIT_H = np.array([3.53, 6.44, 24.0, 48.0])
Y_RPTEC = np.array([636.0, 1377.0, 3558.0, 2792.0])
Y_NRK   = np.array([299.0,  662.0,  831.0,  104.0])
LIN_KM  = 1e9        # effectively-linear K_m for reproducing the linear fit
KU_INIT, KU_LO, KU_HI = 1e-6, 1e-9, 1e-4
KD_INIT, KD_LO, KD_HI = 1e-3, 1e-5, 1e-1


def _resid(theta, fit_keys, base_ku, base_kd, V_cell, y_data):
    vals = {'k_uptake': base_ku, 'k_deg': base_kd}
    for key, lg in zip(fit_keys, theta):
        vals[key] = 10.0 ** lg
    # linear regime: V_max = k_uptake * LIN_KM so uptake ~= k_uptake * C_ext
    p = make_params(vals['k_uptake'] * LIN_KM, LIN_KM, vals['k_deg'])
    _, model = simulate(p, V_cell, CAL_CONC, T_FIT_H)
    return np.log10(np.clip(model, 1e-9, None)) - np.log10(y_data)


def _fit(fit_keys, base_ku, base_kd, V_cell, y_data):
    init, lo, hi = [], [], []
    for key in fit_keys:
        if key == 'k_uptake':
            init.append(np.log10(KU_INIT)); lo.append(np.log10(KU_LO)); hi.append(np.log10(KU_HI))
        else:
            init.append(np.log10(KD_INIT)); lo.append(np.log10(KD_LO)); hi.append(np.log10(KD_HI))
    res = least_squares(_resid, x0=init, bounds=(lo, hi), method='trf',
                        args=(fit_keys, base_ku, base_kd, V_cell, y_data))
    out = {'k_uptake': base_ku, 'k_deg': base_kd}
    for key, lg in zip(fit_keys, res.x):
        out[key] = 10.0 ** lg
    return out

f1 = _fit(['k_uptake', 'k_deg'], KU_INIT, KD_INIT, V_CELL_RPTEC, Y_RPTEC)
k_uptake_RPTEC, k_deg_PB = f1['k_uptake'], f1['k_deg']
f2 = _fit(['k_uptake'], KU_INIT, k_deg_PB, V_CELL_NRK, Y_NRK)
k_uptake_NRK = f2['k_uptake']
print("v0.4 calibrated (linear) parameters under Shipman trafficking rates:")
print(f"  k_uptake_RPTEC={k_uptake_RPTEC:.4g}, k_uptake_NRK={k_uptake_NRK:.4g}, k_deg_PB={k_deg_PB:.4g}")

# v0.4 reference (linear) peak total / peak C_ly at 34 uM, for fit-preservation/comparison
t_grid = np.unique(np.concatenate([np.linspace(0, 6, 1200), np.linspace(6, 50, 800)]))
p_lin_R = make_params(k_uptake_RPTEC * LIN_KM, LIN_KM, k_deg_PB)
p_lin_N = make_params(k_uptake_NRK * LIN_KM, LIN_KM, k_deg_PB)
_, lin_tot_R34 = simulate(p_lin_R, V_CELL_RPTEC, CAL_CONC, t_grid)
v04_peak_total_R34 = lin_tot_R34.max()


def first_crossing_h(t, y, thr):
    above = np.where(y >= thr)[0]
    return t[above[0]] if above.size else None


# ---------------------------------------------------------------------------
# Output 1: Fit-preservation check at 34 uM (4 K_m curves vs data)
# ---------------------------------------------------------------------------
RPTEC_C = np.array([0.0, 636.0, 1377.0, 3558.0, 2792.0])
RPTEC_LO = np.array([0.0, 442.0, 870.0, 1974.0, 1494.0])
RPTEC_HI = np.array([0.0, 883.0, 1909.0, 5156.0, 4065.0])
NRK_C = np.array([0.0, 299.0, 662.0, 831.0, 104.0])
T_ALL = np.array([0.0, 3.53, 6.44, 24.0, 48.0])
km_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(KM_VALUES)))

fig1, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
for c_lbl, ax, ydata, yerr_lo, yerr_hi, kuptake, V in [
        ("RPTEC/TERT1", axL, RPTEC_C, RPTEC_LO, RPTEC_HI, k_uptake_RPTEC, V_CELL_RPTEC),
        ("NRK-52E",     axR, NRK_C,   None,     None,     k_uptake_NRK,   V_CELL_NRK)]:
    for K_m, col in zip(KM_VALUES, km_colors):
        p = make_params(kuptake * (K_m + CAL_CONC), K_m, k_deg_PB)
        _, tot = simulate(p, V, CAL_CONC, t_grid)
        ax.plot(t_grid, tot, color=col, lw=2, label=f"K_m={K_m:.0f} uM")
    if yerr_lo is not None:
        ax.errorbar(T_ALL, ydata, yerr=np.vstack([ydata - yerr_lo, yerr_hi - ydata]),
                    fmt='ko', ms=6, capsize=3, label="v0.2 data")
    else:
        ax.plot(T_ALL, ydata, 'ko', ms=6, label="v0.2 data")
    ax.axvline(24, ls='--', color='grey', alpha=0.6)
    ax.set_title(f"{c_lbl} at 34 uM"); ax.set_xlabel("time (h)")
    ax.set_ylabel("intracellular conc (nM)"); ax.set_xlim(0, 50)
    ax.legend(fontsize=8, loc='upper left')
fig1.suptitle("Output 1: Fit preservation at 34 uM (v0.4 Shipman rates)", fontsize=12)
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(os.path.join(OUTDIR, "v04_output1_fit_preservation.png"), dpi=150)

# ---------------------------------------------------------------------------
# Precompute RPTEC results for all (K_m, dose): store C_ly trajectories + peaks
# ---------------------------------------------------------------------------
results = {}   # (K_m, dose) -> dict
global_cly_max = 0.0
dose_colors = plt.cm.viridis(np.linspace(0, 0.9, len(DOSES)))
for K_m in KM_VALUES:
    V_max_R = k_uptake_RPTEC * (K_m + CAL_CONC)
    for dose in DOSES:
        p = make_params(V_max_R, K_m, k_deg_PB)
        _, tot, comps = simulate(p, V_CELL_RPTEC, dose, t_grid, return_compartments=True)
        cly = comps['C_ly']
        results[(K_m, dose)] = dict(
            V_max=V_max_R, tot=tot, cly=cly,
            peak_total=tot.max(), peak_cly=cly.max(),
            t500=first_crossing_h(t_grid, cly, 500.0),
            t1000=first_crossing_h(t_grid, cly, 1000.0))
        global_cly_max = max(global_cly_max, cly.max())

# ---------------------------------------------------------------------------
# Output 2: lysosomal load vs time, RPTEC, 4 subplots (one per K_m), shared y
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
for ax, K_m in zip(axes2.ravel(), KM_VALUES):
    for dose, col in zip(DOSES, dose_colors):
        ax.plot(t_grid, results[(K_m, dose)]['cly'], color=col, lw=1.8,
                label=f"{dose:.0f} uM")
    for thr in (500.0, 1000.0):
        ax.axhline(thr, ls='--', color='grey', alpha=0.6)
    ax.axvline(24, ls='--', color='grey', alpha=0.5)
    ax.set_ylim(0, global_cly_max * 1.05)
    ax.set_title(f"K_m = {K_m:.0f} uM   (V_max_RPTEC = {k_uptake_RPTEC * (K_m + CAL_CONC):.3g} fmol/cell/min)")
    ax.set_xlabel("time (h)"); ax.set_ylabel("lysosomal C_ly (nM)")
    ax.set_xlim(0, 50); ax.legend(fontsize=7, loc='upper right')
fig2.suptitle("Output 2: RPTEC/TERT1 lysosomal load vs time (v0.4, per K_m, all doses)",
              fontsize=13)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(os.path.join(OUTDIR, "v04_output2_lysosomal_load.png"), dpi=150)

# ---------------------------------------------------------------------------
# Output 3: threshold crossing time vs dose (500 & 1000 nM, 4 K_m)
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 6))
for K_m, col in zip(KM_VALUES, km_colors):
    for thr, ls, mk in [(500.0, '-', 'o'), (1000.0, '--', 's')]:
        xs, ys = [], []
        for dose in DOSES:
            tc = results[(K_m, dose)][f"t{int(thr)}"]
            if tc is not None:
                xs.append(dose); ys.append(tc)
        ax3.plot(xs, ys, ls, marker=mk, color=col,
                 label=f"K_m={K_m:.0f}, {int(thr)}nM")
ax3.set_xscale('log')
ax3.set_xlabel("extracellular concentration (uM)")
ax3.set_ylabel("time to cross threshold (h)")
ax3.set_title("Output 3: Lysosomal threshold-crossing time vs dose (v0.4)\n"
              "(solid = 500 nM, dashed = 1000 nM; colour = K_m)")
ax3.legend(fontsize=7, ncol=2)
ax3.grid(True, which='both', alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTDIR, "v04_output3_threshold_crossing.png"), dpi=150)

# ---------------------------------------------------------------------------
# Output 4: peak C_ly vs dose, 4 K_m curves + linear reference diagonal
# ---------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(9, 6))
for K_m, col in zip(KM_VALUES, km_colors):
    peaks = [results[(K_m, dose)]['peak_cly'] for dose in DOSES]
    ax4.plot(DOSES, peaks, 'o-', color=col, label=f"K_m={K_m:.0f} uM")
# linear-model reference (peak C_ly scales linearly with dose)
_, _, comps_lin = simulate(p_lin_R, V_CELL_RPTEC, CAL_CONC, t_grid, return_compartments=True)
lin_peak34 = comps_lin['C_ly'].max()
lin_ref = [lin_peak34 * d / CAL_CONC for d in DOSES]
ax4.plot(DOSES, lin_ref, 'k--', lw=1.5, label="linear model (v0.4)")
ax4.set_xscale('log'); ax4.set_yscale('log')
ax4.set_xlabel("extracellular concentration (uM)")
ax4.set_ylabel("peak lysosomal C_ly (nM)")
ax4.set_title("Output 4: Peak lysosomal load vs dose (RPTEC/TERT1, v0.4)\n"
              "saturation flattens curves below the linear reference")
ax4.legend(fontsize=9); ax4.grid(True, which='both', alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTDIR, "v04_output4_peak_cly_vs_dose.png"), dpi=150)
plt.close('all')

# ---------------------------------------------------------------------------
# Output 5: summary table (24 rows) + flagged checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("OUTPUT 5: RPTEC/TERT1 summary  (24 rows, v0.4)")
print("=" * 92)
print(f"  {'K_m':>5} {'C_ext':>6} {'V_max':>10} {'peak_total(nM)':>15} "
      f"{'peak_Cly(nM)':>13} {'t>500(h)':>9} {'t>1000(h)':>10} {'plausible?':>11}")
for K_m in KM_VALUES:
    for dose in DOSES:
        r = results[(K_m, dose)]
        plausible = r['peak_total'] < dose * 1000.0   # nM vs uM->nM
        t5 = f"{r['t500']:.2f}" if r['t500'] is not None else "never"
        t1 = f"{r['t1000']:.2f}" if r['t1000'] is not None else "never"
        print(f"  {K_m:>5.0f} {dose:>6.0f} {r['V_max']:>10.3g} {r['peak_total']:>15.1f} "
              f"{r['peak_cly']:>13.1f} {t5:>9} {t1:>10} {('YES' if plausible else 'NO'):>11}")

# ---- Flag 1: K_m=50 calibration degradation at 34 uM ----
print("\n" + "=" * 92)
print("FLAGGED CHECKS")
print("=" * 92)
print("1. 34 uM calibration vs v0.4 linear (peak total intracellular):")
print(f"   v0.4 linear reference = {v04_peak_total_R34:.1f} nM")
for K_m in KM_VALUES:
    pk = results[(K_m, 34.0)]['peak_total']
    print(f"   K_m={K_m:>4.0f}: {pk:8.1f} nM  ({(pk / v04_peak_total_R34 - 1) * 100:+.1f}% vs linear)")

# ---- Flag 2: smallest K_m giving plausible 2000 uM result ----
plausible_kms = [K_m for K_m in KM_VALUES
                 if results[(K_m, 2000.0)]['peak_total'] < 2000.0 * 1000.0]
print("\n2. K_m values where 2000 uM is physically plausible (peak intracellular < extracellular):")
print(f"   {plausible_kms if plausible_kms else 'NONE in tested set'}")
for K_m in KM_VALUES:
    pk = results[(K_m, 2000.0)]['peak_total']
    print(f"   K_m={K_m:>4.0f}: peak={pk:10.1f} nM vs extracellular 2,000,000 nM "
          f"-> {'plausible' if pk < 2e6 else 'IMPLAUSIBLE'}")

# ---- Flag 3: sensitivity of crossing times to K_m at 125 uM ----
print("\n3. Crossing-time spread at 125 uM (the Jarzina 1-2h imaging dose):")
t5_125 = [results[(K_m, 125.0)]['t500'] for K_m in KM_VALUES]
t1_125 = [results[(K_m, 125.0)]['t1000'] for K_m in KM_VALUES]
for K_m, a, b in zip(KM_VALUES, t5_125, t1_125):
    print(f"   K_m={K_m:>4.0f}: t>500nM={a:.2f}h, t>1000nM={b:.2f}h")
print(f"   500nM range across K_m: {min(t5_125):.2f}-{max(t5_125):.2f} h")
print(f"   1000nM range across K_m: {min(t1_125):.2f}-{max(t1_125):.2f} h")

# ---- Flag 4: NRK-52E saturation + cell-line ratio at 125 uM ----
print("\n4. NRK-52E at 125 uM and cell-line peak-C_ly ratio (RPTEC/NRK):")
for K_m in KM_VALUES:
    pN = make_params(k_uptake_NRK * (K_m + CAL_CONC), K_m, k_deg_PB)
    _, _, cN = simulate(pN, V_CELL_NRK, 125.0, t_grid, return_compartments=True)
    peakN = cN['C_ly'].max()
    peakR = results[(K_m, 125.0)]['peak_cly']
    print(f"   K_m={K_m:>4.0f}: peak_Cly RPTEC={peakR:9.1f}, NRK={peakN:8.1f}, ratio={peakR / peakN:.2f}")

print(f"\nFigures written to ./{OUTDIR}/:")
for f in ("v04_output1_fit_preservation.png", "v04_output2_lysosomal_load.png",
          "v04_output3_threshold_crossing.png", "v04_output4_peak_cly_vs_dose.png"):
    print(f"  {f}")
print("=" * 92)
