"""
Concentration sweep for the v0.2 intracellular accumulation model.

Runs the calibrated polymyxin B model at additional extracellular concentrations
(125, 250, 500, 1000, 2000 uM) alongside the 34 uM calibration dose, for both
cell lines (RPTEC/TERT1 and NRK-52E).

IMPORTANT CAVEAT
----------------
The model uses a LINEAR uptake term (uptake = k_uptake * C_ext), the V_max/K_m
lumped approximation valid only when C_ext << K_m. It was calibrated at 34 uM.
At 125-2000 uM there is no saturation, so intracellular concentration scales
EXACTLY LINEARLY with dose. These curves are extrapolations beyond the calibrated
range, NOT mechanistic predictions -- real uptake would saturate. A saturating
version would require an assumed K_m (not identifiable from single-dose data).

Parameters are re-derived here via the v0.2 Step 1 / Step 2A fits (no stress-test),
so this script is self-contained. Figures are written to ./concentration_sweep/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

OUTDIR = "concentration_sweep"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants / fixed parameters (same as v0.2)
# ---------------------------------------------------------------------------
T_WASHOUT_MIN = 24.0 * 60.0
SIM_END_MIN   = 50.0 * 60.0
V_CELL_RPTEC  = 1.99e-12
V_CELL_NRK    = 1.21e-12
FIXED = dict(k_mat=0.04, k_rec=0.02, k_fuse=0.02, k_esc=0.0002)

CAL_CONC = 34.0                                   # calibration dose (uM)
SWEEP_CONC = [34.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]


# ---------------------------------------------------------------------------
# Core model (C_ext now a free argument)
# ---------------------------------------------------------------------------
def odes(t, y, p, c_ext):
    C_ee, C_le, C_ly = y
    dC_ee = p['k_uptake'] * c_ext - (p['k_mat'] + p['k_rec']) * C_ee
    dC_le = p['k_mat'] * C_ee     - (p['k_fuse'] + p['k_esc']) * C_le
    dC_ly = p['k_fuse'] * C_le    - p['k_deg'] * C_ly
    return [dC_ee, dC_le, dC_ly]


def simulate(p, V_cell, c_ext_um, t_eval_h=None, return_compartments=False):
    if t_eval_h is None:
        t_eval_h = np.linspace(0.0, 50.0, 400)
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
    comp_nM = Y * (1e-6 / V_cell)                 # corrected fmol/cell -> nM
    total = comp_nM.sum(axis=0)
    if return_compartments:
        return t_eval_h, total, dict(C_ee=comp_nM[0], C_le=comp_nM[1], C_ly=comp_nM[2])
    return t_eval_h, total


def make_params(k_uptake, k_deg):
    p = dict(FIXED); p['k_uptake'] = k_uptake; p['k_deg'] = k_deg
    return p


# ---------------------------------------------------------------------------
# Re-derive calibrated parameters (v0.2 Step 1 + Step 2A), at 34 uM
# ---------------------------------------------------------------------------
T_FIT_H = np.array([3.53, 6.44, 24.0, 48.0])
Y_RPTEC = np.array([636.0, 1377.0, 3558.0, 2792.0])
Y_NRK   = np.array([299.0,  662.0,  831.0,  104.0])
KU_INIT, KU_LO, KU_HI = 1e-6, 1e-9, 1e-4
KD_INIT, KD_LO, KD_HI = 1e-3, 1e-5, 1e-1


def residuals(theta, fit_keys, base_ku, base_kd, V_cell, y_data):
    vals = {'k_uptake': base_ku, 'k_deg': base_kd}
    for key, lg in zip(fit_keys, theta):
        vals[key] = 10.0 ** lg
    _, model = simulate(make_params(vals['k_uptake'], vals['k_deg']),
                        V_cell, CAL_CONC, t_eval_h=T_FIT_H)
    return np.log10(np.clip(model, 1e-9, None)) - np.log10(y_data)


def fit(fit_keys, base_ku, base_kd, V_cell, y_data):
    init, lo, hi = [], [], []
    for key in fit_keys:
        if key == 'k_uptake':
            init.append(np.log10(KU_INIT)); lo.append(np.log10(KU_LO)); hi.append(np.log10(KU_HI))
        else:
            init.append(np.log10(KD_INIT)); lo.append(np.log10(KD_LO)); hi.append(np.log10(KD_HI))
    res = least_squares(residuals, x0=init, bounds=(lo, hi), method='trf',
                        args=(fit_keys, base_ku, base_kd, V_cell, y_data))
    out = {'k_uptake': base_ku, 'k_deg': base_kd}
    for key, lg in zip(fit_keys, res.x):
        out[key] = 10.0 ** lg
    return out

f1 = fit(['k_uptake', 'k_deg'], KU_INIT, KD_INIT, V_CELL_RPTEC, Y_RPTEC)
k_uptake_RPTEC, k_deg_PB = f1['k_uptake'], f1['k_deg']
f2 = fit(['k_uptake'], KU_INIT, k_deg_PB, V_CELL_NRK, Y_NRK)
k_uptake_NRK = f2['k_uptake']

p_RPTEC = make_params(k_uptake_RPTEC, k_deg_PB)
p_NRK   = make_params(k_uptake_NRK, k_deg_PB)

print("Re-derived calibrated parameters (34 uM):")
print(f"  k_uptake_RPTEC = {k_uptake_RPTEC:.4g}, k_deg_PB = {k_deg_PB:.4g}")
print(f"  k_uptake_NRK   = {k_uptake_NRK:.4g}")

CELLS = [
    ("RPTEC/TERT1", "RPTEC", p_RPTEC, V_CELL_RPTEC),
    ("NRK-52E",     "NRK",   p_NRK,   V_CELL_NRK),
]
THRESHOLDS = [500.0, 1000.0, 2000.0, 3000.0]
t_dense = np.linspace(0.0, 50.0, 600)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(SWEEP_CONC)))


# ---------------------------------------------------------------------------
# Figure 1: total intracellular conc vs time, all concentrations (per cell line)
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (name, tag, p, V) in zip(axes1, CELLS):
    for c, col in zip(SWEEP_CONC, colors):
        _, tot = simulate(p, V, c, t_eval_h=t_dense)
        lbl = f"{c:.0f} uM" + (" (calibration)" if c == CAL_CONC else "")
        ax.plot(t_dense, tot, color=col, lw=2, label=lbl)
    ax.axvline(24, ls='--', color='grey', alpha=0.7)
    ax.set_yscale('log')
    ax.set_title(f"{name} + polymyxin B")
    ax.set_xlabel("time (h)"); ax.set_ylabel("total intracellular conc (nM)")
    ax.set_xlim(0, 50); ax.legend(fontsize=7, loc='lower right')
fig1.suptitle("Concentration sweep: total intracellular accumulation "
              "(LINEAR model -- scales with dose; extrapolation > 34 uM)", fontsize=12)
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(os.path.join(OUTDIR, "sweep_total_timecourse.png"), dpi=150)

# ---------------------------------------------------------------------------
# Figure 2: lysosomal load (C_ly) vs time, all concentrations (per cell line)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (name, tag, p, V) in zip(axes2, CELLS):
    for c, col in zip(SWEEP_CONC, colors):
        _, _, comps = simulate(p, V, c, t_eval_h=t_dense, return_compartments=True)
        lbl = f"{c:.0f} uM" + (" (calibration)" if c == CAL_CONC else "")
        ax.plot(t_dense, comps['C_ly'], color=col, lw=2, label=lbl)
    for thr in THRESHOLDS:
        ax.axhline(thr, ls=':', color='grey', alpha=0.6)
    ax.axvline(24, ls='--', color='grey', alpha=0.7)
    ax.set_yscale('log')
    ax.set_title(f"{name} + polymyxin B -- lysosomal load")
    ax.set_xlabel("time (h)"); ax.set_ylabel("lysosomal conc C_ly (nM)")
    ax.set_xlim(0, 50); ax.legend(fontsize=7, loc='lower right')
fig2.suptitle("Concentration sweep: lysosomal load (dotted lines = KE1 thresholds)",
              fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(os.path.join(OUTDIR, "sweep_lysosomal_load.png"), dpi=150)

# ---------------------------------------------------------------------------
# Figure 3: threshold-crossing time vs concentration (RPTEC), AOP/KE1 view
# ---------------------------------------------------------------------------
def first_crossing_h(t, y, thr):
    above = np.where(y >= thr)[0]
    return t[above[0]] if above.size else None

fig3, ax3 = plt.subplots(figsize=(8, 6))
for thr in THRESHOLDS:
    xs, ys = [], []
    for c in SWEEP_CONC:
        _, _, comps = simulate(p_RPTEC, V_CELL_RPTEC, c, t_eval_h=t_dense,
                               return_compartments=True)
        tc = first_crossing_h(t_dense, comps['C_ly'], thr)
        if tc is not None:
            xs.append(c); ys.append(tc)
    ax3.plot(xs, ys, 'o-', label=f"C_ly = {int(thr)} nM")
ax3.set_xscale('log')
ax3.set_xlabel("extracellular concentration (uM)")
ax3.set_ylabel("time to cross lysosomal threshold (h)")
ax3.set_title("RPTEC/TERT1: lysosomal threshold-crossing time vs dose\n"
              "(proxy for KE1 onset; linear-uptake extrapolation)")
ax3.legend(fontsize=9)
ax3.grid(True, which='both', alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTDIR, "sweep_threshold_crossing_RPTEC.png"), dpi=150)
plt.close('all')


# ---------------------------------------------------------------------------
# Summary table (printed)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CONCENTRATION SWEEP SUMMARY  (linear model -- values scale with dose)")
print("=" * 78)
for name, tag, p, V in CELLS:
    print(f"\n{name} + polymyxin B:")
    print(f"  {'C_ext(uM)':>10} {'peak_total(nM)':>16} {'C@24h(nM)':>12} "
          f"{'peak_Cly(nM)':>14} {'t>500nM Cly(h)':>16}")
    for c in SWEEP_CONC:
        _, tot, comps = simulate(p, V, c, t_eval_h=t_dense, return_compartments=True)
        _, tot24 = simulate(p, V, c, t_eval_h=[24.0])
        tc = first_crossing_h(t_dense, comps['C_ly'], 500.0)
        tc_s = f"{tc:.2f}" if tc is not None else "never"
        print(f"  {c:>10.0f} {tot.max():>16.1f} {tot24[0]:>12.1f} "
              f"{comps['C_ly'].max():>14.1f} {tc_s:>16}")

print("\nNOTE: total intracellular conc scales linearly with C_ext (no saturation).")
print("      At 2000 uM the model predicts ~hundreds of uM intracellular -- this is")
print("      an extrapolation; real uptake would saturate (Michaelis-Menten).")
print("\nFigures written to ./concentration_sweep/:")
for f in ("sweep_total_timecourse.png", "sweep_lysosomal_load.png",
          "sweep_threshold_crossing_RPTEC.png"):
    print(f"  {f}")
print("=" * 78)
