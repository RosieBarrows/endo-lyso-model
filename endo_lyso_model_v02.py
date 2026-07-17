"""
Intracellular drug accumulation model -- v0.2
Calibrated against Jarzina et al. (2022) Front. Toxicol. 4:864441, Figure 6A.

v0.2 changes (per model_v02_updated_instructions.md):
  * Polymyxin B only, two cell lines (RPTEC/TERT1, NRK-52E). All colistin work dropped.
  * New digitised data with a t=0 constraint and asymmetric RPTEC error bars.
  * Fitting: log-space residuals (t=0 excluded). Step 1 (RPTEC: k_uptake, k_deg),
    Step 2A (NRK: k_uptake only, k_deg fixed), Step 2B (NRK: k_uptake + k_deg free).
  * Stress-test: vary each fixed trafficking param over 5 values, refit Steps 1 & 2A.

Unchanged from v0.1: 3-compartment ODE structure, fixed trafficking params, BDF
solver, and the CORRECTED fmol/cell -> nM conversion (factor 1e-6, not 1e6).

Per user direction: the final time point is evaluated at t = 48 h (not 47.46 h).
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

np.set_printoptions(suppress=True)

# ---------------------------------------------------------------------------
# 2. Constants and fixed parameters
# ---------------------------------------------------------------------------
T_WASHOUT_MIN = 24.0 * 60.0      # drug removed at 24 h
SIM_END_MIN   = 50.0 * 60.0      # integrate washout phase out to 50 h (for plots)
C_EXT_UM      = 34.0             # extracellular conc during exposure (uM)

V_CELL_RPTEC = 1.99e-12          # L, RPTEC/TERT1 (d ~ 15.6 um)
V_CELL_NRK   = 1.21e-12          # L, NRK-52E     (d ~ 13.2 um)

# Default fixed trafficking rate constants (/min) -- cell machinery, not drug.
FIXED_DEFAULT = dict(k_mat=0.04, k_rec=0.02, k_fuse=0.02, k_esc=0.0002)

# Fitted-parameter bounds / initial guesses (v0.2 wide bounds -- "let the data speak")
KUPTAKE_INIT, KUPTAKE_LO, KUPTAKE_HI = 1e-6, 1e-9, 1e-4   # fmol/cell/min/uM
KDEG_INIT,    KDEG_LO,    KDEG_HI    = 1e-3, 1e-5, 1e-1   # /min


# ---------------------------------------------------------------------------
# 3. ODE system
# ---------------------------------------------------------------------------
def odes(t, y, p, c_ext):
    """RHS of the 3-compartment system. y = [C_ee, C_le, C_ly] in fmol/cell."""
    C_ee, C_le, C_ly = y
    dC_ee = p['k_uptake'] * c_ext - (p['k_mat'] + p['k_rec']) * C_ee
    dC_le = p['k_mat'] * C_ee     - (p['k_fuse'] + p['k_esc']) * C_le
    dC_ly = p['k_fuse'] * C_le    - p['k_deg'] * C_ly
    return [dC_ee, dC_le, dC_ly]


# ---------------------------------------------------------------------------
# 4. Simulation (exposure 0-24h at 34 uM, washout 24h-> with C_ext = 0)
# ---------------------------------------------------------------------------
def simulate(p, V_cell, t_eval_h=None, n_dense=400, return_compartments=False):
    """Return total intracellular conc (nM) at requested times (hours)."""
    if t_eval_h is None:
        t_eval_h = np.linspace(0.0, 50.0, n_dense)
    t_eval_h = np.atleast_1d(np.asarray(t_eval_h, dtype=float))
    t_eval_min = t_eval_h * 60.0

    mask1 = t_eval_min <= T_WASHOUT_MIN
    mask2 = ~mask1
    y0 = [0.0, 0.0, 0.0]

    sol1 = solve_ivp(odes, (0.0, T_WASHOUT_MIN), y0, args=(p, C_EXT_UM),
                     method='BDF', dense_output=True, rtol=1e-8, atol=1e-12)
    sol2 = solve_ivp(odes, (T_WASHOUT_MIN, SIM_END_MIN), sol1.y[:, -1],
                     args=(p, 0.0), method='BDF', dense_output=True,
                     rtol=1e-8, atol=1e-12)

    Y = np.zeros((3, t_eval_min.size))
    if mask1.any():
        Y[:, mask1] = sol1.sol(t_eval_min[mask1])
    if mask2.any():
        Y[:, mask2] = sol2.sol(t_eval_min[mask2])

    Y = np.clip(Y, 0.0, None)
    # fmol/cell -> nM:  conc_nM = X_fmol * 1e-6 / V_L  (corrected factor)
    comp_nM = Y * (1e-6 / V_cell)
    total_nM = comp_nM.sum(axis=0)
    if return_compartments:
        return t_eval_h, total_nM, dict(C_ee=comp_nM[0], C_le=comp_nM[1], C_ly=comp_nM[2])
    return t_eval_h, total_nM


def make_params(k_uptake, k_deg, fixed=None):
    p = dict(FIXED_DEFAULT if fixed is None else fixed)
    p['k_uptake'] = k_uptake
    p['k_deg'] = k_deg
    return p


# ---------------------------------------------------------------------------
# 5. Residuals + fitting (log-space residuals, log10 parameters)
# ---------------------------------------------------------------------------
def residuals(theta, fit_keys, base_kuptake, base_kdeg, V_cell,
              t_data_h, y_data_nM, fixed):
    vals = {'k_uptake': base_kuptake, 'k_deg': base_kdeg}
    for key, lg in zip(fit_keys, theta):
        vals[key] = 10.0 ** lg
    p = make_params(vals['k_uptake'], vals['k_deg'], fixed)
    _, model = simulate(p, V_cell, t_eval_h=t_data_h)
    model = np.clip(model, 1e-9, None)
    return np.log10(model) - np.log10(y_data_nM)


def fit(fit_keys, base_kuptake, base_kdeg, V_cell, t_data_h, y_data_nM, fixed=None):
    """least_squares (trf) in log10 parameter space. Returns dict, log-SSE, result."""
    init, lo, hi = [], [], []
    for key in fit_keys:
        if key == 'k_uptake':
            init.append(np.log10(KUPTAKE_INIT)); lo.append(np.log10(KUPTAKE_LO)); hi.append(np.log10(KUPTAKE_HI))
        else:
            init.append(np.log10(KDEG_INIT)); lo.append(np.log10(KDEG_LO)); hi.append(np.log10(KDEG_HI))
    res = least_squares(residuals, x0=init, bounds=(lo, hi), method='trf',
                        args=(fit_keys, base_kuptake, base_kdeg, V_cell,
                              t_data_h, y_data_nM, fixed))
    fitted = {'k_uptake': base_kuptake, 'k_deg': base_kdeg}
    for key, lg in zip(fit_keys, res.x):
        fitted[key] = 10.0 ** lg
    return fitted, float(np.sum(res.fun ** 2)), res


# ---------------------------------------------------------------------------
# 6. Experimental data (v0.2 replacement; final point evaluated at t = 48 h)
# ---------------------------------------------------------------------------
T_ALL_H = np.array([0.0, 3.53, 6.44, 24.0, 48.0])

RPTEC_CENTRAL = np.array([0.0, 636.0, 1377.0, 3558.0, 2792.0])
RPTEC_LOWER   = np.array([0.0, 442.0,  870.0, 1974.0, 1494.0])
RPTEC_UPPER   = np.array([0.0, 883.0, 1909.0, 5156.0, 4065.0])

NRK_CENTRAL   = np.array([0.0, 299.0,  662.0,  831.0,  104.0])

FIT_IDX = np.array([1, 2, 3, 4])          # drop t=0 (model passes through 0 by construction)
T_FIT_H = T_ALL_H[FIT_IDX]
y_RPTEC = RPTEC_CENTRAL[FIT_IDX]
y_NRK   = NRK_CENTRAL[FIT_IDX]


# ---------------------------------------------------------------------------
# 7. Fitting: Steps 1, 2A, 2B
# ---------------------------------------------------------------------------
print("=" * 72)
print("FITTING (v0.2)")
print("=" * 72)

# Step 1: RPTEC -- free k_uptake_RPTEC, k_deg_PB
fit1, sse1, _ = fit(['k_uptake', 'k_deg'], KUPTAKE_INIT, KDEG_INIT,
                    V_CELL_RPTEC, T_FIT_H, y_RPTEC)
k_uptake_RPTEC = fit1['k_uptake']
k_deg_PB       = fit1['k_deg']
p_RPTEC = make_params(k_uptake_RPTEC, k_deg_PB)
print(f"Step 1  RPTEC : k_uptake_RPTEC={k_uptake_RPTEC:.4g}, k_deg_PB={k_deg_PB:.4g}  (SSE_log={sse1:.4f})")

# Step 2A: NRK -- free k_uptake only, k_deg fixed = k_deg_PB
fit2a, sse2a, _ = fit(['k_uptake'], KUPTAKE_INIT, k_deg_PB,
                      V_CELL_NRK, T_FIT_H, y_NRK)
k_uptake_NRK_2A = fit2a['k_uptake']
p_NRK_2A = make_params(k_uptake_NRK_2A, k_deg_PB)
print(f"Step 2A NRK   : k_uptake_NRK={k_uptake_NRK_2A:.4g}  (k_deg fixed)  (SSE_log={sse2a:.4f})")

# Step 2B: NRK -- free k_uptake AND k_deg
fit2b, sse2b, _ = fit(['k_uptake', 'k_deg'], KUPTAKE_INIT, KDEG_INIT,
                      V_CELL_NRK, T_FIT_H, y_NRK)
k_uptake_NRK_2B = fit2b['k_uptake']
k_deg_NRK       = fit2b['k_deg']
p_NRK_2B = make_params(k_uptake_NRK_2B, k_deg_NRK)
print(f"Step 2B NRK   : k_uptake_NRK={k_uptake_NRK_2B:.4g}, k_deg_NRK={k_deg_NRK:.4g}  (SSE_log={sse2b:.4f})")


# ---------------------------------------------------------------------------
# 8. Stress-test of fixed trafficking parameters
#    For each fixed param x 5 values: refit Step 1 (k_uptake_RPTEC, k_deg) and
#    Step 2A (k_uptake_NRK, using refitted k_deg). Record fitted values + ratio.
# ---------------------------------------------------------------------------
STRESS = {
    'k_mat':  [0.004, 0.01, 0.04, 0.1, 0.4],
    'k_rec':  [0.002, 0.005, 0.02, 0.05, 0.2],
    'k_fuse': [0.002, 0.005, 0.02, 0.05, 0.2],
    'k_esc':  [0.00002, 0.00005, 0.0002, 0.0005, 0.002],
}
stress_results = {}
print("\n" + "=" * 72)
print("STRESS-TEST (refit Steps 1 & 2A for each fixed-parameter value)")
print("=" * 72)
for pname, values in STRESS.items():
    rows = []
    for v in values:
        fixed = dict(FIXED_DEFAULT); fixed[pname] = v
        f1, s1, _ = fit(['k_uptake', 'k_deg'], KUPTAKE_INIT, KDEG_INIT,
                        V_CELL_RPTEC, T_FIT_H, y_RPTEC, fixed=fixed)
        f2, s2, _ = fit(['k_uptake'], KUPTAKE_INIT, f1['k_deg'],
                        V_CELL_NRK, T_FIT_H, y_NRK, fixed=fixed)
        ratio = f1['k_uptake'] / f2['k_uptake']
        rows.append(dict(val=v, k_uptake_R=f1['k_uptake'], k_deg=f1['k_deg'],
                         k_uptake_N=f2['k_uptake'], ratio=ratio, sse1=s1, sse2=s2))
    stress_results[pname] = rows
    ku = [r['k_uptake_R'] for r in rows]
    kd = [r['k_deg'] for r in rows]
    rr = [r['ratio'] for r in rows]
    print(f"  {pname:7s}: k_uptake_R fold={max(ku)/min(ku):5.2f}x, "
          f"k_deg fold={max(kd)/min(kd):5.2f}x, ratio range=[{min(rr):.2f}, {max(rr):.2f}]")


# ---------------------------------------------------------------------------
# 9. Plotting: Outputs 1, 2, 4, 5
# ---------------------------------------------------------------------------
t_dense = np.linspace(0.0, 50.0, 400)

# ---- Output 1: time-course fits (2 panels) ----
fig1, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
ymax = max(RPTEC_UPPER.max(), NRK_CENTRAL.max(),
           simulate(p_RPTEC, V_CELL_RPTEC, t_eval_h=t_dense)[1].max())
ylim = (0, 1.1 * ymax)

# RPTEC panel with asymmetric error bars
_, totR = simulate(p_RPTEC, V_CELL_RPTEC, t_eval_h=t_dense)
yerr = np.vstack([RPTEC_CENTRAL - RPTEC_LOWER, RPTEC_UPPER - RPTEC_CENTRAL])
axL.plot(t_dense, totR, '-', color='C0', lw=2, label='model fit (Step 1)')
axL.errorbar(T_ALL_H, RPTEC_CENTRAL, yerr=yerr, fmt='ko', ms=6,
             capsize=3, label='data ± digitised bounds')
axL.axvline(24, ls='--', color='grey', alpha=0.7)
axL.text(24.3, ylim[1] * 0.93, 'washout', color='grey', fontsize=8)
axL.set_title('RPTEC/TERT1 + polymyxin B')
axL.set_xlabel('time (h)'); axL.set_ylabel('intracellular conc (nM)')
axL.set_xlim(0, 50); axL.set_ylim(*ylim); axL.legend(fontsize=8, loc='upper left')

# NRK panel: Step 2A (solid) and Step 2B (dashed)
_, totN_2A = simulate(p_NRK_2A, V_CELL_NRK, t_eval_h=t_dense)
_, totN_2B = simulate(p_NRK_2B, V_CELL_NRK, t_eval_h=t_dense)
axR.plot(t_dense, totN_2A, '-', color='C1', lw=2, label='Step 2A (k_deg fixed)')
axR.plot(t_dense, totN_2B, '--', color='C3', lw=2, label='Step 2B (k_deg free)')
axR.plot(T_ALL_H, NRK_CENTRAL, 'ko', ms=6, label='data')
axR.axvline(24, ls='--', color='grey', alpha=0.7)
axR.text(24.3, ylim[1] * 0.93, 'washout', color='grey', fontsize=8)
axR.set_title('NRK-52E + polymyxin B')
axR.set_xlabel('time (h)'); axR.set_ylabel('intracellular conc (nM)')
axR.set_xlim(0, 50); axR.set_ylim(*ylim); axR.legend(fontsize=8, loc='upper left')
fig1.suptitle('Output 1: Time-course fits (v0.2) - Jarzina (2022) Fig 6A', fontsize=13)
fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('v02_output1_timecourse.png', dpi=150)

# ---- Output 2: compartment breakdown (RPTEC only) ----
_, _, compsR = simulate(p_RPTEC, V_CELL_RPTEC, t_eval_h=t_dense, return_compartments=True)
fig2, ax2 = plt.subplots(figsize=(8, 5.5))
ax2.stackplot(t_dense, compsR['C_ee'], compsR['C_le'], compsR['C_ly'],
              labels=['early endosome', 'late endosome', 'lysosome'],
              colors=['#9ecae1', '#fdae6b', '#a1d99b'], alpha=0.9)
ax2.axvline(24, ls='--', color='grey', alpha=0.7)
ax2.set_title('Output 2: Compartmental breakdown (RPTEC/TERT1 + PB)')
ax2.set_xlabel('time (h)'); ax2.set_ylabel('intracellular conc (nM)')
ax2.set_xlim(0, 50); ax2.legend(fontsize=9, loc='upper left')
fig2.tight_layout()
fig2.savefig('v02_output2_compartments.png', dpi=150)

# ---- Output 4: stress-test (4 subplots, one per fixed param) ----
fig4, axes4 = plt.subplots(2, 2, figsize=(13, 9))
for ax, (pname, rows) in zip(axes4.ravel(), stress_results.items()):
    vals = [r['val'] for r in rows]
    ax.plot(vals, [r['k_uptake_R'] for r in rows], 'o-', color='C0', label='k_uptake_RPTEC')
    ax.plot(vals, [r['k_deg'] for r in rows], 's-', color='C2', label='k_deg')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axvline(FIXED_DEFAULT[pname], ls='--', color='grey', alpha=0.7)
    ax.set_xlabel(f'{pname} (fixed value)')
    ax.set_ylabel('fitted rate constant (log)')
    ax.set_title(f'Vary {pname}')
    ax2r = ax.twinx()
    ratios = [r['ratio'] for r in rows]
    ax2r.plot(vals, ratios, '^:', color='C3', label='ratio k_uptake R/N')
    ax2r.set_ylabel('k_uptake_RPTEC / k_uptake_NRK', color='C3')
    ax2r.tick_params(axis='y', labelcolor='C3')
    # Fixed range so the (constant) ratio is not exaggerated by autoscaling noise.
    ax2r.set_ylim(0, max(ratios) * 1.4)
    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='best')
fig4.suptitle('Output 4: Stress-test of fixed trafficking parameters', fontsize=13)
fig4.tight_layout(rect=[0, 0, 1, 0.96])
fig4.savefig('v02_output4_stresstest.png', dpi=150)

# ---- Output 5: lysosomal load (RPTEC) with thresholds ----
C_ly = compsR['C_ly']
fig5, ax5 = plt.subplots(figsize=(9, 6))
ax5.plot(t_dense, C_ly, '-', color='#238b45', lw=2, label='lysosomal conc (C_ly)')
ax5.axvline(24, ls='--', color='grey', alpha=0.7)
ax5.text(24.3, C_ly.max() * 0.95, 'washout', color='grey', fontsize=8)
cross_info = []
for thr in (500.0, 1000.0, 2000.0, 3000.0):
    ax5.axhline(thr, ls=':', color='grey')
    above = np.where(C_ly >= thr)[0]
    if above.size:
        tc = t_dense[above[0]]
        cross_info.append((thr, tc))
        ax5.plot(tc, thr, 'rv')
        ax5.annotate(f'{int(thr)} nM @ {tc:.1f} h', (tc, thr),
                     textcoords='offset points', xytext=(6, 5), fontsize=8)
    else:
        cross_info.append((thr, None))
        ax5.annotate(f'{int(thr)} nM: not reached', (1, thr),
                     textcoords='offset points', xytext=(6, 5), fontsize=8, color='grey')
ax5.set_xlabel('time (h)'); ax5.set_ylabel('lysosomal concentration (nM)')
ax5.set_xlim(0, 50)
ax5.set_title('Output 5: Lysosomal load trajectory (RPTEC/TERT1 + PB)')
ax5.legend(loc='lower right')
fig5.tight_layout()
fig5.savefig('v02_output5_lysosomal_load.png', dpi=150)
plt.close('all')


# ---------------------------------------------------------------------------
# 10. Output 3: parameter summary table  +  commentary helpers
# ---------------------------------------------------------------------------
def half_life_h(k_per_min):
    """Lysosomal half-life (hours) from a degradation rate constant (/min)."""
    return np.log(2.0) / k_per_min / 60.0

def comp_frac_at(p, V, t_h=24.0):
    _, tot, comps = simulate(p, V, t_eval_h=[t_h], return_compartments=True)
    return {k: comps[k][0] / tot[0] for k in comps}

print("\n" + "=" * 72)
print("OUTPUT 3: PARAMETER SUMMARY (v0.2)")
print("=" * 72)
rows = [
    ("k_mat (fixed)",                          f"{FIXED_DEFAULT['k_mat']:.4g} /min"),
    ("k_rec (fixed)",                          f"{FIXED_DEFAULT['k_rec']:.4g} /min"),
    ("k_fuse (fixed)",                         f"{FIXED_DEFAULT['k_fuse']:.4g} /min"),
    ("k_esc (fixed)",                          f"{FIXED_DEFAULT['k_esc']:.4g} /min"),
    ("k_uptake_RPTEC (Step 1)",                f"{k_uptake_RPTEC:.4g} fmol/cell/min/uM"),
    ("k_deg_PB (Step 1)",                      f"{k_deg_PB:.4g} /min"),
    ("k_uptake_NRK (Step 2A, k_deg fixed)",    f"{k_uptake_NRK_2A:.4g} fmol/cell/min/uM"),
    ("k_uptake_NRK (Step 2B, k_deg free)",     f"{k_uptake_NRK_2B:.4g} fmol/cell/min/uM"),
    ("k_deg_NRK (Step 2B)",                    f"{k_deg_NRK:.4g} /min"),
    ("Ratio k_uptake_RPTEC/NRK (Step 2A)",     f"{k_uptake_RPTEC / k_uptake_NRK_2A:.2f}"),
    ("Ratio k_uptake_RPTEC/NRK (Step 2B)",     f"{k_uptake_RPTEC / k_uptake_NRK_2B:.2f}"),
    ("Ratio k_deg_NRK / k_deg_PB (Step 2B)",   f"{k_deg_NRK / k_deg_PB:.2f}"),
    ("Lysosomal half-life from k_deg_PB",      f"{half_life_h(k_deg_PB):.2f} h"),
    ("Lysosomal half-life from k_deg_NRK",     f"{half_life_h(k_deg_NRK):.2f} h"),
    ("SSE Step 1 (log10)",                     f"{sse1:.4f}"),
    ("SSE Step 2A (log10)",                    f"{sse2a:.4f}"),
    ("SSE Step 2B (log10)",                    f"{sse2b:.4f}"),
]
for label, val in rows:
    print(f"  {label:42s}: {val}")

# RPTEC compartment fractions at 24 h
fr = comp_frac_at(p_RPTEC, V_CELL_RPTEC, 24.0)
print("\nRPTEC/TERT1 drug localisation at 24 h:")
for k in ('C_ee', 'C_le', 'C_ly'):
    print(f"  {k}: {fr[k] * 100:5.1f}%")

# Does the RPTEC fit fall within the error bars?
_, fitR_at = simulate(p_RPTEC, V_CELL_RPTEC, t_eval_h=T_FIT_H)
print("\nRPTEC fit vs error bars (per fitted point):")
for t, m, lo, hi in zip(T_FIT_H, fitR_at, RPTEC_LOWER[FIT_IDX], RPTEC_UPPER[FIT_IDX]):
    inside = "within" if lo <= m <= hi else "OUTSIDE"
    print(f"  t={t:5.2f}h : model={m:7.1f} nM  [{lo:.0f}, {hi:.0f}]  -> {inside}")

# Stress-test fold-change summary statements
print("\nStress-test fold-changes across each tested range:")
for pname, rrows in stress_results.items():
    ku = [r['k_uptake_R'] for r in rrows]; kd = [r['k_deg'] for r in rrows]
    rr = [r['ratio'] for r in rrows]
    print(f"  Varying {pname} 100x -> k_uptake_RPTEC {max(ku)/min(ku):.2f}x, "
          f"k_deg {max(kd)/min(kd):.2f}x, ratio {min(rr):.2f}-{max(rr):.2f}")

print("\nFigures written: v02_output1_timecourse.png, v02_output2_compartments.png,")
print("                 v02_output4_stresstest.png, v02_output5_lysosomal_load.png")
print("=" * 72)
