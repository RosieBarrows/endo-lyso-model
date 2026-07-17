"""
Intracellular drug accumulation model (endosome -> late endosome -> lysosome)
Calibrated against Jarzina et al. (2022) Front. Toxicol. 4:864441, Figure 6A.

Implements the model described in model_implementation_instructions.md, using the
digitised data supplied in jarzina_data.xlsx (NOT the placeholder values in the
instructions).

Fitting choices (per user direction):
  * Residuals are computed in log10 space (relative error), so all points across
    the wide dynamic range (50 -> 3600 nM) are balanced.
  * The 1-minute points (all ~50 nM = LOD) are DROPPED from the fit; we fit to the
    2 h, 6 h, 24 h and 48 h points. The 1-min point is still plotted for reference.
  * The "~2 h (Unlabeled tick)" point is treated as t = 2.0 h.

Author: generated for the ORCA intracellular module calibration.
"""

# ---------------------------------------------------------------------------
# 1. Import statements
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

np.set_printoptions(suppress=True)

# ---------------------------------------------------------------------------
# 2. Constants and fixed parameters (clearly labelled)
# ---------------------------------------------------------------------------

# --- Exposure / washout timing (minutes; rate constants are per minute) ---
T_WASHOUT_MIN = 24.0 * 60.0      # drug removed at 24 h
T_END_MIN     = 48.0 * 60.0      # simulation ends at 48 h
C_EXT_UM      = 34.0             # extracellular conc during exposure (uM)

# --- Cell volumes (litres), from Jarzina trypsinised-cell imaging ---
V_CELL_RPTEC = 1.99e-12          # RPTEC/TERT1, d ~ 15.6 um
V_CELL_NRK   = 1.21e-12          # NRK-52E,     d ~ 13.2 um

# --- Fixed trafficking rate constants (/min) -- properties of the cell, ---
# --- not the drug.  Held constant across all conditions.                ---
K_MAT  = 0.04      # EE -> LE maturation
K_REC  = 0.02      # EE -> surface recycling
K_FUSE = 0.02      # LE -> lysosome fusion
K_ESC  = 0.0002    # LE -> cytosol escape (~1% of fusion)

FIXED = dict(k_mat=K_MAT, k_rec=K_REC, k_fuse=K_FUSE, k_esc=K_ESC)

# --- Fitted-parameter bounds and initial guesses (linear space) ---
# NOTE: the instructions' k_uptake bounds [0.001, 0.1] were calibrated to the
# (incorrect, off-by-1e12) unit conversion. With the physically correct fmol->nM
# conversion, and given that the Jarzina data show intracellular < extracellular
# concentration (modest, non-concentrative uptake), the required k_uptake is
# ~1e-6. The lower bound is widened accordingly so the optimiser can reach it.
KUPTAKE_INIT, KUPTAKE_LO, KUPTAKE_HI = 1e-5, 1e-8, 1e-2      # fmol/cell/min/uM
KDEG_INIT,    KDEG_LO,    KDEG_HI    = 0.002, 0.0001, 0.05    # /min


# ---------------------------------------------------------------------------
# 3. ODE system function
# ---------------------------------------------------------------------------
def odes(t, y, p, c_ext):
    """Right-hand side of the 3-compartment system.

    y       = [C_ee, C_le, C_ly] in fmol/cell
    p       = dict of rate constants (k_uptake, k_mat, k_rec, k_fuse, k_esc, k_deg)
    c_ext   = extracellular concentration (uM), constant within a phase
    """
    C_ee, C_le, C_ly = y

    dC_ee = p['k_uptake'] * c_ext - (p['k_mat'] + p['k_rec']) * C_ee
    dC_le = p['k_mat'] * C_ee     - (p['k_fuse'] + p['k_esc']) * C_le
    dC_ly = p['k_fuse'] * C_le    - p['k_deg'] * C_ly

    return [dC_ee, dC_le, dC_ly]


# ---------------------------------------------------------------------------
# 4. Simulation function
# ---------------------------------------------------------------------------
def simulate(p, V_cell, t_eval_h=None, n_dense=400, return_compartments=False):
    """Run the exposure + washout phases and return total intracellular conc (nM).

    Two sequential solve_ivp calls (stiff BDF solver):
        phase 1: 0 -> 24 h with C_ext = 34 uM
        phase 2: 24 h -> 48 h with C_ext = 0
    The final state of phase 1 seeds phase 2.

    Parameters
    ----------
    p        : dict of all six rate constants
    V_cell   : cell volume (L)
    t_eval_h : array of times (hours) at which to return concentrations.
               If None, a dense grid over 0-48 h is used.
    return_compartments : if True, also return per-compartment nM arrays.

    Returns
    -------
    t_h   : times (hours)
    total : total intracellular concentration (nM) = (C_ee+C_le+C_ly)/V_cell*1e6
    (optional) dict of compartment concentrations in nM
    """
    if t_eval_h is None:
        t_eval_h = np.linspace(0.0, 48.0, n_dense)
    t_eval_h = np.atleast_1d(np.asarray(t_eval_h, dtype=float))
    t_eval_min = t_eval_h * 60.0

    # Split requested times into the two phases.
    mask1 = t_eval_min <= T_WASHOUT_MIN
    mask2 = ~mask1
    teval1 = t_eval_min[mask1]
    teval2 = t_eval_min[mask2]

    y0 = [0.0, 0.0, 0.0]   # all compartments start empty

    # ---- Phase 1: exposure ----
    sol1 = solve_ivp(odes, (0.0, T_WASHOUT_MIN), y0,
                     args=(p, C_EXT_UM), method='BDF',
                     dense_output=True, rtol=1e-8, atol=1e-12)
    y_at_washout = sol1.y[:, -1]

    # ---- Phase 2: washout (C_ext = 0) ----
    sol2 = solve_ivp(odes, (T_WASHOUT_MIN, T_END_MIN), y_at_washout,
                     args=(p, 0.0), method='BDF',
                     dense_output=True, rtol=1e-8, atol=1e-12)

    # Evaluate compartments at requested times via dense output.
    Y = np.zeros((3, t_eval_min.size))
    if teval1.size:
        Y[:, mask1] = sol1.sol(teval1)
    if teval2.size:
        Y[:, mask2] = sol2.sol(teval2)

    Y = np.clip(Y, 0.0, None)             # guard against tiny negative undershoot
    # fmol/cell -> nM:  X fmol = X*1e-15 mol; conc(M) = X*1e-15 / V_L; *1e9 -> nM
    # => conc_nM = X * 1e-6 / V_L.  (The instructions' "/V_cell * 1e6" is a typo,
    #    off by 1e12; the physically correct factor is * 1e-6.)
    to_nM = 1e-6 / V_cell                 # fmol/cell -> nM
    comp_nM = Y * to_nM
    total_nM = comp_nM.sum(axis=0)

    if return_compartments:
        comps = dict(C_ee=comp_nM[0], C_le=comp_nM[1], C_ly=comp_nM[2])
        return t_eval_h, total_nM, comps
    return t_eval_h, total_nM


def make_params(k_uptake, k_deg):
    """Assemble a full parameter dict from the two drug/cell-specific values."""
    p = dict(FIXED)
    p['k_uptake'] = k_uptake
    p['k_deg'] = k_deg
    return p


# ---------------------------------------------------------------------------
# 5. Residual function for fitting (log-space residuals, log-space parameters)
# ---------------------------------------------------------------------------
def residuals(theta, fit_keys, base_kuptake, base_kdeg,
              V_cell, t_data_h, y_data_nM):
    """Vector of log10(model) - log10(data) residuals.

    theta      : log10 of the free parameters (length 1 or 2)
    fit_keys   : list naming which params are free, e.g. ['k_uptake','k_deg']
    base_*     : values used for parameters that are held fixed in this fit
    """
    vals = {'k_uptake': base_kuptake, 'k_deg': base_kdeg}
    for key, lg in zip(fit_keys, theta):
        vals[key] = 10.0 ** lg
    p = make_params(vals['k_uptake'], vals['k_deg'])

    _, model_nM = simulate(p, V_cell, t_eval_h=t_data_h)
    model_nM = np.clip(model_nM, 1e-6, None)
    return np.log10(model_nM) - np.log10(y_data_nM)


def fit(fit_keys, base_kuptake, base_kdeg, V_cell, t_data_h, y_data_nM):
    """Run least_squares (trf, bounded) in log10 parameter space.

    Returns fitted-value dict, SSE in log-space, and the OptimizeResult.
    """
    init, lo, hi = [], [], []
    for key in fit_keys:
        if key == 'k_uptake':
            init.append(np.log10(KUPTAKE_INIT)); lo.append(np.log10(KUPTAKE_LO)); hi.append(np.log10(KUPTAKE_HI))
        else:  # k_deg
            init.append(np.log10(KDEG_INIT));    lo.append(np.log10(KDEG_LO));    hi.append(np.log10(KDEG_HI))

    res = least_squares(residuals, x0=init, bounds=(lo, hi), method='trf',
                        args=(fit_keys, base_kuptake, base_kdeg, V_cell, t_data_h, y_data_nM))

    fitted = {'k_uptake': base_kuptake, 'k_deg': base_kdeg}
    for key, lg in zip(fit_keys, res.x):
        fitted[key] = 10.0 ** lg
    sse_log = float(np.sum(res.fun ** 2))
    return fitted, sse_log, res


def sse_linear(p, V_cell, t_data_h, y_data_nM):
    """Plain linear SSE (nM^2) for reporting alongside the log-space objective."""
    _, model = simulate(p, V_cell, t_eval_h=t_data_h)
    return float(np.sum((model - y_data_nM) ** 2))


# ---------------------------------------------------------------------------
# 6. Experimental data (from jarzina_data.xlsx)
# ---------------------------------------------------------------------------
# Full digitised set (5 points). 1-min point = ~50 nM = LOD for every condition.
DATA_T_ALL_H = np.array([1.0 / 60.0, 2.0, 6.0, 24.0, 48.0])
DATA = {
    'RPTEC_PB':       np.array([50.0,  650.0, 1400.0, 3600.0, 2800.0]),
    'NRK_PB':         np.array([50.0,  300.0,  650.0,  850.0,  100.0]),
    'RPTEC_colistin': np.array([50.0,  400.0,  750.0, 1900.0, 1000.0]),
    'NRK_colistin':   np.array([50.0,  350.0,  350.0,  300.0,   50.0]),
}
# Fit mask: drop the 1-min LOD point (index 0); fit the remaining 4.
FIT_IDX = np.array([1, 2, 3, 4])
DATA_T_FIT_H = DATA_T_ALL_H[FIT_IDX]


# ---------------------------------------------------------------------------
# 7. Fitting: Steps 1-4
# ---------------------------------------------------------------------------
print("=" * 70)
print("FITTING")
print("=" * 70)

# ---- Step 1: RPTEC/TERT1 + polymyxin B  (free: k_uptake_RPTEC, k_deg_PB) ----
y1 = DATA['RPTEC_PB'][FIT_IDX]
fit1, sse1_log, res1 = fit(['k_uptake', 'k_deg'],
                           KUPTAKE_INIT, KDEG_INIT,
                           V_CELL_RPTEC, DATA_T_FIT_H, y1)
k_uptake_RPTEC = fit1['k_uptake']
k_deg_PB       = fit1['k_deg']
p1 = make_params(k_uptake_RPTEC, k_deg_PB)
print(f"Step 1  RPTEC/PB : k_uptake_RPTEC={k_uptake_RPTEC:.4g}, k_deg_PB={k_deg_PB:.4g}  "
      f"(SSE_log={sse1_log:.3f})")

# ---- Step 2: NRK-52E + polymyxin B  (free: k_uptake_NRK52E; k_deg fixed) ----
y2 = DATA['NRK_PB'][FIT_IDX]
fit2, sse2_log, res2 = fit(['k_uptake'],
                           KUPTAKE_INIT, k_deg_PB,
                           V_CELL_NRK, DATA_T_FIT_H, y2)
k_uptake_NRK = fit2['k_uptake']
p2 = make_params(k_uptake_NRK, k_deg_PB)
print(f"Step 2  NRK/PB   : k_uptake_NRK52E={k_uptake_NRK:.4g}  (k_deg_PB fixed)  "
      f"(SSE_log={sse2_log:.3f})")

# ---- Step 3: RPTEC/TERT1 + colistin -- try BOTH single-parameter variants ----
y3 = DATA['RPTEC_colistin'][FIT_IDX]
# Variant A: free k_deg_colistin (k_uptake fixed = k_uptake_RPTEC)
fitA, sseA_log, _ = fit(['k_deg'], k_uptake_RPTEC, KDEG_INIT,
                        V_CELL_RPTEC, DATA_T_FIT_H, y3)
k_deg_colistin = fitA['k_deg']
pA = make_params(k_uptake_RPTEC, k_deg_colistin)
# Variant B: free k_uptake_colistin (k_deg fixed = k_deg_PB)
fitB, sseB_log, _ = fit(['k_uptake'], KUPTAKE_INIT, k_deg_PB,
                        V_CELL_RPTEC, DATA_T_FIT_H, y3)
k_uptake_colistin = fitB['k_uptake']
pB = make_params(k_uptake_colistin, k_deg_PB)
print(f"Step 3a RPTEC/col: free k_deg_colistin   ={k_deg_colistin:.4g}  (SSE_log={sseA_log:.3f})")
print(f"Step 3b RPTEC/col: free k_uptake_colistin={k_uptake_colistin:.4g}  (SSE_log={sseB_log:.3f})")
better = 'A (k_deg)' if sseA_log <= sseB_log else 'B (k_uptake)'
print(f"        -> better single-parameter fit: variant {better}")
# Use variant A (free k_deg_colistin) for the RPTEC/colistin panel and Step 4,
# as the instructions' Step 4 is defined in terms of k_deg_colistin.
p3 = pA

# ---- Step 4: NRK-52E + colistin -- PREDICTION (no fitting) ----
# k_uptake_NRK52E (Step 2) + k_deg_colistin (Step 3a)
p4 = make_params(k_uptake_NRK, k_deg_colistin)
y4 = DATA['NRK_colistin'][FIT_IDX]
_, pred4 = simulate(p4, V_CELL_NRK, t_eval_h=DATA_T_FIT_H)
pred4_resid_log = np.log10(np.clip(pred4, 1e-6, None)) - np.log10(y4)
pred4_sse_log = float(np.sum(pred4_resid_log ** 2))
print(f"Step 4  NRK/col  : PREDICTION (no fit) using k_uptake_NRK52E + k_deg_colistin  "
      f"(SSE_log={pred4_sse_log:.3f})")


# ---------------------------------------------------------------------------
# 8. Plotting: Outputs 1-5
# ---------------------------------------------------------------------------
PANELS = [
    ('RPTEC/TERT1 + polymyxin B',      'RPTEC_PB',       p1, V_CELL_RPTEC, False),
    ('NRK-52E + polymyxin B',          'NRK_PB',         p2, V_CELL_NRK,   False),
    ('RPTEC/TERT1 + colistin',         'RPTEC_colistin', p3, V_CELL_RPTEC, False),
    ('NRK-52E + colistin (PREDICTION)','NRK_colistin',   p4, V_CELL_NRK,   True),
]

t_dense = np.linspace(0.0, 48.0, 400)

# ---- Output 1: time-course fits ----
# Common y-limit so magnitude differences are visible.
ymax = 0.0
for _, key, p, V, _ in PANELS:
    _, tot = simulate(p, V, t_eval_h=t_dense)
    ymax = max(ymax, tot.max(), DATA[key].max())
ylim = (0, 1.1 * ymax)

fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
for ax, (title, key, p, V, is_pred) in zip(axes1.ravel(), PANELS):
    _, tot = simulate(p, V, t_eval_h=t_dense)
    ax.plot(t_dense, tot, '-', color='C0', lw=2,
            label='model (prediction)' if is_pred else 'model fit')
    # all data points (open marker for the dropped 1-min LOD point)
    ax.plot(DATA_T_ALL_H[0], DATA[key][0], 'o', mfc='white', mec='k',
            ms=7, label='data (1 min, LOD - not fitted)')
    ax.plot(DATA_T_ALL_H[FIT_IDX], DATA[key][FIT_IDX], 'ko', ms=7,
            label='data (fitted)')
    ax.axvline(24, ls='--', color='grey', alpha=0.7)
    ax.text(24.3, ylim[1] * 0.93, 'washout', color='grey', fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('intracellular conc (nM)')
    ax.set_xlim(0, 48)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=7, loc='upper left')
fig1.suptitle('Output 1: Intracellular accumulation - model vs Jarzina (2022) Fig 6A',
              fontsize=13)
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig('output1_timecourse.png', dpi=150)

# ---- Output 2: compartment breakdown (stacked area) ----
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
for ax, (title, key, p, V, _) in zip(axes2.ravel(), PANELS):
    _, tot, comps = simulate(p, V, t_eval_h=t_dense, return_compartments=True)
    ax.stackplot(t_dense, comps['C_ee'], comps['C_le'], comps['C_ly'],
                 labels=['early endosome', 'late endosome', 'lysosome'],
                 colors=['#9ecae1', '#fdae6b', '#a1d99b'], alpha=0.9)
    ax.axvline(24, ls='--', color='grey', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('intracellular conc (nM)')
    ax.set_xlim(0, 48)
    ax.legend(fontsize=7, loc='upper left')
fig2.suptitle('Output 2: Compartmental breakdown (stacked)', fontsize=13)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig('output2_compartments.png', dpi=150)

# ---- Output 4: one-at-a-time sensitivity (tornado) for RPTEC/PB ----
base = dict(p1)
_, base_24 = simulate(base, V_CELL_RPTEC, t_eval_h=[24.0])
base_24 = base_24[0]
sens_keys = ['k_uptake', 'k_mat', 'k_rec', 'k_fuse', 'k_esc', 'k_deg']
sens = []
for key in sens_keys:
    p_lo = dict(base); p_lo[key] = base[key] * 0.5
    p_hi = dict(base); p_hi[key] = base[key] * 1.5
    _, lo24 = simulate(p_lo, V_CELL_RPTEC, t_eval_h=[24.0])
    _, hi24 = simulate(p_hi, V_CELL_RPTEC, t_eval_h=[24.0])
    sens.append((key,
                 (lo24[0] - base_24) / base_24 * 100,
                 (hi24[0] - base_24) / base_24 * 100))
# order by total swing
sens.sort(key=lambda s: abs(s[1]) + abs(s[2]))
fig4, ax4 = plt.subplots(figsize=(8, 5))
ypos = np.arange(len(sens))
for i, (key, lo, hi) in enumerate(sens):
    ax4.barh(i, hi, color='#d6604d', alpha=0.8)
    ax4.barh(i, lo, color='#4393c3', alpha=0.8)
ax4.set_yticks(ypos)
ax4.set_yticklabels([s[0] for s in sens])
ax4.axvline(0, color='k', lw=0.8)
ax4.set_xlabel('% change in C_total at 24 h')
ax4.set_title('Output 4: One-at-a-time sensitivity (RPTEC/PB)\n'
              'blue = -50% parameter, red = +50% parameter')
fig4.tight_layout()
fig4.savefig('output4_sensitivity.png', dpi=150)

# ---- Output 5: lysosomal load trajectory (RPTEC/PB) with thresholds ----
_, _, comps1 = simulate(p1, V_CELL_RPTEC, t_eval_h=t_dense, return_compartments=True)
C_ly = comps1['C_ly']
fig5, ax5 = plt.subplots(figsize=(9, 6))
ax5.plot(t_dense, C_ly, '-', color='#238b45', lw=2, label='lysosomal conc (C_ly)')
ax5.axvline(24, ls='--', color='grey', alpha=0.7)
ax5.text(24.3, C_ly.max() * 0.95, 'washout', color='grey', fontsize=8)
for thr in (1000.0, 2000.0, 3000.0):
    ax5.axhline(thr, ls=':', color='grey')
    # first crossing time (rising phase)
    above = np.where(C_ly >= thr)[0]
    if above.size:
        tc = t_dense[above[0]]
        ax5.plot(tc, thr, 'rv')
        ax5.annotate(f'{int(thr)} nM @ {tc:.1f} h', (tc, thr),
                     textcoords='offset points', xytext=(6, 6), fontsize=8)
    else:
        ax5.annotate(f'{int(thr)} nM: not reached', (1, thr),
                     textcoords='offset points', xytext=(6, 6), fontsize=8, color='grey')
ax5.set_xlabel('time (h)')
ax5.set_ylabel('lysosomal concentration (nM)')
ax5.set_xlim(0, 48)
ax5.set_title('Output 5: Lysosomal load trajectory (RPTEC/TERT1 + polymyxin B)\n'
              'threshold crossings ~ onset of KE1 (lysosomal dysfunction)')
ax5.legend(loc='lower right')
fig5.tight_layout()
fig5.savefig('output5_lysosomal_load.png', dpi=150)

plt.close('all')


# ---------------------------------------------------------------------------
# 9. Output 3: parameter summary table
# ---------------------------------------------------------------------------
def frac_in_lyso(p, V):
    _, tot, comps = simulate(p, V, t_eval_h=[24.0], return_compartments=True)
    return comps['C_ly'][0] / tot[0]

print()
print("=" * 70)
print("OUTPUT 3: PARAMETER SUMMARY")
print("=" * 70)
print("\nFixed trafficking parameters (/min):")
for k, v in FIXED.items():
    print(f"  {k:10s} = {v:.4g}")

print("\nFitted parameters:")
print(f"  k_uptake_RPTEC    = {k_uptake_RPTEC:.4g}  fmol/cell/min/uM   [Step 1, RPTEC/PB]")
print(f"  k_uptake_NRK52E   = {k_uptake_NRK:.4g}  fmol/cell/min/uM   [Step 2, NRK/PB]")
print(f"  k_deg_PB          = {k_deg_PB:.4g}  /min               [Step 1, RPTEC/PB]")
print(f"  k_deg_colistin    = {k_deg_colistin:.4g}  /min               [Step 3a, RPTEC/colistin]")
print(f"  (alt) k_uptake_colistin = {k_uptake_colistin:.4g}             [Step 3b, RPTEC/colistin]")

print("\nDiagnostic ratios:")
print(f"  k_uptake_RPTEC / k_uptake_NRK52E = {k_uptake_RPTEC / k_uptake_NRK:.2f}  (expect ~3-5x)")
print(f"  k_deg_colistin / k_deg_PB        = {k_deg_colistin / k_deg_PB:.2f}")

print("\nGoodness of fit (log10-space SSE over 4 fitted points; linear SSE in nM^2):")
print(f"  Step 1 RPTEC/PB      : SSE_log={sse1_log:.3f}   SSE_lin={sse_linear(p1, V_CELL_RPTEC, DATA_T_FIT_H, y1):.3g}")
print(f"  Step 2 NRK/PB        : SSE_log={sse2_log:.3f}   SSE_lin={sse_linear(p2, V_CELL_NRK,   DATA_T_FIT_H, y2):.3g}")
print(f"  Step 3 RPTEC/colistin: SSE_log={sseA_log:.3f}   SSE_lin={sse_linear(p3, V_CELL_RPTEC, DATA_T_FIT_H, y3):.3g}")
print(f"  Step 4 NRK/colistin  : SSE_log={pred4_sse_log:.3f}   SSE_lin={sse_linear(p4, V_CELL_NRK, DATA_T_FIT_H, y4):.3g}  (PREDICTION)")

print("\nDrug localisation at 24 h (fraction in lysosome):")
for title, key, p, V, _ in PANELS:
    print(f"  {title:34s}: {frac_in_lyso(p, V) * 100:5.1f}% in lysosome")

print("\nFigures written:")
for f in ['output1_timecourse.png', 'output2_compartments.png',
          'output4_sensitivity.png', 'output5_lysosomal_load.png']:
    print(f"  {f}")
print("=" * 70)
