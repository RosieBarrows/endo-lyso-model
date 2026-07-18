"""
v0.4 -- time-varying exposure, first look: constant vs decaying C_ext at matched AUC.

Swaps the constant extracellular concentration for an analytic IV-bolus profile
C_ext(t) = C0 * exp(-k_e * t), driven through model_core_v04.simulate_profile().
No refit is involved: k_uptake and k_deg are cell/drug properties, not exposure
properties, so the frozen v0.4 RPTEC calibration is reused verbatim.

The comparison holds exposure AUC fixed. The reference is the Jarzina step protocol
(34 uM for 24 h, then washout), whose exposure AUC is 34 * 24 = 816 uM.h. Three
bolus profiles are built with the SAME total AUC (C0 = AUC * k_e) but different
decay half-lives, so they differ only in how the fixed exposure is distributed in
time -- a fast spike vs a slow smear.

The point to see: the lysosome integrates exposure. At matched AUC a fast, tall
spike and a slow, low smear land at a similar terminal lysosomal load, but the
peak C_ly is delayed and blunted relative to holding the concentration constant.

Figure -> figures/pk_profiles_v04/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import model_core_v04 as mc

OUTDIR = os.path.join("figures", "pk_profiles_v04")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Frozen v0.4 RPTEC calibration (no refit) + default K_m.
# ---------------------------------------------------------------------------
P = mc.make_params(mc.FITTED["k_uptake_RPTEC"], mc.KM_DEFAULT, mc.FITTED["k_deg_PB"])
V = mc.V_CELL["RPTEC/TERT1"]

CAL_CONC   = mc.CAL_CONC     # 34 uM
T_EXPOSE_H = 24.0            # reference exposure window
SIM_END_H  = 72.0           # watch the lysosome fill and drain
AUC_TARGET = CAL_CONC * T_EXPOSE_H       # 816 uM.h of exposure, held fixed

# Matched-AUC bolus profiles, labelled by exposure half-life. C0 = AUC * k_e keeps
# the analytic integral C0/k_e equal to AUC_TARGET for every profile.
HALF_LIVES_H = [2.0, 8.0, 24.0]
BOLUS = []
for t_half in HALF_LIVES_H:
    k_e = np.log(2.0) / t_half
    C0 = AUC_TARGET * k_e
    BOLUS.append(dict(t_half=t_half, k_e=k_e, C0=C0, fn=mc.pk_bolus(C0, k_e)))

t_grid = np.unique(np.concatenate([
    np.linspace(0.0, 6.0, 1200),
    np.linspace(6.0, SIM_END_H, 1400),
]))

# ---------------------------------------------------------------------------
# Reference: constant 34 uM step + washout, via the profile path (identical to
# simulate(); using pk_step keeps every curve on the same code path).
# ---------------------------------------------------------------------------
step = mc.pk_step(CAL_CONC, T_EXPOSE_H)
_, tot_ref, comp_ref = mc.simulate_profile(P, V, step, t_grid,
                                           sim_end_h=SIM_END_H,
                                           breakpoints=step.breakpoints)
cly_ref = comp_ref["C_ly"]

# ---------------------------------------------------------------------------
# Bolus runs.
# ---------------------------------------------------------------------------
for b in BOLUS:
    _, tot, comp = mc.simulate_profile(P, V, b["fn"], t_grid,
                                       sim_end_h=SIM_END_H,
                                       breakpoints=b["fn"].breakpoints)
    b["cly"] = comp["C_ly"]
    b["tot"] = tot
    b["cext"] = b["fn"](t_grid)
    b["auc_check"] = np.trapezoid(b["cext"], t_grid)   # sanity vs AUC_TARGET

cext_ref = step(t_grid)
auc_ref = np.trapezoid(cext_ref, t_grid)


def peak_and_time(y):
    i = int(np.argmax(y))
    return y[i], t_grid[i]


# ---------------------------------------------------------------------------
# Figure: exposure profiles (left) and lysosomal load (right).
# ---------------------------------------------------------------------------
colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(BOLUS)))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

axL.plot(t_grid, cext_ref, "k-", lw=2.2, label=f"constant 34 uM / {T_EXPOSE_H:.0f} h")
for b, col in zip(BOLUS, colors):
    axL.plot(t_grid, b["cext"], color=col, lw=2,
             label=f"bolus t½={b['t_half']:.0f} h (C0={b['C0']:.0f} uM)")
axL.set_title(f"Extracellular exposure C_ext(t)\n(all matched to AUC = {AUC_TARGET:.0f} uM·h)")
axL.set_xlabel("time (h)"); axL.set_ylabel("C_ext (uM)")
axL.set_xlim(0, SIM_END_H); axL.legend(fontsize=8)

axR.plot(t_grid, cly_ref, "k-", lw=2.2, label="constant 34 uM")
for b, col in zip(BOLUS, colors):
    axR.plot(t_grid, b["cly"], color=col, lw=2, label=f"bolus t½={b['t_half']:.0f} h")
axR.set_title("Lysosomal load C_ly(t) at matched exposure AUC")
axR.set_xlabel("time (h)"); axR.set_ylabel("lysosomal C_ly (nM)")
axR.set_xlim(0, SIM_END_H); axR.legend(fontsize=8)

fig.suptitle("v0.4 first look: constant vs decaying exposure (RPTEC/TERT1, frozen calibration)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
outpath = os.path.join(OUTDIR, "v04_pk_profile_comparison.png")
fig.savefig(outpath, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary.
# ---------------------------------------------------------------------------
print("=" * 78)
print("v0.4 constant-vs-bolus exposure at matched AUC (RPTEC/TERT1)")
print("=" * 78)
print(f"target exposure AUC = {AUC_TARGET:.1f} uM.h "
      f"(constant-step numeric AUC over grid = {auc_ref:.1f})")
print(f"{'profile':>22} {'C0(uM)':>8} {'AUC(uM.h)':>10} "
      f"{'peakC_ly(nM)':>13} {'t_peak(h)':>10} {'C_ly@72h(nM)':>13}")

pk, tp = peak_and_time(cly_ref)
print(f"{'constant 34 uM':>22} {CAL_CONC:>8.1f} {auc_ref:>10.1f} "
      f"{pk:>13.1f} {tp:>10.2f} {cly_ref[-1]:>13.1f}")
for b in BOLUS:
    pk, tp = peak_and_time(b["cly"])
    print(f"{'bolus t½=' + format(b['t_half'], '.0f') + ' h':>22} "
          f"{b['C0']:>8.1f} {b['auc_check']:>10.1f} "
          f"{pk:>13.1f} {tp:>10.2f} {b['cly'][-1]:>13.1f}")

print(f"\nfigure written to {outpath}")
print("=" * 78)
