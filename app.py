"""
Streamlit explorer for the endo-lysosomal drug accumulation model.

Lets you interactively vary dose, K_m (Michaelis-Menten uptake saturation),
and the fixed trafficking rate constants, and see the effect on:
  - the total/compartmental time course (vs. the Jarzina et al. 2022 calibration data)
  - the lysosomal load trajectory and threshold-crossing times
  - a dose-response sweep of peak lysosomal load / threshold-crossing time

Uses frozen v0.4 fitted parameters (k_uptake, k_deg) as defaults -- no live
refitting. v0.4 adopts the Shipman et al. (2022) proximal-tubule trafficking
rate constants (k_mat, k_fuse) and treats k_rec as an uncertain 0.02-0.046 /min
band (nominal 0.02). For NRK-52E, a checkbox switches between the Step 2A fit
(k_deg fixed to the RPTEC value) and the Step 2B fit (k_deg free).
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import model_core_v04 as mc

st.set_page_config(page_title="Endo-lysosomal accumulation model", layout="wide")

CELL_LINES = list(mc.V_CELL.keys())
KUPTAKE_KEY = {"RPTEC/TERT1": "k_uptake_RPTEC", "NRK-52E": "k_uptake_NRK"}
DATA_CENTRAL = {"RPTEC/TERT1": mc.RPTEC_CENTRAL, "NRK-52E": mc.NRK_CENTRAL}

# Exposure-protocol / PK-profile menu labels (kept as constants so the sidebar
# widgets, session-state defaults, and downstream branching can't drift apart).
EXPOSURE_CONST = "Constant + washout (in vitro protocol)"
EXPOSURE_PROFILE = "Time-varying (analytic PK profile)"
PROFILE_BOLUS = "IV bolus (single, decaying)"
PROFILE_TRAIN = "Repeated IV boluses"
PROFILE_INFUSION = "Infusion then washout"


def build_exposure(spec):
    """
    Turn a hashable exposure `spec` tuple into (c_ext_fn, breakpoints) for
    simulate_profile(). spec[0] selects the model_core_v04 profile factory:
      ("bolus",    C0,   k_e)                 -> pk_bolus
      ("train",    C0,   k_e, tau_h, n)       -> pk_bolus_train
      ("infusion", C_ss, k_e, t_off_h)        -> pk_infusion
    Keeping it a tuple (not a lambda) is what lets run_sim stay @st.cache_data'd.
    """
    kind = spec[0]
    if kind == "bolus":
        fn = mc.pk_bolus(spec[1], spec[2])
    elif kind == "train":
        fn = mc.pk_bolus_train(spec[1], spec[2], spec[3], spec[4])
    elif kind == "infusion":
        fn = mc.pk_infusion(spec[1], spec[2], spec[3])
    else:
        raise ValueError(f"unknown exposure kind: {kind!r}")
    return fn, fn.breakpoints


def add_cext_overlay(fig, t_grid, cext):
    """Draw the driving extracellular exposure C_ext(t) on a right-hand axis."""
    fig.add_trace(go.Scatter(x=t_grid, y=cext, mode="lines", name="C_ext(t)",
                             line=dict(color="crimson", width=1.5, dash="dot"),
                             yaxis="y2"))
    fig.update_layout(yaxis2=dict(title="C_ext (uM)", overlaying="y", side="right",
                                  showgrid=False, rangemode="tozero"))

# ---------------------------------------------------------------------------
# Default state + reset handling
# ---------------------------------------------------------------------------
DEFAULTS = {
    "cell_line": "RPTEC/TERT1",
    "nrk_kdeg_free": False,
    "use_mm": True,
    "K_m": mc.KM_DEFAULT,
    "dose": 34.0,
    "exposure_mode": EXPOSURE_CONST,
    "profile_type": PROFILE_BOLUS,
    "pk_thalf": 8.0,
    "dose_interval": 12.0,
    "n_doses": 4,
    "infusion_off": 24.0,
    "k_mat": mc.FIXED_DEFAULT["k_mat"],
    "k_rec": mc.FIXED_DEFAULT["k_rec"],
    "k_fuse": mc.FIXED_DEFAULT["k_fuse"],
    "k_esc": mc.FIXED_DEFAULT["k_esc"],
    "t_washout": 24.0,
    "sim_end": 50.0,
    "thresholds": [500, 1000],
}

for _key, _val in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def reset_to_defaults():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


# Representative polymyxin B plasma exposure (see the About tab). Total plasma Cmax
# is ~5-8 mg/L; at MW ~1200 g/mol that is ~4-7 uM total. Polymyxin B is ~50-60%
# protein-bound, and it is the UNBOUND drug that drives the filtered load presented
# to the proximal tubule, so the preset uses ~3 uM (unbound Cmax). Terminal half-life
# is ~9-12 h; 10 h is the default. These set a time-varying IV bolus, in contrast to
# the 34 uM constant in vitro calibration dose.
INVIVO_PRESET = {
    "exposure_mode": EXPOSURE_PROFILE,
    "profile_type": PROFILE_BOLUS,
    "dose": 3.0,
    "pk_thalf": 10.0,
}


def load_invivo_preset():
    for key, val in INVIVO_PRESET.items():
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Model controls")
st.sidebar.button("Reset to defaults", on_click=reset_to_defaults, use_container_width=True)
st.sidebar.markdown("---")

cell_line = st.sidebar.selectbox("Cell line", CELL_LINES, key="cell_line")

if cell_line == "NRK-52E":
    nrk_kdeg_free = st.sidebar.checkbox(
        "NRK: use k_deg-free fit (Step 2B)", key="nrk_kdeg_free",
        help="Default (unchecked) = Step 2A: NRK shares the RPTEC-fitted k_deg, only "
             "k_uptake is refitted. Checked = Step 2B: both k_uptake and k_deg are "
             "refitted for NRK, giving a ~6.7x faster lysosomal degradation "
             "(t1/2 ~7h vs ~47h) that captures the steep washout decline.",
    )
else:
    nrk_kdeg_free = False

use_mm = st.sidebar.checkbox(
    "Saturating (Michaelis-Menten) uptake", key="use_mm",
    help="Uncheck to use the v0.2 linear uptake approximation "
         "(valid only when dose << K_m)."
)
if use_mm:
    K_m = st.sidebar.slider("K_m (uM)", min_value=20.0, max_value=1000.0, step=10.0,
                             key="K_m",
                             help="Extracellular concentration at which uptake reaches half V_max.")
else:
    K_m = 1e9  # effectively linear

st.sidebar.markdown("**Exposure protocol**")
exposure_mode = st.sidebar.radio(
    "Extracellular exposure C_ext(t)",
    [EXPOSURE_CONST, EXPOSURE_PROFILE],
    key="exposure_mode",
    help="Constant = the in vitro protocol: a fixed medium concentration for the "
         "exposure window, then washout to zero (what Jarzina et al. did, and what "
         "k_uptake/k_deg were calibrated on). Time-varying = drive the SAME calibrated "
         "cell with an analytic pharmacokinetic profile whose concentration rises and "
         "falls, as drug would in vivo. No refit: k_uptake and k_deg are properties of "
         "the cell and drug, not of the exposure shape.",
)
is_profile = exposure_mode == EXPOSURE_PROFILE

st.sidebar.button(
    "Load in vivo plasma preset (polymyxin B)",
    on_click=load_invivo_preset, use_container_width=True,
    help="Approximate polymyxin B plasma exposure: a time-varying IV bolus peaking at "
         "~3 uM (unbound Cmax; total plasma ~2x higher) with a terminal half-life of "
         "~10 h. Unbound drug drives the filtered load reaching the tubule. Contrast "
         "with the 34 uM constant in vitro calibration dose.",
)

dose = st.sidebar.slider(
    "Peak / plateau C_ext (uM)" if is_profile else "Extracellular dose, C_ext (uM)",
    min_value=0.5, max_value=2000.0, step=0.5, key="dose",
    help="Constant mode: the fixed medium concentration. Time-varying mode: the "
         "amplitude of the profile — the bolus peak, or the infusion plateau target. "
         "The 34 uM default is the Jarzina calibration dose; realistic in vivo "
         "polymyxin B plasma peaks are far lower (~2-7 uM total, ~1-3 uM unbound).",
)

if is_profile:
    profile_type = st.sidebar.selectbox(
        "PK profile shape", [PROFILE_BOLUS, PROFILE_TRAIN, PROFILE_INFUSION],
        key="profile_type",
        help="Bolus: instantaneous spike, then first-order decay. Repeated boluses: the "
             "same spike redosed at a fixed interval (superposed). Infusion: zero-order "
             "rise towards a plateau, then washout after the infusion stops.",
    )
    pk_thalf = st.sidebar.slider(
        "Exposure half-life, t½ (h)", 0.5, 48.0, step=0.5, key="pk_thalf",
        help="Elimination half-life of the extracellular profile; k_e = ln2 / t½. For a "
             "fixed peak, a shorter t½ means a briefer exposure (smaller AUC).",
    )
    if profile_type == PROFILE_TRAIN:
        dose_interval = st.sidebar.slider("Dosing interval (h)", 2.0, 48.0, step=1.0,
                                          key="dose_interval")
        n_doses = st.sidebar.slider("Number of doses", 1, 10, step=1, key="n_doses")
    elif profile_type == PROFILE_INFUSION:
        infusion_off = st.sidebar.slider("Infusion stop time (h)", 1.0, 48.0, step=1.0,
                                         key="infusion_off")
else:
    profile_type = None

st.sidebar.markdown("---")
st.sidebar.markdown("**Trafficking rate constants (/min)**")
st.sidebar.caption(
    "v0.4 defaults: k_mat, k_fuse from Shipman et al. (2022) OK-cell megalin model; "
    "k_esc from Gilleron et al. (2013). Vary them to stress-test their influence."
)
k_mat = st.sidebar.slider("k_mat (EE -> LE maturation)", 0.004, 0.4,
                           step=0.001, format="%.3f", key="k_mat",
                           help="v0.4 default 0.048 (Shipman k_m,1).")
k_rec = st.sidebar.slider("k_rec (EE -> surface recycling)", 0.002, 0.2,
                           step=0.0005, format="%.4f", key="k_rec",
                           help="v0.4 nominal 0.02; plausible band 0.02-0.046 "
                                "(upper = Shipman k_DAT,f). Uncertain/swept, not hard-fixed.")
k_fuse = st.sidebar.slider("k_fuse (LE -> lysosome fusion)", 0.001, 0.1,
                            step=0.0002, format="%.4f", key="k_fuse",
                            help="v0.4 default 0.0094 (Shipman k_m,2), ~2x slower than v0.2. "
                                 "Transparent to the total-intracellular fit but sets the "
                                 "rate of lysosomal filling.")
k_esc = st.sidebar.slider("k_esc (LE -> cytosol escape)", 0.00002, 0.002,
                           step=0.00002, format="%.5f", key="k_esc",
                           help="v0.4 default 0.0002 (Gilleron). Negligible for accumulation; "
                                "central to the planned ON endosomal-escape/efficacy extension.")

st.sidebar.markdown("---")
if not is_profile:
    t_washout = st.sidebar.slider("Washout time (h)", 6.0, 48.0, step=1.0, key="t_washout")
    sim_min = t_washout + 2
else:
    # Washout is a constant-mode concept; keep the last value for the sweep tab.
    t_washout = st.session_state.get("t_washout", DEFAULTS["t_washout"])
    sim_min = 6.0
sim_end = st.sidebar.slider("Simulation end (h)", sim_min, 96.0, step=2.0, key="sim_end")

fixed = dict(k_mat=k_mat, k_rec=k_rec, k_fuse=k_fuse, k_esc=k_esc)
if cell_line == "NRK-52E" and nrk_kdeg_free:
    k_uptake = mc.FITTED["k_uptake_NRK_free"]
    k_deg = mc.FITTED["k_deg_NRK_free"]
    fit_label = "Step 2B (NRK, k_deg free)"
else:
    k_uptake = mc.FITTED[KUPTAKE_KEY[cell_line]]
    k_deg = mc.FITTED["k_deg_PB"]
    fit_label = "Step 2A (NRK, k_deg fixed)" if cell_line == "NRK-52E" else "Step 1 (RPTEC)"
V_cell = mc.V_CELL[cell_line]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fitted (frozen, v0.4): k_uptake_{cell_line.split('/')[0]} = {k_uptake:.3g} "
    f"fmol/cell/min/uM, k_deg = {k_deg:.3g} /min  ·  {fit_label}"
)

# ---------------------------------------------------------------------------
# Resolve the sidebar controls into a hashable exposure spec + display strings.
# ---------------------------------------------------------------------------
if is_profile:
    k_e = np.log(2.0) / pk_thalf              # /h elimination rate from half-life
    if profile_type == PROFILE_BOLUS:
        exposure_spec = ("bolus", float(dose), k_e)
        exposure_title = f"IV bolus, peak {dose:.0f} uM, t½ {pk_thalf:.0f} h"
    elif profile_type == PROFILE_TRAIN:
        exposure_spec = ("train", float(dose), k_e, float(dose_interval), int(n_doses))
        exposure_title = (f"{int(n_doses)}× {dose:.0f} uM bolus q{dose_interval:.0f}h, "
                          f"t½ {pk_thalf:.0f} h")
    else:  # PROFILE_INFUSION
        exposure_spec = ("infusion", float(dose), k_e, float(infusion_off))
        exposure_title = (f"infusion → {dose:.0f} uM, stop {infusion_off:.0f} h, "
                          f"t½ {pk_thalf:.0f} h")
else:
    exposure_spec = ("const", float(dose), float(t_washout))
    exposure_title = f"{dose:.0f} uM constant, washout {t_washout:.0f} h"

# ---------------------------------------------------------------------------
# Cached simulation wrapper
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_sim(k_uptake, K_m, k_deg, fixed_tuple, V_cell, exposure_spec, sim_end, n=400):
    fixed_dict = dict(fixed_tuple)
    p = mc.make_params(k_uptake, K_m, k_deg, fixed=fixed_dict)
    t_grid = np.linspace(0.0, sim_end, n)
    if exposure_spec[0] == "const":
        _, dose_c, t_wash = exposure_spec
        t, tot, comps = mc.simulate(p, V_cell, dose_c, t_grid,
                                     t_washout_h=t_wash, sim_end_h=sim_end)
        cext = np.where(t_grid < t_wash, dose_c, 0.0)
    else:
        fn, bpts = build_exposure(exposure_spec)
        t, tot, comps = mc.simulate_profile(p, V_cell, fn, t_grid,
                                            sim_end_h=sim_end, breakpoints=bpts)
        cext = np.asarray(fn(t_grid), dtype=float)
    return t, tot, comps, cext


fixed_tuple = tuple(sorted(fixed.items()))
t_grid, total_nM, comps, cext_trace = run_sim(k_uptake, K_m, k_deg, fixed_tuple,
                                              V_cell, exposure_spec, sim_end)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Endo-lysosomal drug accumulation model")
st.caption(
    "A mechanistic model of how polymyxin B builds up inside kidney proximal tubule "
    "cells and gets trapped in lysosomes -- calibrated against real in vitro data. "
    "Start with the **About** tab if you're new here."
)

tab_about, tab1, tab2, tab3 = st.tabs(
    ["About this model", "Time course", "Lysosomal load", "Dose-response sweep"]
)

# ---------------------------------------------------------------------------
# Tab 0: About -- experiment, cell lines, equations, motivation
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
## What is this?

This tool simulates how **polymyxin B** — a nephrotoxic antibiotic used as a
last-resort treatment for drug-resistant infections — enters a kidney cell,
travels through the endosomal system, and accumulates inside **lysosomes**.
The model's rate constants are fitted to real measurements of drug uptake in
cultured kidney cells, so it can be used to explore "what if" questions that
weren't directly measured: different doses, different trafficking speeds,
different assumptions about how uptake saturates at high concentration.

It's built as the intracellular module of a prospective larger multi-scale model of 
kidney injury caused by drugs accumulation via megalin-mediated endocytosis.
""")

    st.markdown("### The in vitro experiment being modelled")
    st.markdown("""
The calibration data comes from **Jarzina et al. (2022), *Frontiers in
Toxicology* 4:864441**. In that study:

1. Kidney epithelial cells growing in culture were exposed to **34 µM
   polymyxin B** in the medium for **24 hours** (the *exposure phase*).
2. At the 24-hour mark, the drug-containing medium was replaced with fresh,
   drug-free medium for another 24 hours (the *washout phase*) — a 48-hour
   experiment in total.
3. At several time points, cells were lysed and the intracellular drug
   concentration was measured by LC-MS/MS.

Those measured concentrations (shown as black points with error bars on the
**Time course** tab, at the 34 µM calibration dose) are what the model's
uptake rate (`k_uptake`) and lysosomal degradation rate (`k_deg`) were fitted
to reproduce.
""")

    st.markdown("### The cell lines")
    st.markdown("""
**RPTEC/TERT1**
A telomerase-immortalised **human** renal proximal tubule epithelial cell
line. This is the primary human in vitro model of the kidney cell type most
vulnerable to polymyxin/aminoglycoside toxicity in patients, and is the main
dataset the model is calibrated to.

**NRK-52E**
A **rat** kidney epithelial cell line, included as a cross-species
comparator. It takes up markedly less drug than RPTEC/TERT1 — in the model
this shows up as an uptake rate constant roughly **8x lower**. Comparing the
two cell lines tests whether that difference can be captured just by
changing one parameter (uptake rate), with all trafficking kinetics held
the same.
""")

    st.markdown("### The model structure")
    st.markdown("""
Drug moves through three intracellular compartments before being degraded:

**extracellular medium → early endosome → late endosome → lysosome (degraded)**

with two side branches: a fraction of early-endosome content is **recycled**
back to the cell surface, and a small fraction of late-endosome content
**escapes** to the cytosol rather than continuing to the lysosome.
""")
    st.latex(r"""
\frac{dC_{ee}}{dt} = \text{Uptake}(C_{ext}) - (k_{mat} + k_{rec})\, C_{ee}
""")
    st.latex(r"""
\frac{dC_{le}}{dt} = k_{mat}\, C_{ee} - (k_{fuse} + k_{esc})\, C_{le}
""")
    st.latex(r"""
\frac{dC_{ly}}{dt} = k_{fuse}\, C_{le} - k_{deg}\, C_{ly}
""")
    st.markdown("Uptake is modelled either as saturating (Michaelis-Menten) or, "
                "when saturation is switched off, as a linear approximation valid "
                "only when dose ≪ K_m:")
    st.latex(r"""
\text{Uptake}(C_{ext}) = \frac{V_{max}\, C_{ext}}{K_m + C_{ext}}
\quad\xrightarrow[C_{ext}\ll K_m]{}\quad k_{uptake}\, C_{ext}
""")
    st.markdown("""
The two forms are tied together through the calibration. `k_uptake` is fitted in
the linear regime against the single 34 µM dataset, so `V_max` is back-calculated
to reproduce that fitted uptake **at the 34 µM calibration dose**:
`V_max = k_uptake·(K_m + 34)`. This anchors the model to the data at 34 µM for
*any* K_m — the saturating and linear curves coincide exactly at the calibration
dose, and K_m only reshapes the response at higher concentrations. (Anchoring
instead with `V_max = k_uptake·K_m` pins the C_ext→0 tangent, which leaves the
saturating curve sitting ~15% below the data at K_m=200 µM, because 34 µM is not
much smaller than K_m.)
""")
    st.markdown("""
| Symbol | Meaning | Units | Default (v0.4) | Source |
|---|---|---|---|---|
| `C_ee`, `C_le`, `C_ly` | drug amount in early / late endosome / lysosome | fmol/cell | state variables (start at 0) | — |
| `C_ext` | extracellular concentration (the exposure) | µM | 34 (calibration dose); set in sidebar | Jarzina et al. 2022 |
| `K_m` | half-saturation concentration for uptake | µM | 200 | chosen mid-range; swept 20–1000 |
| `k_uptake` (→ `V_max`) | linear uptake rate (`V_max = k_uptake·(K_m+34)`) | fmol/cell/min/µM | RPTEC 2.6×10⁻⁷; NRK 3.4×10⁻⁸ | **fitted** to Jarzina et al. 2022 |
| `k_mat` | early → late endosome maturation | /min | 0.048 (t½ ≈ 14 min) | Shipman et al. 2022 (k_m,1) |
| `k_rec` | early-endosome recycling to surface | /min | 0.02, band 0.02–0.046 (t½ ≈ 35–15 min) | Shipman et al. 2022 (k_DAT,f = upper bound) |
| `k_fuse` | late endosome → lysosome fusion | /min | 0.0094 (t½ ≈ 74 min) | Shipman et al. 2022 (k_m,2) |
| `k_esc` | late-endosome escape to cytosol | /min | 0.0002 (t½ ≈ 58 h) | Gilleron et al. 2013 (~1% escape) |
| `k_deg` | lysosomal degradation / clearance | /min | 2.4×10⁻⁴ (t½ ≈ 47 h); NRK-free 1.6×10⁻³ (t½ ≈ 7 h) | **fitted** to Jarzina et al. 2022 |
| `V_cell` | cell volume (converts fmol/cell → nM) | L | RPTEC 1.99×10⁻¹²; NRK 1.21×10⁻¹² | cell diameter (≈15.6 / 13.2 µm) |

Model amounts (fmol/cell) are converted to the intracellular concentrations
(nM) shown in the charts using each cell line's measured cell volume. As of
**v0.4**, the maturation and fusion rates (`k_mat`, `k_fuse`) are taken from
Shipman et al. (2022), a proximal-tubule-specific megalin trafficking model
in opossum kidney cells, rather than the earlier generic estimates; `k_rec`
is treated as an uncertain 0.02–0.046 /min band (nominal 0.02); `k_esc`
stays at the Gilleron et al. (2013) value. You can vary all four in the
sidebar to stress-test how much they matter — note that `k_fuse` barely
affects the *total* fitted curve but strongly shifts the *lysosomal* load
timing. `k_uptake` and `k_deg` are fitted per cell line to the Jarzina data
and held fixed here (not adjustable), so the calibration to real data is
always preserved.

For **NRK-52E** only, a sidebar checkbox switches between two fitted
parameter sets: the default *Step 2A* fit (NRK shares the RPTEC-derived
`k_deg`, only its uptake is refitted) and the *Step 2B* fit (both `k_uptake`
and `k_deg` refitted for NRK). Step 2B gives a ~6.7× faster lysosomal
degradation and reproduces the steep post-washout decline the shared-`k_deg`
fit misses.
""")

    st.markdown("### Where the default values come from")
    st.markdown("""
The defaults fall into three groups by how firmly they are pinned:

**Directly measured (trafficking machinery).** `k_mat`, `k_fuse` and the upper
bound of `k_rec` are the proximal-tubule-specific rates measured by Shipman et
al. (2022) in opossum-kidney cells, reported as percent-per-minute and converted
to /min (×0.01): k_m,1 = 4.8 → `k_mat` 0.048 (t½ ≈ 14 min), k_m,2 = 0.936 →
`k_fuse` 0.0094 (t½ ≈ 74 min), k_DAT,f = 4.61 → `k_rec` upper 0.046. This is the
closest available measured trafficking kinetics for the megalin pathway. `k_rec`
is treated as a **0.02–0.046 band** (nominal 0.02) rather than a point value
because surface recycling is cargo-dependent and Shipman's figure is an upper
bound for our lumped early-endosome→surface flux.

**Borrowed by analogy (`k_esc`).** There is no measured escape rate for megalin
cargo, so `k_esc` is held at the small value implied by Gilleron et al. (2013),
where ~1% of endosomal cargo reaches the cytosol. Its t½ (~58 h) is far slower
than fusion, so it is negligible for accumulation — it is kept in the model as
the hook for a future endosomal-escape / efficacy extension, not because it
shapes the current curves.

**Fitted to data (`k_uptake`, `k_deg`).** These are *not* assumed — they are
least-squares fitted to the Jarzina et al. (2022) 34 µM time course, separately
per cell line. The fitted lysosomal degradation half-life (`k_deg` t½ ≈ 47 h) is
what reproduces the slow post-washout decline in RPTEC; the NRK *k_deg*-free
option (t½ ≈ 7 h) is what captures NRK's much steeper decline. `K_m` has no
direct measurement here, so 200 µM is a deliberate mid-range default and is swept
(20–1000 µM) to show where uptake saturation sets in.

**Exposure timescale (the PK profile defaults).** The time-varying profiles are
parameterised by an elimination half-life t½ (k_e = ln2 / t½). The 10 h default,
and the in vivo preset, reflect polymyxin B's reported terminal plasma half-life
of roughly **9–12 h** in patients; the preset's ~3 µM peak is the unbound plasma
Cmax (see the exposure section above).
""")

    st.markdown("### Constant vs time-varying exposure")
    st.markdown("""
The calibration experiment held the medium concentration **constant** for 24 h
and then washed it out. In a patient, drug concentration in the blood — and
therefore around the kidney cell — instead **rises and falls** after each dose.
The **Exposure protocol** control in the sidebar lets you swap the constant
exposure for an analytic **pharmacokinetic (PK) profile**:

- **IV bolus** — an instantaneous spike to the chosen peak, then first-order
  decay set by an elimination half-life t½.
- **Repeated IV boluses** — the same spike redosed at a fixed interval, with
  the doses superposed (accumulation between doses is captured automatically).
- **Infusion then washout** — a zero-order rise toward a plateau, then decay
  once the infusion stops.

Crucially, **no re-fitting is involved**. `k_uptake` and `k_deg` describe the
cell and the drug, not the shape of the exposure, so the *same* calibrated model
is simply driven by a different `C_ext(t)`. When a time-varying profile is
active, its `C_ext(t)` is drawn as a **red dotted line on a right-hand axis** on
the Time course and Lysosomal load tabs, so you can see the exposure that is
driving the intracellular response.

Why this matters mechanistically: the lysosome **integrates** exposure. Two
profiles with the *same* area under the concentration curve (AUC) but different
shapes — a tall brief spike vs. a low sustained level — reach a similar eventual
lysosomal load, but the **peak** lysosomal concentration, and *when* it is
reached, depend on the exposure shape, not on AUC alone. That distinction is
exactly what a single constant-exposure experiment cannot reveal, and it is the
first step toward driving this intracellular model with a realistic (e.g.
PBPK-derived) exposure in the wider multi-scale model.

**In vivo context.** The 34 µM calibration dose is an *in vitro* concentration,
far above what circulates in a patient. Reported polymyxin B plasma exposures
are roughly a steady-state average of 2–4 mg/L and a peak (Cmax) of ~5–8 mg/L;
at a molecular weight of ~1200 g/mol that is only about **2–7 µM total**, and
roughly **half that unbound** (polymyxin B is ~50–60% protein-bound). Because it
is the unbound drug that drives the filtered load presented to the proximal
tubule, the sidebar's **"Load in vivo plasma preset"** button sets a
representative time-varying exposure on the *unbound* scale — a ~3 µM IV bolus
with a ~10 h terminal half-life — so you can see how much lysosomal load a
realistic systemic exposure would drive, versus the deliberately high calibration
dose. Note this is the concentration *presented to* the cell (plasma-derived
filtered load); the marked renal-cortical accumulation of polymyxin B is what the
model then *predicts*, not an input.
""")

    st.markdown("### Why it matters")
    st.markdown("""
Polymyxin- and aminoglycoside-induced kidney injury is described by an
**Adverse Outcome Pathway (AOP)** in which the first proposed key event
(**KE1**) is *lysosomal dysfunction* triggered by excess drug accumulating
inside lysosomes. This model turns an external dose into a mechanistic
prediction of **how much drug piles up in the lysosome and how fast** —
giving a testable, biologically-grounded link between drug exposure and
early kidney injury risk, rather than relying only on an empirical
dose-toxicity correlation. The **Dose-response sweep** tab is the most
directly translational output: it shows how long it takes lysosomal drug
load to cross a candidate "harmful" threshold, across a range of doses.
""")

    st.markdown("### References")
    st.markdown("""
- **Jarzina et al. (2022).** *Frontiers in Toxicology* 4:864441. — Calibration
  dataset: intracellular polymyxin B uptake in RPTEC/TERT1 and NRK-52E (Fig 6A).
- **Shipman KE, Long KR, Cowan IA, Rbaibi Y, Baty CJ, Weisz OA (2022).** *An
  Adaptable Physiological Model of Endocytic Megalin Trafficking in Opossum
  Kidney Cells and Mouse Kidney Proximal Tubule.* FUNCTION 3(6):zqac046.
  doi:10.1093/function/zqac046. — Source of `k_mat`, `k_fuse`, and the `k_rec`
  upper bound (comprehensive OK-cell model, Fig 4C).
- **Gilleron et al. (2013).** *Nature Biotechnology* 31:638–646. — Basis for the
  ~1% endosomal-escape scale used to set `k_esc`.

Polymyxin B plasma pharmacokinetics (Cmax, terminal half-life, protein binding)
used for the in vivo preset are drawn from the clinical population-PK literature
(e.g. Sandri et al. 2013, *Clinical Infectious Diseases*); confirm against the
primary source before citing specific numbers.
""")

# ---------------------------------------------------------------------------
# Tab 1: time course + compartment breakdown
# ---------------------------------------------------------------------------
with tab1:
    st.markdown(
        "The model's predicted intracellular drug concentration over time, split "
        "into the three compartments it moves through. At the 34 µM calibration "
        "dose, the actual measured data points are overlaid for comparison.\n\n"
        "**Why this matters:** this is the direct validation check — it's the same "
        "quantity Jarzina et al. measured by LC-MS/MS, so at 34 µM you can see for "
        "yourself whether the fitted uptake and degradation rates genuinely reproduce "
        "the real experiment, not just a single fitted number. The compartment "
        "breakdown on the right shows the same total split by location: watch how the "
        "balance should shift from endosomes towards lysosome-dominance as exposure "
        "continues, then decline everywhere once the drug is washed out."
    )
    st.caption(f"**Exposure:** {exposure_title}."
               + ("  The driving C_ext(t) is the red dotted line (right-hand axis)."
                  if is_profile else
                  "  The dashed grey line marks washout."))
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_grid, y=total_nM, mode="lines",
                                  name="model (total)", line=dict(width=3)))
        show_data = (not is_profile) and dose == mc.CAL_CONC
        if show_data:
            ydata = DATA_CENTRAL[cell_line]
            if cell_line == "RPTEC/TERT1":
                err_lo = mc.RPTEC_CENTRAL - mc.RPTEC_LOWER
                err_hi = mc.RPTEC_UPPER - mc.RPTEC_CENTRAL
                fig.add_trace(go.Scatter(
                    x=mc.T_ALL_H, y=ydata, mode="markers", name="data (Jarzina 2022)",
                    error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo),
                    marker=dict(size=9, color="black")))
            else:
                fig.add_trace(go.Scatter(x=mc.T_ALL_H, y=ydata, mode="markers",
                                          name="data (Jarzina 2022)",
                                          marker=dict(size=9, color="black")))
        if is_profile:
            add_cext_overlay(fig, t_grid, cext_trace)
        else:
            fig.add_vline(x=t_washout, line_dash="dash", line_color="grey",
                          annotation_text="washout")
        fig.update_layout(title=f"{cell_line} — total intracellular",
                           xaxis_title="time (h)", yaxis_title="intracellular conc (nM)",
                           height=450)
        st.plotly_chart(fig, use_container_width=True)
        if (not is_profile) and dose != mc.CAL_CONC:
            st.caption("Calibration data is only shown at the 34 uM constant calibration dose.")

    with col2:
        fig2 = go.Figure()
        for name, color in [("C_ee", "#9ecae1"), ("C_le", "#fdae6b"), ("C_ly", "#a1d99b")]:
            fig2.add_trace(go.Scatter(x=t_grid, y=comps[name], mode="lines",
                                       stackgroup="one", name=name,
                                       line=dict(color=color)))
        if is_profile:
            add_cext_overlay(fig2, t_grid, cext_trace)
        else:
            fig2.add_vline(x=t_washout, line_dash="dash", line_color="grey")
        fig2.update_layout(title=f"{cell_line} — compartment breakdown",
                            xaxis_title="time (h)", yaxis_title="intracellular conc (nM)",
                            height=450)
        st.plotly_chart(fig2, use_container_width=True)

    frac = {k: comps[k][-1] / max(total_nM[-1], 1e-12) for k in comps}
    st.markdown(
        f"**Localisation at t = {sim_end:.0f}h:** "
        f"early endosome {frac['C_ee']*100:.1f}% · "
        f"late endosome {frac['C_le']*100:.1f}% · "
        f"lysosome {frac['C_ly']*100:.1f}%"
    )

# ---------------------------------------------------------------------------
# Tab 2: lysosomal load + thresholds
# ---------------------------------------------------------------------------
with tab2:
    st.markdown(
        "Lysosomal drug concentration alone (`C_ly`), rather than the total shown in "
        "the Time course tab.\n\n"
        "**Why this matters:** the total intracellular concentration mixes drug that's "
        "still in transit (endosomes) with drug that has actually reached the lysosome. "
        "The AOP identifies lysosomal accumulation specifically, not total cellular "
        "drug, as the trigger for the first key event — so isolating `C_ly`, and asking "
        "when it crosses a candidate \"harmful load\" threshold, is the more "
        "mechanistically relevant readout. There's no established numeric threshold for "
        "polymyxin B yet, so the values below are illustrative reference points to test "
        "sensitivity, not validated safety limits."
    )
    thresholds = st.multiselect("Threshold levels (nM)", [250, 500, 1000, 2000, 3000],
                                 key="thresholds")
    st.caption(f"**Exposure:** {exposure_title}."
               + ("  The driving C_ext(t) is the red dotted line (right-hand axis)."
                  if is_profile else
                  "  The dashed grey line marks washout."))
    cly = comps["C_ly"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t_grid, y=cly, mode="lines", name="C_ly",
                               line=dict(color="#238b45", width=3)))
    if is_profile:
        add_cext_overlay(fig3, t_grid, cext_trace)
    else:
        fig3.add_vline(x=t_washout, line_dash="dash", line_color="grey",
                       annotation_text="washout")

    rows = []
    for thr in thresholds:
        fig3.add_hline(y=thr, line_dash="dot", line_color="grey")
        tc = mc.first_crossing_h(t_grid, cly, thr)
        rows.append({"threshold (nM)": thr,
                      "time to cross (h)": f"{tc:.2f}" if tc is not None else "not reached"})
        if tc is not None:
            fig3.add_trace(go.Scatter(x=[tc], y=[thr], mode="markers+text",
                                       marker=dict(color="red", size=10, symbol="triangle-down"),
                                       text=[f"{tc:.1f}h"], textposition="top right",
                                       showlegend=False))
    fig3.update_layout(title=f"{cell_line} — lysosomal load",
                        xaxis_title="time (h)", yaxis_title="lysosomal conc, C_ly (nM)",
                        height=500)
    st.plotly_chart(fig3, use_container_width=True)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=False)

# ---------------------------------------------------------------------------
# Tab 3: dose-response sweep (peak C_ly / crossing time vs dose)
# ---------------------------------------------------------------------------
with tab3:
    st.markdown(
        "Sweeps dose (at the current cell line, K_m, and trafficking settings) across "
        "a wider range than was ever tested in vitro, and shows how peak lysosomal "
        "load and time-to-threshold scale with extracellular concentration.\n\n"
        "**Why this matters:** real toxicology studies can only test a handful of "
        "doses, but this sweep extrapolates the same mechanistic rate constants "
        "(fitted to a single 34 µM experiment) across the full dose range — producing "
        "the two most directly translational outputs of the whole model. The top chart "
        "shows how peak lysosomal burden scales with dose, and where saturating uptake "
        "(governed by `K_m`) makes it plateau below the naive linear prediction. The "
        "bottom chart shows how quickly a given threshold would be reached at each "
        "dose — the kind of dose/exposure-vs-risk relationship a safety assessment "
        "would actually want."
    )
    if is_profile:
        st.info(
            "This sweep is a **constant-exposure** analysis — it holds each dose fixed "
            "for the exposure window, so it is independent of the time-varying profile "
            "selected in the sidebar. The Time course and Lysosomal load tabs reflect "
            "that profile; this tab always sweeps constant doses."
        )
    doses = [34.0, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0]

    @st.cache_data(show_spinner=False)
    def sweep(k_uptake, K_m, k_deg, fixed_tuple, V_cell, doses, t_washout, sim_end):
        fixed_dict = dict(fixed_tuple)
        p = mc.make_params(k_uptake, K_m, k_deg, fixed=fixed_dict)
        t_grid = np.linspace(0.0, sim_end, 600)
        out = []
        for d in doses:
            _, tot, comps = mc.simulate(p, V_cell, d, t_grid,
                                         t_washout_h=t_washout, sim_end_h=sim_end)
            out.append(dict(
                dose=d, peak_total=tot.max(), peak_cly=comps["C_ly"].max(),
                t500=mc.first_crossing_h(t_grid, comps["C_ly"], 500.0),
                t1000=mc.first_crossing_h(t_grid, comps["C_ly"], 1000.0),
            ))
        return out

    sweep_rows = sweep(k_uptake, K_m, k_deg, fixed_tuple, V_cell, tuple(doses),
                        t_washout, sim_end)
    df = pd.DataFrame(sweep_rows)

    fig4 = make_subplots(specs=[[{"secondary_y": False}]])
    fig4.add_trace(go.Scatter(x=df["dose"], y=df["peak_cly"], mode="lines+markers",
                               name="peak C_ly"))
    lin_ref = df["peak_cly"].iloc[0] * df["dose"] / df["dose"].iloc[0]
    fig4.add_trace(go.Scatter(x=df["dose"], y=lin_ref, mode="lines", name="linear reference",
                               line=dict(dash="dash", color="grey")))
    fig4.update_xaxes(type="log", title="C_ext (uM)")
    fig4.update_yaxes(type="log", title="peak lysosomal C_ly (nM)")
    fig4.update_layout(title=f"{cell_line} — peak lysosomal load vs dose (K_m={K_m:.0f} uM)",
                        height=450)
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    for col, label in [("t500", "time to 500 nM"), ("t1000", "time to 1000 nM")]:
        sub = df.dropna(subset=[col])
        fig5.add_trace(go.Scatter(x=sub["dose"], y=sub[col], mode="lines+markers", name=label))
    fig5.update_xaxes(type="log", title="C_ext (uM)")
    fig5.update_yaxes(title="time to cross threshold (h)")
    fig5.update_layout(title="Threshold-crossing time vs dose", height=450)
    st.plotly_chart(fig5, use_container_width=True)

    st.dataframe(df.round(2), hide_index=True, use_container_width=True)
