"""
Streamlit explorer for the endo-lysosomal drug accumulation model.

Lets you interactively vary dose, K_m (Michaelis-Menten uptake saturation),
and the fixed trafficking rate constants, and see the effect on:
  - the total/compartmental time course (vs. the Jarzina et al. 2022 calibration data)
  - the lysosomal load trajectory and threshold-crossing times
  - a dose-response sweep of peak lysosomal load / threshold-crossing time

Uses frozen v0.2 fitted parameters (k_uptake, k_deg) as defaults -- no live
refitting. Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import model_core as mc

st.set_page_config(page_title="Endo-lysosomal accumulation model", layout="wide")

CELL_LINES = list(mc.V_CELL.keys())
KUPTAKE_KEY = {"RPTEC/TERT1": "k_uptake_RPTEC", "NRK-52E": "k_uptake_NRK"}
DATA_CENTRAL = {"RPTEC/TERT1": mc.RPTEC_CENTRAL, "NRK-52E": mc.NRK_CENTRAL}

# ---------------------------------------------------------------------------
# Default state + reset handling
# ---------------------------------------------------------------------------
DEFAULTS = {
    "cell_line": "RPTEC/TERT1",
    "use_mm": True,
    "K_m": mc.KM_DEFAULT,
    "dose": 34.0,
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


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Model controls")
st.sidebar.button("Reset to defaults", on_click=reset_to_defaults, use_container_width=True)
st.sidebar.markdown("---")

cell_line = st.sidebar.selectbox("Cell line", CELL_LINES, key="cell_line")

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

dose = st.sidebar.slider(
    "Extracellular dose, C_ext (uM)",
    min_value=34.0, max_value=2000.0, step=1.0, key="dose",
    help="Continuous extracellular drug concentration.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Trafficking rate constants (/min)**")
k_mat = st.sidebar.slider("k_mat (EE -> LE maturation)", 0.004, 0.4,
                           step=0.001, format="%.3f", key="k_mat")
k_rec = st.sidebar.slider("k_rec (EE -> surface recycling)", 0.002, 0.2,
                           step=0.001, format="%.3f", key="k_rec")
k_fuse = st.sidebar.slider("k_fuse (LE -> lysosome fusion)", 0.002, 0.2,
                            step=0.001, format="%.3f", key="k_fuse")
k_esc = st.sidebar.slider("k_esc (LE -> cytosol escape)", 0.00002, 0.002,
                           step=0.00002, format="%.5f", key="k_esc")

st.sidebar.markdown("---")
t_washout = st.sidebar.slider("Washout time (h)", 6.0, 48.0, step=1.0, key="t_washout")
sim_end = st.sidebar.slider("Simulation end (h)", t_washout + 2, 96.0, step=2.0, key="sim_end")

fixed = dict(k_mat=k_mat, k_rec=k_rec, k_fuse=k_fuse, k_esc=k_esc)
k_uptake = mc.FITTED[KUPTAKE_KEY[cell_line]]
k_deg = mc.FITTED["k_deg_PB"]
V_cell = mc.V_CELL[cell_line]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fitted (frozen, v0.2): k_uptake_{cell_line.split('/')[0]} = {k_uptake:.3g} "
    f"fmol/cell/min/uM, k_deg = {k_deg:.3g} /min"
)

# ---------------------------------------------------------------------------
# Cached simulation wrapper
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_sim(k_uptake, K_m, k_deg, fixed_tuple, V_cell, dose, t_washout, sim_end, n=400):
    fixed_dict = dict(fixed_tuple)
    p = mc.make_params(k_uptake, K_m, k_deg, fixed=fixed_dict)
    t_grid = np.linspace(0.0, sim_end, n)
    t, tot, comps = mc.simulate(p, V_cell, dose, t_grid,
                                 t_washout_h=t_washout, sim_end_h=sim_end)
    return t, tot, comps


fixed_tuple = tuple(sorted(fixed.items()))
t_grid, total_nM, comps = run_sim(k_uptake, K_m, k_deg, fixed_tuple, V_cell,
                                   dose, t_washout, sim_end)

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
| Symbol | Meaning | Units |
|---|---|---|
| `C_ee`, `C_le`, `C_ly` | drug amount in early endosome / late endosome / lysosome | fmol per cell |
| `C_ext` | extracellular drug concentration (the dose) | µM |
| `k_uptake` / `V_max`, `K_m` | rate (or saturating rate) at which drug is taken up from the medium | fmol/cell/min (per µM) |
| `k_mat` | rate early endosomes mature into late endosomes | /min |
| `k_rec` | rate early-endosome content is recycled back to the surface | /min |
| `k_fuse` | rate late endosomes fuse with lysosomes | /min |
| `k_esc` | rate late-endosome content escapes to the cytosol | /min |
| `k_deg` | rate drug is degraded/cleared once in the lysosome | /min |

Model amounts (fmol/cell) are converted to the intracellular concentrations
(nM) shown in the charts using each cell line's measured cell volume. The
four trafficking rates (`k_mat`, `k_rec`, `k_fuse`, `k_esc`) are literature
values for generic endosomal trafficking kinetics, not drug-specific — you
can vary them in the sidebar to stress-test how much they matter.
`k_uptake` and `k_deg` are fitted per cell line to the Jarzina data and held
fixed here (not adjustable), so the calibration to real data is always
preserved.
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
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_grid, y=total_nM, mode="lines",
                                  name="model (total)", line=dict(width=3)))
        if dose == mc.CAL_CONC:
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
        fig.add_vline(x=t_washout, line_dash="dash", line_color="grey",
                      annotation_text="washout")
        fig.update_layout(title=f"{cell_line} @ {dose:.0f} uM — total intracellular",
                           xaxis_title="time (h)", yaxis_title="intracellular conc (nM)",
                           height=450)
        st.plotly_chart(fig, use_container_width=True)
        if dose != mc.CAL_CONC:
            st.caption("Calibration data is only shown at the 34 uM calibration dose.")

    with col2:
        fig2 = go.Figure()
        for name, color in [("C_ee", "#9ecae1"), ("C_le", "#fdae6b"), ("C_ly", "#a1d99b")]:
            fig2.add_trace(go.Scatter(x=t_grid, y=comps[name], mode="lines",
                                       stackgroup="one", name=name,
                                       line=dict(color=color)))
        fig2.add_vline(x=t_washout, line_dash="dash", line_color="grey")
        fig2.update_layout(title=f"{cell_line} @ {dose:.0f} uM — compartment breakdown",
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
    cly = comps["C_ly"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t_grid, y=cly, mode="lines", name="C_ly",
                               line=dict(color="#238b45", width=3)))
    fig3.add_vline(x=t_washout, line_dash="dash", line_color="grey", annotation_text="washout")

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
    fig3.update_layout(title=f"{cell_line} @ {dose:.0f} uM — lysosomal load",
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
