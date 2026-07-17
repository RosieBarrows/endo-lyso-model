# Brief: update trafficking rate constants in the endo-lysosomal model

## Context / what we're changing and why

We're updating the fixed trafficking rate constants in the endo-lysosomal model
using directly measured, proximal-tubule-specific values from Shipman et al.
(2022), a five-compartment ODE model of megalin trafficking in OK cells whose
compartments map cleanly onto ours (their AEE = our EE, their AV/Rab7 = our
LE/MVB, their Lys = our LY; their separate Rab11a "DAT" recycling compartment
collapses into our `k_rec` efflux term). Currently our maturation and recycling
rates are taken from generic Rab-conversion timescales and a rough "~1/3
recycled" assumption; Shipman provides measured PT-cell rates for exactly these
transitions.

Specifically: set `k_mat` (EE→LE) from 0.04 to **0.048 /min** (their k_m,1,
AEE→AV maturation — this independently corroborates our current value to within
20%), and set `k_fuse` (LE→LY) from 0.02 to **0.0094 /min** (their k_m,2,
AV→lysosome maturation — roughly 2× slower than we assumed, which will lengthen
predicted lysosomal residence and matters for the accumulation threshold). Treat
`k_rec` (EE→surface) as an uncertain/swept parameter over **0.02–0.046 /min**,
with the upper bound set by their k_DAT,f (fast recycling from the early
endosome); do not hard-fix it.

Deliberately do **not** change two things: (1) `k_deg` stays drug-fitted —
Shipman's lysosomal degradation rate is a property of megalin the protein
(~9-min residence), whereas ours is the nuclease/protease resistance of the
drug, an entirely different quantity; and (2) do **not** add a recycling route
to the LE despite Shipman showing heavy AV→surface recycling, because that
recycling is a *receptor* property — cargo and receptor dissociate in the
acidifying late endosome, so free drug is committed to the lysosome. Our current
LE structure (fuse or escape only) is biologically correct for cargo and should
stay.

## Compartment mapping

| Our model (3-comp) | Shipman (5-comp) | Marker | Notes |
|---|---|---|---|
| extracellular / surface | Surface | — | State in Shipman; boundary condition in ours |
| EE (early endosome) | AEE | EEA1 | Clean 1:1 — entry/sorting compartment |
| LE / MVB | AV (apical vacuole) | Rab7 | Clean 1:1 — Rab7 is the canonical late-endosome marker |
| LY | Lys | LysoTracker | Clean 1:1 |
| *(k_rec efflux, no compartment)* | DAT | Rab11a | Shipman's recycling tubules collapse into our direct EE→surface flux |

## Parameter mapping (Shipman OK-cell → endo-lysosomal model, all /min)

| Param | Current | Shipman analogue | Suggested | Transfer confidence |
|---|---|---|---|---|
| `k_mat` (EE→LE) | 0.04 | k_m,1 = 0.048 | **0.048** | High — membrane maturation; corroborates current value |
| `k_rec` (EE→surface) | 0.02 | k_DAT,f = 0.046 | **0.02–0.046 (sweep)** | Medium — cargo-dependent; Shipman value is an upper bound |
| `k_fuse` (LE→LY) | 0.02 | k_m,2 = 0.0094 | **0.0094** | High — maturation, but ~2× slower than assumed |
| `k_esc` (LE→cytosol) | 0.0002 | *(none)* | keep 0.0002 | Do not transfer — receptor doesn't escape; keep Gilleron |
| `k_deg` (LY) | fitted (2.4e-4) | k_d = 0.108 | **keep fitted** | Do not transfer — cargo chemistry, not compartment property |
| uptake / `V_max` | fitted | k_e = 0.127 | keep fitted | k_e is receptor internalisation, not cargo flux |

## Tasks

1. **Assess the re-run impact.** These are *fixed* parameters that the *fitted*
   parameters were conditioned on, so changing them invalidates the current fit.
   Enumerate exactly what must be re-run — at minimum I expect: the Step 1 fit
   (`k_uptake_RPTEC`, `k_deg_PB`), Steps 2A/2B (`k_uptake_NRK`, cell-line ratio),
   the Section 2.6 fixed-parameter stress-test, the Section 2.7 Michaelis–Menten
   concentration sweep and threshold-crossing timing, and all downstream result
   figures/numbers in Section 3. Confirm or correct this list before running.

2. **Set it up as a new version** (v0.4, following the v0.2/v0.3 lineage),
   keeping the old version intact for comparison, and report how the refit
   parameters and key predictions shift.

3. **Update the write-up in the right places:** the Section 2.4 parameter tables
   (values *and* the "Source" column), the Section 1.2 modelling-gap framing
   (Shipman is the closest existing PT trafficking model and should be cited
   rather than only Janssen / Lauffenburger & Linderman), the Section 2.2
   model-structure text, and add the reference.
   I'd like you to keep a version of the old write up as well.

## Reference

Shipman KE, Long KR, Cowan IA, Rbaibi Y, Baty CJ, Weisz OA (2022). *An Adaptable
Physiological Model of Endocytic Megalin Trafficking in Opossum Kidney Cells and
Mouse Kidney Proximal Tubule.* FUNCTION 3(6):zqac046.
doi:10.1093/function/zqac046.

Values are from the comprehensive OK-cell model (Fig 4C):
k_m,1 = 4.8 %·min⁻¹, k_m,2 = 0.936 %·min⁻¹, k_DAT,f = 4.61 %·min⁻¹
(percent-per-minute × 0.01 = /min).