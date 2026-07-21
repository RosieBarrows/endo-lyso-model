# Calibrate a lysosomal-load → injury function (C_ly → cell death) from Jarzina (2022) key-event data

## What you are building and why

We have a **calibrated intracellular accumulation model** (call it the *accumulation model*) that predicts, for polymyxin B in proximal tubule cells, the time course of lysosomal drug load **C_ly(t)** given an extracellular exposure profile. It was calibrated on LC-MS/MS *total* intracellular concentration at a **single dose (34 µM)** in two cell lines (human RPTEC/TERT1 and rat NRK-52E).

We now want to turn that accumulation model into something that predicts **injury**, by fitting an injury function of the form

> injury rate (or death fraction) = f(C_ly)

using Jarzina et al. (2022)'s key-event dose–response data. Concretely: take each nominal concentration Jarzina exposed cells to, **run our accumulation model to get the corresponding C_ly**, and regress Jarzina's cell-death readout against **C_ly** instead of against nominal concentration. The fitted midpoint **C_ly50** is the lysosomal-load injury threshold. This function is intended to later replace the phenomenological *drug-concentration → injury* term in a Gebremichael-style (2018) population injury model — but **that coupling is out of scope here**; this task only produces and characterises f(C_ly).

**The single most important thing to get right:** the injury data lives at **7.8–2000 µM**, i.e. **1–2 orders of magnitude above the 34 µM calibration point**, deep in the saturating regime of uptake. So C_ly at these concentrations — and therefore C_ly50 — is **dominated by the Michaelis constant K_m, which is not identifiable from the single-dose calibration**. C_ly50 must therefore be reported as a **K_m-conditional band, not a single number.** Treat this as the headline result and headline uncertainty. Do not paper over it.

---

## STEP 0 — Repository reconnaissance (do this before writing any modelling code)

The accumulation model already exists in this repository. **Do not reconstruct it from equations or from any write-up.** Instead, explore the repo and report back, in a short `RECON.md`, the following before proceeding:

1. The module/function that **integrates the accumulation ODEs** and returns C_ly over time. Note its exact signature: how the extracellular exposure is passed (constant scalar? a C_ext(t) callable? a time/level breakpoint list?), and what object/array holds C_ly.
2. Where the **calibrated parameters** live and their names — in particular the per-cell-line uptake rate (expect something like `k_uptake_RPTEC` and `k_uptake_NRK`), the degradation rate `k_deg`, trafficking rates (`k_mat`, `k_fuse`, `k_rec`), the Michaelis constant `K_m`, and however `V_max` is defined.
3. Confirm the **V_max anchoring convention**. The model anchors saturating uptake to the calibration dose as `V_max = k_uptake * (K_m + 34 µM)`. Verify this is how the code computes V_max (so that changing K_m keeps the fit at 34 µM exact). If the code does it differently, stop and report — do not "fix" it.
4. The **units** of C_ly the model emits (nM? fmol/cell? concentration vs amount) and of C_ext it expects.

Conform to the existing interfaces and naming. Add a **new calibration module** (e.g. `injury_calibration/`) that *imports* the accumulation model; do not edit, rename, or refactor the existing model code.

If any of the four items above cannot be found or is ambiguous, **ask before proceeding** rather than guessing.

---

## STEP 1 — Load the key-event data

The injury data has been **digitised by hand from Jarzina Fig. 5** (do **not** attempt to auto-digitise anything — that step is deliberately kept under human control). Read it from CSVs with this schema:

```
concentration_uM, mean, sd
```

one file per (key event × cell line). Expected files (create a `data/` folder and place them there; if a file is absent, skip that arm gracefully):

- `ke3_viability_rptec.csv`  — primary target, human
- `ke3_viability_nrk.csv`    — primary target, rat
- `ke1_lamp_rptec.csv`       — corroboration, human
- `ke1_lamp_nrk.csv`         — corroboration, rat

**If the CSVs are not yet present, seed them with the values below** (already digitised; units = % of control; `sd` symmetric linear ±SD). Mark clearly in a comment that these are figure-digitised and provisional.

### `ke3_viability_rptec.csv` (human — PRIMARY)
```
concentration_uM, mean, sd
31.25,  91.0, 7.4     # upper cap buried in cluster; SD from lower cap, treat as wide
62.5,   40.0, 2.9     # slope anchor — clean, both caps
125.0,   5.8, 2.5
250.0,   0.8, 1.0     # floor
500.0,   0.0, 1.0     # floor (digitised -0.2 -> clamp to 0)
1000.0,  0.0, 1.0     # floor (digitised -0.2 -> clamp to 0)
```

### `ke3_viability_nrk.csv` (rat — PRIMARY)
```
concentration_uM, mean, sd
15.6,   95.5, 3.0     # ~baseline
62.5,   91.7, 3.0     # ~baseline
125.0,  91.5, 3.0     # ~baseline
250.0,  77.9, 2.2
500.0,  54.0, 4.7     # slope anchor
1000.0, 23.0, 7.7
```

### `ke1_lamp_rptec.csv` (human — CORROBORATION)
```
concentration_uM, mean, sd
62.5,    349.0, 199.0   # SD from UPPER cap only (lower caps unreadable/clustered)
250.0,   489.0, 303.0
500.0,   956.0, 483.0
1000.0, 1041.0, 591.0
# baseline: treat all concentrations below 62.5 uM as ~100% of control (fixed, not fitted)
```

### `ke1_lamp_nrk.csv` (rat — CORROBORATION)
```
concentration_uM, mean, sd
250.0,  249.0, NaN     # centrals only, no reliable SDs -> fit unweighted for this arm
500.0,  349.0, NaN
1000.0, 428.0, NaN
# baseline: treat below 250 uM as ~100% of control
```

**Data caveats to encode as comments / assertions:**
- Viability readouts are bounded [0, 100]; clamp any digitised negatives to 0.
- The **foot of every curve is fixed by construction** (readout is % of control → baseline = 100%). Do not try to fit the low-dose baseline; constrain it.
- **Cathepsin D (Jarzina KE2, Fig 5C) is deliberately excluded.** Jarzina themselves flag it as unreliable (it responded out of AOP order and they express "concern regarding the cathepsin assay as a reliable marker for KE2"). Do not request or fit it.
- Never fit to Jarzina's *fitted* curves — only these discrete measured points. (Fitting model-to-model manufactures false precision.)

---

## STEP 2 — Map nominal concentration → C_ly (the core step)

For each cell line, for each nominal concentration in that line's data:

1. Drive the accumulation model with a **constant extracellular exposure** equal to that nominal concentration, for **24 h** (Jarzina's readouts are all 24 h single-timepoint).
2. Extract **C_ly at t = 24 h**. This is the default C_ly summary statistic.
3. Use that cell line's **own** uptake parameterisation (`k_uptake_RPTEC` for human, `k_uptake_NRK` for rat). **Never** map the rat data through human uptake or vice versa — the whole human/rat sensitivity difference is uptake-driven, so mixing them would corrupt the central result (see Step 4).

Make the **C_ly summary a swappable function**: default `C_ly(24h)`, with `peak C_ly` and `∫C_ly dt over 0–24h` (cumulative lysosomal exposure) available as one-line alternatives. We may need them later; the high-content-imaging time course could eventually justify the integral. Default is the 24 h instantaneous value.

**Sanity checks (assert these):**
- At 34 µM the model must reproduce the calibration C_ly (regression guard against a broken import).
- C_ly must be monotonic non-decreasing in C_ext.

---

## STEP 3 — Sweep K_m and fit the injury function

This is where the headline uncertainty is handled.

For each **K_m in {50, 100, 200, 500} µM** (make this list a parameter):

1. Recompute `V_max = k_uptake * (K_m + 34)` for each cell line (keeping the 34 µM calibration exact), then redo the Step-2 concentration→C_ly mapping under that K_m.
2. **Fit the viability curve on the C_ly axis** with a Hill function, **asymptotes fixed**:

   ```
   viability(C_ly) = 100 / (1 + (C_ly / C_ly50) ** n)
   ```

   - Fit only **C_ly50** and **n** (top fixed at 100, bottom fixed at 0).
   - **Variance-weighted** least squares, weight ∝ 1/sd² (where sd present; unweighted if the arm has only NaN sds).
   - Fit RPTEC and NRK **separately**.
   - Report C_ly50, n, and a confidence interval on C_ly50 (e.g. from the covariance of the fit, or bootstrap over the points).

3. Record `(K_m, cell_line, C_ly50, C_ly50_CI, n)`.

The primary result is then **C_ly50 as a band across the K_m sweep**, per cell line. Present it as a table and as a plot (C_ly50 vs K_m, with CI whiskers).

**Corroboration fit (KE1 / LAMP):** LAMP rises with dose and has **no upper plateau in range** (it reaches ~1000%+), so do **not** use the fixed-asymptote Hill. Fit any monotonic increasing form (e.g. `100 + A * (C_ly**h)/(C_ly50L**h + C_ly**h)` with free top, or an exponential in log-C_ly), and extract the **C_ly at which LAMP crosses 2× control (200%)** as the "lysosomal-disturbance onset" load. This is used only for the ordering check in Step 4; do not over-invest in its functional form.

---

## STEP 4 — The two things this calibration is actually a test of

Report both explicitly; they matter more than the point estimates.

**(a) Human vs rat C_ly50 — does the sensitivity gap collapse?**
Human and rat differ ~10× in *nominal* EC50 (~57 µM vs ~575 µM). Jarzina attribute this to **uptake** (RPTEC accumulates more). If C_ly is the correct injury variable, then mapping each cell line through its *own* k_uptake should make the two **C_ly50 values converge**. For each K_m, report the ratio C_ly50(NRK) / C_ly50(RPTEC).
- Ratio ≈ 1 → strong support that C_ly is a cell-intrinsic injury threshold and the whole C_ly-indexing premise is sound.
- Ratio far from 1 → either the injury machinery genuinely differs between species, or the uptake calibration doesn't fully explain the sensitivity gap. Flag this prominently — it's a real finding either way.

**(b) AOP ordering on the C_ly axis.**
Check that the LAMP-onset C_ly (KE1, from Step 3) is **below** the viability C_ly50 (KE3) for each cell line — i.e. lysosomal disturbance precedes death in lysosomal-load space, consistent with the AOP. Report pass/fail.

**Identifiability / saturation diagnostic.**
For each K_m, plot **C_ly vs nominal concentration** across the data range. If C_ly **saturates** (plateaus) across the concentrations where viability is still falling, then several data points collapse onto near-identical C_ly with very different viabilities → C_ly50 is **poorly identified from the top of the curve**, and it is evidence that **non-lysosomal mechanisms** (mitochondrial etc.) contribute to killing at high dose. Detect this (e.g. flag if the C_ly spread across the viability-transition concentrations is below some fraction of a decade) and state it in the results. This is expected to be worse at low K_m.

---

## STEP 5 — Optional independent cross-check (Payasi 2024)

If `data/payasi_viability_rptec.csv` is provided (marketed-PMB arm; calcein/EthD-1 viability, 1–200 µM, RPTEC/hTERT1), overlay its viability points on the **predicted C_ly → death curve** (mapped through the human accumulation model). Treat this as a **qualitative** consistency check only, and print these caveats in the output:
- Payasi is a **3D perfused kidney-on-a-chip**, not Jarzina's static 2D — flow/shear raises endocytic capacity, so the C_ext→C_ly map is biased between the two systems. Expect an offset.
- Strong **conflict of interest** (authors/funder are the developer of the comparator formulation); use only the marketed-PMB arm.
Do **not** co-fit Payasi with Jarzina; it is a visual sanity check, nothing more.

---

## Outputs

Produce, under `injury_calibration/results/`:

1. `RECON.md` — the Step-0 findings.
2. A **results table** (CSV + rendered markdown): rows = (cell_line × K_m), columns = C_ly50, C_ly50_CI, n, and the human/rat ratio.
3. **Figures:**
   - viability vs C_ly with the fitted Hill, one panel per K_m, RPTEC and NRK overlaid;
   - C_ly50 vs K_m (the band), CI whiskers, both cell lines;
   - C_ly vs nominal concentration (the saturation diagnostic), per K_m;
   - LAMP vs C_ly with onset marker (corroboration);
   - if Payasi provided, the cross-check overlay.
4. `SUMMARY.md` — a short, plainly-worded write-up stating: the C_ly50 band (range across K_m) for each cell line; whether human and rat C_ly50 collapse (result (a)); whether AOP ordering holds (result (b)); and whether/where the saturation diagnostic fires. Lead with the caveat that C_ly50 is K_m-conditional and that K_m is unidentified from the single-dose calibration.

## Guardrails (recap)
- Repo reconnaissance first; import the existing model, never reconstruct or refactor it.
- No auto-digitisation; consume provided CSVs.
- Fix viability asymptotes at 100/0; fit only C_ly50 and n; variance-weight.
- Keep RPTEC and NRK on separate uptake parameters end-to-end.
- Never fit to Jarzina's fitted curves; exclude cathepsin D.
- Report C_ly50 as a K_m band, not a point; surface the human/rat collapse test, the AOP-ordering check, and the saturation diagnostic as first-class results.
