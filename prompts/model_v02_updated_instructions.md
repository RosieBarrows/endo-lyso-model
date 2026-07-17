# Updated Instructions — Intracellular Accumulation Model v0.2

These instructions **replace** the data, fitting procedure, and outputs from the original instructions. The model structure (3-compartment ODE), fixed trafficking parameters, solver choice (BDF), and unit conversion (the corrected version you implemented) are all unchanged.

**Drop all colistin work.** Steps 3a/3b from the original instructions, and all colistin data, are removed. We are fitting polymyxin B only, in two cell lines.

---

## 1. Replacement data

Discard the data tables from the original instructions and from jarzina_data.xlsx. Use these digitised values instead.

### RPTEC/TERT1 + polymyxin B (with error bars)

| Time (h) | Central (nM) | Lower bound (nM) | Upper bound (nM) |
|----------|-------------|-------------------|-------------------|
| 0 | 0 | 0 | 0 |
| 3.53 | 636 | 442 | 883 |
| 6.44 | 1377 | 870 | 1909 |
| 24.0 | 3558 | 1974 | 5156 |
| 47.46 | 2792 | 1494 | 4065 |

### NRK-52E + polymyxin B (no error bars)

| Time (h) | Central (nM) |
|----------|-------------|
| 0 | 0 |
| 3.53 | 299 |
| 6.44 | 662 |
| 24.0 | 831 |
| 47.46 | 104 |

Notes on these data:
- The t = 0 point is an exact physical constraint (cells have never seen drug), not a measurement.
- Time points are digitised from the figure — the first non-zero point is at ~3.5h, not 2h as previously estimated.
- RPTEC/TERT1 error bars are asymmetric. Store lower and upper bounds separately.
- NRK-52E has no error bars in the figure.

---

## 2. Replacement fitting procedure

### Residuals

Use **log-space residuals**: minimise Σ[log10(model) - log10(data)]² across time points.

For the t = 0 point: exclude it from the residual calculation. The model passes through zero by construction (all initial conditions are zero), so it adds no information to the fit. Including it would require handling log(0), which is undefined.

### Step 1: Fit RPTEC/TERT1

- **Free parameters:** k_uptake_RPTEC, k_deg_PB
- **Fixed parameters:** k_mat = 0.04, k_rec = 0.02, k_fuse = 0.02, k_esc = 0.0002 (all /min)
- **Data:** 4 non-zero RPTEC/TERT1 points (3.53h, 6.44h, 24h, 47.46h)
- **Exposure protocol:** C_ext = 34 µM for t ∈ [0, 24h], C_ext = 0 for t ∈ (24h, 47.46h]
- **Cell volume:** V_cell = 1.99e-12 L
- **Bounds:** k_uptake in [1e-9, 1e-4], k_deg in [1e-5, 1e-1] (wide bounds — let the data speak)
- **Optimiser:** scipy.optimize.least_squares, method 'trf', working in log10 of parameters

### Step 2A: Fit NRK-52E (k_deg fixed from Step 1)

- **Free parameter:** k_uptake_NRK only
- **Fixed:** k_deg_PB from Step 1; all trafficking parameters unchanged
- **Data:** 4 non-zero NRK-52E points
- **Cell volume:** V_cell = 1.21e-12 L
- **Bounds:** k_uptake in [1e-9, 1e-4]

This tests the hypothesis that the cell-line difference is purely an uptake rate difference.

### Step 2B: Fit NRK-52E (k_deg also free)

- **Free parameters:** k_uptake_NRK, k_deg_NRK
- **Everything else same as Step 2A**

This relaxes the assumption that k_deg is cell-line-independent. Compare the fitted k_deg_NRK to k_deg_PB from Step 1. If k_deg_NRK is substantially higher (say >3× higher), that suggests a real species difference in lysosomal processing.

Report both Step 2A and Step 2B results. We expect Step 2A will fit the uptake phase well but overpredict the 47.46h washout value, because the RPTEC-derived k_deg implies ~40% clearance in 24h but the NRK data shows ~88% clearance.

---

## 3. Stress-test of fixed trafficking parameters

After Steps 1 and 2, run the following analysis to assess how sensitive the **fitted** parameter values are to our assumptions about the fixed trafficking parameters.

### Procedure

For each of the 4 fixed parameters (k_mat, k_rec, k_fuse, k_esc), vary it across 5 values spanning an order of magnitude either side of its default, while keeping the other 3 fixed parameters at their defaults:

| Parameter | Default | Test values |
|-----------|---------|-------------|
| k_mat | 0.04 | [0.004, 0.01, 0.04, 0.1, 0.4] |
| k_rec | 0.02 | [0.002, 0.005, 0.02, 0.05, 0.2] |
| k_fuse | 0.02 | [0.002, 0.005, 0.02, 0.05, 0.2] |
| k_esc | 0.0002 | [0.00002, 0.00005, 0.0002, 0.0005, 0.002] |

For each of the 20 test cases (4 parameters × 5 values):
1. Refit Step 1 (k_uptake_RPTEC and k_deg to RPTEC data)
2. Refit Step 2A (k_uptake_NRK to NRK data, using the k_deg from the refitted Step 1)
3. Record: fitted k_uptake_RPTEC, fitted k_deg, fitted k_uptake_NRK, ratio k_uptake_RPTEC/k_uptake_NRK, total SSE for each fit

This gives us 20 sets of fitted values. We're looking for:
- **Stability of fitted values:** Do k_uptake and k_deg stay roughly the same regardless of what we assume for the trafficking parameters? If yes, the model is robust. If no, the fixed parameter uncertainty is a problem.
- **Stability of the cell-line ratio:** Even if absolute values of k_uptake shift, does the ratio k_uptake_RPTEC/k_uptake_NRK stay constant? If yes, the relative comparison is trustworthy even if absolute values aren't.
- **Any cases where the fit fails:** Does any trafficking parameter value make it impossible to fit the data? That would rule out that region of parameter space.

---

## 4. Required outputs

### Output 1: Time-course fits (2 panels, not 4)

Create a figure with **2 subplots** side by side:
- Left: RPTEC/TERT1 + polymyxin B
- Right: NRK-52E + polymyxin B

Each subplot:
- X-axis: time (hours), 0 to 50
- Y-axis: intracellular concentration (nM)
- Vertical dashed line at t = 24h (washout start)
- Data as markers; for RPTEC include vertical error bars from digitised bounds
- Model fit as solid line

For the NRK-52E panel, show **both** Step 2A (k_deg fixed, solid line) and Step 2B (k_deg free, dashed line) so the difference is visible.

### Output 2: Compartment breakdown (RPTEC/TERT1 only)

Stacked area plot showing C_ee, C_le, C_ly contributions over time for the RPTEC/TERT1 fit. Report the percentage in each compartment at t = 24h.

### Output 3: Parameter summary table

Print a table with:

| Quantity | Value |
|----------|-------|
| k_mat (fixed) | ... |
| k_rec (fixed) | ... |
| k_fuse (fixed) | ... |
| k_esc (fixed) | ... |
| k_uptake_RPTEC (Step 1) | ... |
| k_deg_PB (Step 1) | ... |
| k_uptake_NRK (Step 2A, k_deg fixed) | ... |
| k_uptake_NRK (Step 2B, k_deg free) | ... |
| k_deg_NRK (Step 2B) | ... |
| Ratio k_uptake_RPTEC / k_uptake_NRK (Step 2A) | ... |
| Ratio k_uptake_RPTEC / k_uptake_NRK (Step 2B) | ... |
| Ratio k_deg_NRK / k_deg_PB (Step 2B) | ... |
| Lysosomal half-life from k_deg_PB | ... hours |
| Lysosomal half-life from k_deg_NRK | ... hours |
| SSE Step 1 | ... |
| SSE Step 2A | ... |
| SSE Step 2B | ... |

### Output 4: Stress-test results

Create **4 subplots** (one per fixed parameter being varied):
- X-axis: fixed parameter value (log scale)
- Y-axis (left): fitted k_uptake_RPTEC and k_deg (log scale, two lines)
- Y-axis (right) or second row: the ratio k_uptake_RPTEC / k_uptake_NRK

Mark the default fixed parameter value with a vertical dashed line.

Also create a summary statement: for each fixed parameter, report the fold-change in fitted k_uptake and k_deg across the tested range. Example: "Varying k_mat by 100× changes fitted k_uptake by X× and fitted k_deg by Y×."

### Output 5: Lysosomal load trajectory

Same as before but only for RPTEC/TERT1. Plot C_ly over time with horizontal threshold lines at 500, 1000, 2000, 3000 nM. Annotate threshold-crossing times.

---

## 5. Things to flag in commentary

After running everything, comment on:

1. **Does the RPTEC/TERT1 fit fall within the error bars at all time points?**

2. **Step 2A vs 2B for NRK-52E:** How much worse is the Step 2A fit (fixed k_deg) compared to Step 2B (free k_deg)? Specifically, does Step 2A capture the uptake phase but miss the washout, as we expect?

3. **What does the fitted k_deg_NRK imply?** If it's much higher than k_deg_RPTEC, discuss whether this could reflect a real species difference in lysosomal degradation, or whether the NRK washout data point (104 nM) might simply be unreliable (near LOD).

4. **Stress-test stability:** Are k_uptake and k_deg stable across the fixed parameter ranges? Is the cell-line ratio stable? Flag any fixed parameter that causes large swings in fitted values.

5. **Where is the drug at 24h?** Report % in each compartment.
