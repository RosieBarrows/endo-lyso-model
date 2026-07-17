# Updated Instructions — Michaelis-Menten Uptake and Concentration Sweep v0.3

## Context

The v0.2 concentration sweep showed that the linear uptake approximation (k_uptake × C_ext) produces physically implausible predictions at high doses — at 2000 µM, the model predicts ~230 µM intracellular, exceeding the extracellular concentration. This is because the linear form has no saturation. We need to restore the full Michaelis-Menten uptake term so the model behaves correctly at concentrations above K_m.

## What to change

### Replace the uptake term

In the ODE for dC_ee/dt, replace:

```
uptake = k_uptake * C_ext
```

with:

```
uptake = V_max * C_ext / (K_m + C_ext)
```

Where:
- V_max = k_uptake_fitted × K_m (back-calculated so that at low C_ext, the behaviour matches the v0.2 fit)
- K_m is a new explicit parameter (µM)

This preserves the v0.2 calibration: when C_ext << K_m, V_max × C_ext / K_m ≈ k_uptake × C_ext, so the 34 µM fit is unchanged. At high C_ext >> K_m, uptake saturates at V_max.

Do the same for NRK-52E: V_max_NRK = k_uptake_NRK_fitted × K_m.

### Keep all other parameters from v0.2

k_deg, k_mat, k_rec, k_fuse, k_esc — all unchanged from the v0.2 Step 1 / Step 2A fits.

## What to run

### Sweep over K_m values

Run the concentration sweep at **4 values of K_m**: 50, 100, 200, 500 µM.

For each K_m value:
- Back-calculate V_max_RPTEC = k_uptake_RPTEC × K_m
- Back-calculate V_max_NRK = k_uptake_NRK × K_m
- Simulate RPTEC/TERT1 at C_ext = 34, 125, 250, 500, 1000, 2000 µM
- 24h exposure + 24h washout (same protocol as before)

### Sanity check

For each K_m and each C_ext, verify that peak intracellular concentration < C_ext (in matching units). If any case still exceeds extracellular concentration, flag it. At K_m = 50 µM, the 34 µM calibration point is no longer in the linear regime (C_ext/K_m = 0.68), so the fit quality at 34 µM may degrade slightly — report this.

## Required outputs

### Output 1: Fit preservation check

A small 2-panel figure:
- Left: RPTEC/TERT1 at 34 µM, showing the v0.2 data points with error bars and 4 overlaid model curves (one per K_m value). These should nearly overlap if the back-calculation is working — but K_m = 50 µM may deviate because the linear approximation is less accurate there.
- Right: NRK-52E at 34 µM, same layout.

This confirms we haven't broken the calibration.

### Output 2: Lysosomal load vs time, RPTEC/TERT1

A figure with **4 subplots** (one per K_m value). Each subplot shows C_ly vs time for all 6 dose levels (34, 125, 250, 500, 1000, 2000 µM) overlaid as separate lines.

Use the same y-axis limits across all 4 subplots so the effect of K_m on saturation is visually obvious. Include horizontal dashed lines at 500 and 1000 nM as reference thresholds.

Label each subplot with the K_m value and the corresponding V_max_RPTEC.

### Output 3: Threshold crossing time vs dose

A single figure with **2 threshold levels** (500 nM and 1000 nM) as separate line styles:
- X-axis: C_ext (µM), log scale, from 34 to 2000
- Y-axis: time to cross threshold (hours)
- 4 curves per threshold (one per K_m value), distinguished by colour
- Legend indicating K_m value and threshold

This is the key translational output — it answers "for a given extracellular concentration and assumed K_m, how long until the lysosome reaches a harmful load?"

### Output 4: Peak lysosomal concentration vs dose

A single figure:
- X-axis: C_ext (µM), log scale
- Y-axis: peak C_ly during the 48h simulation (nM), log scale
- 4 curves (one per K_m), distinguished by colour
- A diagonal dashed line showing what the linear model would predict (for reference)

This shows where saturation kicks in for each K_m value. At K_m = 50 µM, the curve should flatten noticeably above ~100 µM. At K_m = 500 µM, it should stay close to linear across the whole range.

### Output 5: Summary table

Print a table with one row per (K_m, C_ext) combination for RPTEC/TERT1:

| K_m (µM) | C_ext (µM) | V_max | Peak total intracellular (nM) | Peak C_ly (nM) | Time to C_ly > 500 nM (h) | Time to C_ly > 1000 nM (h) |
|-----------|-----------|-------|-------------------------------|----------------|---------------------------|----------------------------|

That's 4 × 6 = 24 rows.

## Things to flag

1. **Does K_m = 50 µM break the 34 µM calibration?** If the fit at 34 µM is noticeably worse for K_m = 50, report by how much (e.g. "34 µM intracellular prediction drops by X% relative to v0.2").

2. **At which K_m does 2000 µM first produce a physically plausible result?** (i.e. peak intracellular < extracellular)

3. **How sensitive are the threshold crossing times to K_m?** At 125 µM (the dose where Jarzina saw lysosomal effects at 1-2h), what is the range of predicted crossing times across K_m = 50-500 µM?

4. **Does NRK-52E show the same saturation pattern?** Briefly run the 125 µM case for NRK-52E at each K_m and report whether the cell-line ratio in peak C_ly stays near ~8× or changes with saturation.
