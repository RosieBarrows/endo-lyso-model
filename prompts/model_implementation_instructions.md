# Instructions: Implement an Intracellular Drug Accumulation Model in Python

## Context

I am building a compartmental ODE model of drug accumulation inside a single proximal tubule (PT) epithelial cell. The model tracks how polymyxin B moves from the extracellular culture medium through the endosomal trafficking pathway and accumulates in lysosomes.

This model will eventually be used as the intracellular module of a larger multi-scale model of oligonucleotide-induced nephrotoxicity (the ORCA model). For now, we are calibrating it against in vitro data from Jarzina et al. (2022), Frontiers in Toxicology 4:864441.

## The model

### State variables

| Variable | Description | Units |
|----------|-------------|-------|
| C_ee | Drug amount in early endosomes (per cell) | fmol/cell |
| C_le | Drug amount in late endosomes (per cell) | fmol/cell |
| C_ly | Drug amount in lysosomes (per cell) | fmol/cell |

C_ext (extracellular drug concentration) is treated as a **fixed constant** during the exposure phase. This is justified because the total cellular uptake over 24h is negligible relative to the drug in the medium (cells were exposed to 34 µM in ~1 mL medium = 34 nmol total; total uptake is on the order of hundreds of pmol). During the washout phase (drug removed from medium), C_ext = 0.

### Ordinary differential equations

```
dC_ee/dt = k_uptake * C_ext - (k_mat + k_rec) * C_ee

dC_le/dt = k_mat * C_ee - (k_fuse + k_esc) * C_le

dC_ly/dt = k_fuse * C_le - k_deg * C_ly
```

Where:
- `k_uptake` = V_max / K_m — lumped uptake rate constant (fmol/cell/min per µM). This is the linear approximation of Michaelis-Menten uptake, valid when C_ext << K_m. We use the lumped form because V_max and K_m are not separately identifiable from data at a single concentration.
- `k_mat` = EE→LE maturation rate constant (/min)
- `k_rec` = EE→surface recycling rate constant (/min)
- `k_fuse` = LE→lysosome fusion rate constant (/min)
- `k_esc` = LE→cytosol escape rate constant (/min)
- `k_deg` = lysosomal degradation rate constant (/min)

### Model observable

The experimentally measured quantity (LC-MS/MS intracellular concentration) corresponds to:

```
C_total_intracellular = C_ee + C_le + C_ly
```

This is reported in nM (intracellular concentration). To convert from fmol/cell to nM:

```
C_intracellular_nM = (C_ee + C_le + C_ly) / V_cell * 1e6
```

Where V_cell is the cell volume in litres. Jarzina et al. measured cell volumes by imaging trypsinised (rounded) cells:
- RPTEC/TERT1: mean diameter ~15.6 µm → V_cell ≈ 1990 µm³ ≈ 1.99e-12 L
- NRK-52E: mean diameter ~13.2 µm → V_cell ≈ 1205 µm³ ≈ 1.21e-12 L

(These are approximate — the paper calculates V = (1/6)*π*d³. Use these values.)

## Experimental data to reproduce

The primary calibration target is Jarzina et al. (2022) Figure 6A. I will provide the approximate data points digitised from the figure. If I do not upload the data, use the following values read from the figure:

### RPTEC/TERT1 cells, polymyxin B, 34 µM continuous exposure:

| Time (hours) | Intracellular concentration (nM) |
|--------------|----------------------------------|
| 0.017 (1 min) | ~50 |
| 1 | ~400 |
| 3 | ~800 |
| 6 | ~1500 |
| 24 | ~5000 |
| 48 (24h recovery) | ~2500 |

### NRK-52E cells, polymyxin B, 34 µM continuous exposure:

| Time (hours) | Intracellular concentration (nM) |
|--------------|----------------------------------|
| 0.017 (1 min) | ~30 |
| 1 | ~150 |
| 3 | ~250 |
| 6 | ~400 |
| 24 | ~1000 |
| 48 (24h recovery) | ~500 |

### RPTEC/TERT1 cells, colistin, 34 µM continuous exposure:

| Time (hours) | Intracellular concentration (nM) |
|--------------|----------------------------------|
| 0.017 (1 min) | ~30 |
| 1 | ~200 |
| 3 | ~400 |
| 6 | ~700 |
| 24 | ~2500 |
| 48 (24h recovery) | ~1200 |

### NRK-52E cells, colistin, 34 µM continuous exposure:

| Time (hours) | Intracellular concentration (nM) |
|--------------|----------------------------------|
| 0.017 (1 min) | ~20 |
| 1 | ~80 |
| 3 | ~120 |
| 6 | ~200 |
| 24 | ~500 |
| 48 (24h recovery) | ~250 |

**Important:** These are approximate values read from the figure. Error bars in the original data are substantial. Treat these as order-of-magnitude targets, not precise calibration points. The LOD was ~50 nM and LOQ was ~15 nM.

**Washout protocol:** The experiment runs for 24 hours with drug in the medium (C_ext = 34 µM). At t = 24h, the medium is replaced with fresh drug-free medium (C_ext = 0). The "48h" time point is actually t = 48h from the start, i.e. 24h of exposure + 24h of recovery.

## Parameters

### Parameters to be fixed from literature

These trafficking rate constants are properties of the cellular machinery, not the drug. Fix them at these values initially:

| Parameter | Value | Units | Rationale |
|-----------|-------|-------|-----------|
| k_mat | 0.04 | /min | EE→LE maturation takes ~15-30 min (half-life ~17 min); consistent with Rab5→Rab7 conversion timescale (Rink et al. 2005) |
| k_rec | 0.02 | /min | EE→surface recycling; roughly half the maturation rate, meaning ~1/3 of endocytosed material is recycled. Consistent with megalin recycling estimates |
| k_fuse | 0.02 | /min | LE→lysosome fusion takes ~30-60 min (half-life ~35 min); consistent with Luzio et al. 2007 |
| k_esc | 0.0002 | /min | ~1% of LE content escapes to cytosol (Gilleron et al. 2013); set as ~1% of k_fuse |

### Parameters to be fitted

| Parameter | Initial guess | Bounds | Units | Notes |
|-----------|--------------|--------|-------|-------|
| k_uptake | 0.01 | [0.001, 0.1] | fmol/cell/min/µM | Lumped uptake rate. Will differ between cell lines |
| k_deg | 0.002 | [0.0001, 0.05] | /min | Lysosomal degradation rate. Will differ between drugs |

## Fitting procedure

Implement the following step-by-step fitting procedure:

### Step 1: Fit RPTEC/TERT1 + polymyxin B

This is the primary dataset — the cell line with the most uptake and the drug with the most accumulation.

- Free parameters: `k_uptake_RPTEC` and `k_deg_PB`
- Fixed parameters: k_mat, k_rec, k_fuse, k_esc as above
- C_ext = 34 µM for t ∈ [0, 24h], C_ext = 0 for t ∈ [24h, 48h]
- V_cell = 1.99e-12 L
- Fit to minimise sum of squared errors between model-predicted C_total_intracellular (in nM) and the 6 data points
- Use scipy.optimize.minimize or scipy.optimize.least_squares with bounds

### Step 2: Fit NRK-52E + polymyxin B

- Free parameter: `k_uptake_NRK52E` only
- Fixed: k_deg_PB from Step 1, all trafficking rates unchanged
- C_ext = 34 µM / 0, V_cell = 1.21e-12 L
- Rationale: the two cell lines differ in endocytic activity (Jarzina showed higher aprotinin uptake in RPTEC/TERT1). We capture this by changing only k_uptake.

### Step 3: Fit RPTEC/TERT1 + colistin

- Free parameter: `k_deg_colistin` (or `k_uptake_colistin` — try both, see which gives better fit)
- Fixed: k_uptake_RPTEC from Step 1, all trafficking rates unchanged
- Rationale: colistin is a structural analogue of polymyxin B with lower toxicity. The difference could be in uptake affinity (different K_m → different k_uptake) or in lysosomal stability (different k_deg), or both.

### Step 4: Predict NRK-52E + colistin

- No fitting. Use k_uptake_NRK52E from Step 2 and k_deg_colistin from Step 3.
- Compare prediction to data. This is a **validation test** — if the model can predict the fourth condition from parameters fitted to the other three, that's evidence the model structure is capturing something real.

## Implementation details

### Solver

Use `scipy.integrate.solve_ivp` with method `'BDF'` or `'Radau'` (stiff solvers). The system may be mildly stiff because k_mat and k_rec are much larger than k_deg.

### Time handling

The simulation has two phases:
1. **Exposure phase** (0 to 24h = 0 to 1440 min): C_ext = 34 µM
2. **Washout phase** (24h to 48h = 1440 to 2880 min): C_ext = 0 µM

Implement this as either:
- Two sequential solve_ivp calls, using the final state of phase 1 as the initial condition for phase 2, OR
- A single solve_ivp call with C_ext implemented as a step function using events or a conditional inside the ODE function

Use minutes as the time unit internally (to match the /min rate constants), but plot in hours.

### Initial conditions

All intracellular compartments start at zero:
```
C_ee(0) = 0
C_le(0) = 0
C_ly(0) = 0
```

### Optimisation

Use `scipy.optimize.least_squares` with method `'trf'` (trust region reflective, supports bounds). The residual function should return the vector of differences between model predictions and data at each time point.

Work in log-space for the parameters (i.e. fit log10(k_uptake) and log10(k_deg)) to ensure they stay positive and to handle the wide range of possible values.

## Required outputs

### Output 1: Time-course plots

Create a figure with **4 subplots** (2×2 grid):
- Top-left: RPTEC/TERT1 + polymyxin B
- Top-right: NRK-52E + polymyxin B
- Bottom-left: RPTEC/TERT1 + colistin
- Bottom-right: NRK-52E + colistin (prediction)

Each subplot should show:
- X-axis: time (hours), 0 to 48
- Y-axis: intracellular concentration (nM)
- A vertical dashed line at t = 24h marking the start of washout
- The experimental data as points with error bars (if available) or as markers
- The model fit/prediction as a solid line
- Title indicating cell line and drug

Use consistent y-axis limits across all subplots so the magnitude differences between conditions are visually apparent.

### Output 2: Compartment breakdown

Create a second figure with **4 subplots** (same 2×2 layout) showing the **stacked contributions** of each intracellular compartment over time:
- C_ee (early endosome)
- C_le (late endosome)
- C_ly (lysosome)

Use a stacked area plot or stacked line plot. The total should match the solid line in Output 1. This tells us where the drug is at any given time — we expect the lysosomal compartment to dominate by 24h.

### Output 3: Parameter summary table

Print a table showing:
- All fixed parameter values
- All fitted parameter values with the condition they were fitted to
- The ratio k_uptake_RPTEC / k_uptake_NRK52E (should be ~3-5× if the model is capturing the cell line difference correctly)
- The ratio k_deg_colistin / k_deg_PB (tells us the relative lysosomal stability)
- The sum of squared errors for each fit
- The prediction error for the NRK-52E + colistin validation case

### Output 4: Sensitivity analysis

Run a one-at-a-time sensitivity analysis on the RPTEC/TERT1 + polymyxin B case. For each parameter (all 6: k_uptake, k_mat, k_rec, k_fuse, k_esc, k_deg):
- Vary it by ±50% from its fitted/fixed value
- Record the change in C_total_intracellular at t = 24h
- Plot a tornado diagram showing which parameters most influence the 24h accumulation

### Output 5: Lysosomal load trajectory

For the RPTEC/TERT1 + polymyxin B case, plot C_ly alone (not the total) as a function of time. Overlay horizontal dashed lines at arbitrary threshold values (e.g. 1000 nM, 2000 nM, 3000 nM). Annotate the time at which each threshold is crossed.

This output is a preview of how the model would connect to the AOP — the threshold crossing time would correspond to the onset of KE1 (lysosomal dysfunction). We expect the model to predict threshold crossing at a time consistent with the Jarzina imaging data (lysosomal effects visible at 1-2h at 125 µM in RPTEC/TERT1, but this is at a higher concentration than 34 µM, so effects would be later at 34 µM).

## Code structure

Please organise the code as follows:

```
# 1. Import statements

# 2. Constants and fixed parameters (clearly labelled)

# 3. ODE system function
#    - Takes t, y, params, C_ext as arguments
#    - Returns [dC_ee/dt, dC_le/dt, dC_ly/dt]

# 4. Simulation function
#    - Takes parameters and experimental conditions (C_ext, t_washout, V_cell)
#    - Runs solve_ivp for exposure + washout phases
#    - Returns time array and total intracellular concentration in nM

# 5. Residual function for fitting
#    - Takes parameter vector, calls simulation function
#    - Returns vector of (model - data) residuals

# 6. Experimental data (hardcoded as arrays)

# 7. Fitting: Steps 1-4 as described above

# 8. Plotting: Outputs 1-5 as described above

# 9. Print parameter summary table
```

Write the entire script in a single Python file. Use clear comments throughout. Use matplotlib for plotting. Save all figures as PNG files.

## What to flag

After running the model, comment on:

1. **Does the model reproduce the uptake curve shape?** Is the predicted curve monotonically increasing during exposure and decreasing during washout, matching the data?

2. **Is the cell line difference captured by changing only k_uptake?** Is the ratio k_uptake_RPTEC / k_uptake_NRK52E biologically reasonable (expect ~3-5×)?

3. **Does the NRK-52E + colistin prediction match the data?** This is the validation test. If the prediction is poor, discuss what additional mechanism might be needed.

4. **Where is the drug at 24h?** What fraction is in lysosomes vs endosomes? We expect >90% in lysosomes by 24h.

5. **Which parameters is the model most sensitive to?** Does the sensitivity analysis suggest that the trafficking rates matter, or is it dominated by k_uptake and k_deg?

6. **Are there any numerical issues?** Stiffness, negative concentrations, solver failures?
