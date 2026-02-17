# Single-GPU Data Assimilation Experiment

## Overview

This directory contains a complete single-GPU data assimilation (DA) workflow at 1/4 degree
resolution in the Atlantic basin. The experiment follows a "perfect model" twin experiment design:

1. **Spinup**: Run the ocean model for 90 days to develop dynamical balance
2. **Nature run**: Continue for 14 days from the spun-up state (the "truth")
3. **Free run**: Start from a perturbed initial condition (noised + biased) with no DA
4. **Divergence test**: Verify that the free run diverges significantly from the nature run
5. **Ensemble DA**: Use a 10-member ETKF to keep the perturbed run close to truth

## Scientific Motivation

In a perfect-model OSSE (Observing System Simulation Experiment), we know the true ocean state
(the nature run). We create synthetic observations from this truth and test whether data
assimilation can recover the true state from perturbed initial conditions.

The perturbation adds:
- **Noise**: Random N(0, 0.5 C) perturbations to temperature, N(0, 0.05 psu) to salinity
- **Bias**: Systematic +0.3 C warm bias in temperature, +0.02 psu in salinity
- Applied to the upper 20 vertical levels only

## Grid and Domain

- Resolution: 1/4 degree (~25 km)
- Domain: Atlantic basin, 100W-0E, 10S-60N
- Grid size: 400 x 280 x 50
- Vertical: 50 levels with exponential stretching (fine near surface)

## Dates

- Spinup period: May 27, 2017 to Aug 25, 2017 (90 days)
- Nature/DA period: Aug 25, 2017 to Sep 8, 2017 (14 days)

## DA Algorithm: Ensemble Transform Kalman Filter (ETKF)

Following Manteia.jl's implementation (Hunt et al., 2007):

- **Ensemble size**: 10 members
- **Analysis window**: 1 day (14 analysis cycles)
- **Observations**: Surface T and S at all grid points (112,000 obs per field)
- **Updated fields**: 3D T and S
- **Inflation**: Multiplicative, factor 1.05
- **Observation error**: sigma_T = 0.1 C, sigma_S = 0.01 psu

The ETKF computes the analysis entirely in ensemble space (10x10 matrices),
making it efficient even with many observations.

## GPU Budget

| Step | Estimated | Actual | GPU-hours |
|------|-----------|--------|-----------|
| Spinup (90 days) | ~45 min | 37 min | 0.62 |
| Nature run (14 days) | ~20 min | 15 min | 0.25 |
| Free run (14 days) | ~20 min | 15 min | 0.25 |
| Comparison plots | ~10 min | 1 min | 0.02 |
| Ensemble DA (10x14 cycles) | ~90 min | 52 min | 0.87 |
| **Total** | **~3.25 hrs** | **2.0 hrs** | **2.01** |

Budget: 100 GPU-hours. Used: ~2.4 GPU-hours (including failed/debug runs).

## Directory Structure

```
single_gpu_da/
    README.md                 # This file (plan + results)
    nature_run.jl             # Step 2: 14-day nature run
    free_run.jl               # Step 3: 14-day free run (noised IC)
    compare_runs.jl           # Step 4: Compare nature vs free run
    ensemble_da.jl            # Step 5: 10-member ETKF
    slurm/
        01_pipeline.sh        # Job 1: spinup + nature + free + compare
        02_ensemble_da.sh     # Job 2: ETKF (depends on Job 1)
        02_ensemble_da_debug.sh  # Debug: 3-cycle ETKF (30-min debug queue)
        submit_all.sh         # Submit both jobs with dependency chain
        logs/                 # SLURM output logs
```

Output goes to: `$SCRATCH/DataAssimilocean/single_gpu_da/`

## How to Run

```bash
cd /global/u1/g/glwagner/DataAssimilocean.jl
bash single_gpu_da/slurm/submit_all.sh
```

This submits Job 1, then Job 2 with `--dependency=afterok:JOB1_ID`.

## Outputs

### From Job 1 (pipeline):
- `spinup/final_state.jld2` - Spun-up ocean state
- `spinup/surface_fields.jld2` - Surface fields during spinup
- `nature_run/surface_fields.jld2` - Nature run surface observations
- `nature_run/final_state.jld2` - Nature run final state
- `nature_run/nature_run_animation.mp4` - SST animation
- `free_run/surface_fields.jld2` - Free run surface fields
- `free_run/final_state.jld2` - Free run final state
- `free_run/free_run_animation.mp4` - SST animation
- `plots/comparison_animation.mp4` - Side-by-side nature vs free
- `plots/rmse_timeseries.png` - RMSE over time
- `plots/final_difference.png` - T, S difference maps at day 14

### From Job 2 (ensemble DA):
- `da_run/ensemble_diagnostics.jld2` - RMSE, spread time series
- `da_run/da_vs_nature_animation.mp4` - Ensemble mean SST vs nature
- `da_run/rmse_comparison.png` - RMSE: DA vs free run vs nature
- `da_run/ensemble_spread.png` - Ensemble spread over time
- `da_run/final_comparison.png` - Final state: nature, DA mean, free run

## Results

### GPU Usage Summary

| Job | SLURM ID | Wall Time | GPU-hrs | Status |
|-----|----------|-----------|---------|--------|
| Pipeline (spinup+nature+free+compare) | 48948679 | 68 min | 1.13 | COMPLETED |
| ETKF (first attempt) | 48948681 | 5 min | 0.08 | FAILED (bugs) |
| ETKF (full 14 cycles) | 48971858 | 52 min | 0.87 | COMPLETED |
| ETKF (debug, 3 cycles) | 48971883 | 21 min | 0.35 | COMPLETED |
| **Total** | | **146 min** | **2.43** | |

Budget: 100 GPU-hours. Used: 2.43 GPU-hours (2.4%).

### Spinup (90 days)
- **Status**: COMPLETED
- **Wall time**: 37 minutes
- **Final state T range**: (-1.88, 33.44) C
- **Final state S range**: (0.0, 40.91) psu

### Nature Run (14 days)
- **Status**: COMPLETED
- **Wall time**: 15 minutes
- Animation: `nature_run/nature_run_animation.mp4`

### Free Run / Divergence Test (14 days)
- **Status**: COMPLETED
- **Wall time**: 15 minutes
- **Perturbation**: T noise=0.5 C, T bias=+0.3 C; S noise=0.05 psu, S bias=+0.02 psu (upper 20 levels)
- **Initial SST RMSE**: 0.456 C
- **Final SST RMSE (day 14)**: 0.269 C
- **Final SSS RMSE (day 14)**: 0.069 psu
- SST RMSE **decreases** over time as atmospheric forcing damps the perturbation
- SSS RMSE **increases** over time (salinity has weaker restoring than temperature)
- Divergence confirmed: significant differences persist throughout the 14-day period

### Ensemble DA (10-member ETKF, 14 cycles)
- **Status**: COMPLETED
- **Wall time**: 42 min (DA loop) + 10 min (compilation)
- **Final forecast SST RMSE**: 0.237 C
- **Final analysis SST RMSE**: 0.237 C
- **Free run SST RMSE (day 14)**: 0.269 C
- **Improvement over free run**: 12% reduction in SST RMSE

#### ETKF cycle-by-cycle SST RMSE:

| Cycle | Day | Forecast RMSE (C) | Analysis RMSE (C) | Spread (C) |
|-------|-----|-------------------|-------------------|------------|
| 1 | 1 | 0.348 | 0.348 | 0.161 |
| 2 | 2 | 0.324 | 0.324 | 0.020 |
| 3 | 3 | 0.310 | 0.310 | 0.013 |
| 4 | 4 | 0.300 | 0.300 | 0.010 |
| 5 | 5 | 0.292 | 0.292 | 0.008 |
| 7 | 7 | 0.277 | 0.277 | 0.006 |
| 10 | 10 | 0.258 | 0.258 | 0.004 |
| 14 | 14 | 0.237 | 0.237 | 0.003 |

### Diagnosis: Ensemble Collapse

The ETKF shows a classic **ensemble collapse** problem:

1. **Spread collapses rapidly**: From 0.161 C (cycle 1) to 0.003 C (cycle 14), while RMSE stays at ~0.24 C
2. **Analysis updates are negligible**: Forecast RMSE ≈ Analysis RMSE at every cycle, meaning the ETKF weight is almost entirely on the forecast (not observations)
3. **Root cause**: With spread << observation error (0.003 C vs 0.1 C), the Kalman gain is near zero

**Why this happens**:
- 10 ensemble members is small for 112,000 surface observations per field
- Multiplicative inflation (1.05) is too weak to maintain spread
- No spatial localization of covariances
- All members converge because they share the same atmospheric forcing

**Recommended improvements for next iteration**:
1. **Additive inflation**: Add stochastic perturbations after each analysis to maintain spread
2. **Stronger multiplicative inflation**: Try 1.5-2.0 instead of 1.05
3. **Covariance localization**: Limit correlation distance (Gaspari-Cohn function)
4. **Observation thinning**: Assimilate every Nth grid point instead of all 112K points
5. **More ensemble members**: 20-50 would help, but increases compute cost linearly

### Plots Generated

All plots are in `$SCRATCH/DataAssimilocean/single_gpu_da/`:

- `plots/rmse_timeseries.png` - SST and SSS RMSE: free run vs nature run
- `plots/final_difference.png` - SST/SSS difference maps at day 14
- `plots/comparison_animation.mp4` - Side-by-side animation of nature vs free run
- `nature_run/nature_run_animation.mp4` - Nature run SST animation
- `nature_run/nature_run_final.png` - Nature run final SST snapshot
- `free_run/free_run_animation.mp4` - Free run SST animation
- `free_run/free_run_final.png` - Free run final SST snapshot
- `da_run/rmse_comparison.png` - RMSE: DA vs free run
- `da_run/ensemble_spread.png` - Ensemble spread diagnostics
- `da_run/final_comparison.png` - Final SST: nature, DA mean, difference
