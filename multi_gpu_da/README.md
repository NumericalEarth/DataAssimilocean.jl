# Multi-GPU Data Assimilation Experiment (4 GPUs, 1/8 degree)

## Lessons Learned from Single-GPU Experiment

### What Worked
- **Pipeline structure**: Spinup → nature run → free run → comparison → ETKF is clean and testable
- **Sequential ensemble on single GPU**: Running 10 members one at a time on 1 GPU, storing state in CPU memory, avoids needing 10× GPU memory
- **ETKF in ensemble space**: Computing 10×10 weight matrices is cheap even with many observations
- **Warm-up run**: A single-timestep run before the DA loop compiles all GPU kernels (~10 min), then subsequent runs are fast
- **Bathymetry caching**: Saving computed bathymetry avoids expensive re-computation
- **Debug queue**: A 3-cycle debug job (30 min) validates code before committing to a long run

### Bugs and Pitfalls
1. **Dual clocks**: `OceanSeaIceModel` has separate `simulation.model.clock` and `ocean.model.clock` - must reset BOTH before re-running a member
2. **Staggered grid fields**: `set!()` works for T, S, e, u, v but NOT for w and eta - use `copyto!(interior(...), data)` for those
3. **String interpolation with unicode**: `$sigma_T°C` parses as variable `sigma_T°C`. Use `$(sigma_T) C` instead
4. **Vector outer product**: `w_mean * ones(1, N)` fails; use `w_mean * ones(N)'` for outer product
5. **SLURM log buffering**: On some Perlmutter nodes, stderr/stdout are buffered until job completion - don't panic if logs appear empty while the job runs

### Ensemble Collapse (Key Finding)
- 10 members with 112K surface observations per field → rapid spread collapse (0.16 → 0.003 C)
- Multiplicative inflation of 1.05 is far too weak
- Analysis updates are negligible when spread << observation error
- DA improvement only 12% over free run (0.237 vs 0.269 C RMSE)

### Recommendations Applied in This Experiment
- **Observation thinning**: Assimilate every 4th grid point (reduces obs from 448K to ~28K per field)
- **Stronger inflation**: Use 1.5 multiplicative factor instead of 1.05

## Overview

This experiment scales the single-GPU DA workflow to 4 GPUs using MPI at 1/8 degree resolution.
This is true **weak scaling**: each GPU gets 400×280×50 grid points, identical to the single-GPU case.

| Parameter | Single-GPU | Multi-GPU |
|-----------|-----------|-----------|
| Resolution | 1/4 degree | 1/8 degree |
| Grid size | 400×280×50 | 800×560×50 |
| GPUs | 1 | 4 (Partition 2×2) |
| Points/GPU | 5.6M | 5.6M (same!) |
| Surface obs (raw) | 112K | 448K |
| Surface obs (thinned) | N/A | ~28K |
| Free surface substeps | 30 | 60 |
| Time step | 5 min | 5 min |

## Workflow

1. **Spinup**: 90 days at 1/8 degree on 4 GPUs (MPI)
2. **Nature run**: 14 days from spinup end state (MPI)
3. **Free run**: 14 days from perturbed IC (MPI)
4. **Comparison**: RMSE and plots (serial, rank 0)
5. **Ensemble DA**: 10-member ETKF with daily cycles (MPI)

## Grid and Domain

- Resolution: 1/8 degree (~12.5 km)
- Domain: Atlantic basin, 100W-0E, 10S-60N
- Grid size: 800 × 560 × 50 (global)
- MPI decomposition: 2×2 (each rank: 400×280×50)
- Vertical: 50 levels with exponential stretching

## DA Algorithm Changes from Single-GPU

### Observation Thinning
- Assimilate every 4th grid point in each horizontal direction
- Thinned local surface: 100×70 = 7,000 per rank per field
- Total thinned obs (T+S): ~56,000
- Ratio to ensemble size: 5,600:1 (vs 11,200:1 in single-GPU)

### Stronger Inflation
- Multiplicative inflation: 1.5 (vs 1.05 in single-GPU)

### MPI-Aware ETKF
- Each rank stores its local ensemble states in CPU memory
- Surface observations gathered via MPI.Allgather (thinned)
- ETKF weights computed independently on all ranks (deterministic)
- Weight application is local (no MPI needed)

## Directory Structure

```
multi_gpu_da/
    README.md                 # This file
    spinup.jl                 # MPI spinup (replaces stage1_spinup.jl)
    nature_run.jl             # MPI nature run
    free_run.jl               # MPI free run (noised IC)
    compare_runs.jl           # Comparison (serial post-processing)
    ensemble_da.jl            # MPI ETKF with observation thinning
    slurm/
        01_pipeline.sh        # Spinup + nature + free + compare (4 GPUs)
        02_ensemble_da.sh     # ETKF (4 GPUs)
        submit_all.sh         # Submit both with dependency
        logs/
```

Output goes to: `$SCRATCH/DataAssimilocean/multi_gpu_da/`

## Environment Setup (Critical for MPI)

Two environment variables are **required** in SLURM scripts for GPU-aware MPI on Perlmutter:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_CUDA_MEMORY_POOL=none
```

Without `JULIA_CUDA_MEMORY_POOL=none`, CUDA.jl's stream-ordered memory pool (`cudaMallocAsync`) is incompatible with `cuIpcGetMemHandle`, which Cray MPICH uses for same-node GPU-to-GPU transfers. This causes `cuIpcGetMemHandle: invalid argument, CUDA_ERROR_INVALID_VALUE` errors during halo communication.

## How to Run

```bash
cd /global/u1/g/glwagner/DataAssimilocean.jl
bash multi_gpu_da/slurm/submit_all.sh
```

## Results

### Timing

| Step | Wall Time | GPU-hours (4 GPUs) |
|------|-----------|-------------------|
| Spinup (90 days) | 77 min | 5.13 |
| Nature run (14 days) | 38 min | 2.53 |
| Free run (14 days) | 38 min | 2.53 |
| Comparison | <1 min | 0.07 |
| Ensemble DA (10×14) | 102 min | 6.80 |
| **Total** | **~4.3 hrs** | **~17** |

**Note**: Each step includes ~10 min of Julia precompilation/JIT warmup overhead (4 separate srun launches = ~40 min overhead total). A combined script could reduce this significantly.

### Weak Scaling Comparison

| Metric | Single-GPU (1/4 deg) | Multi-GPU (1/8 deg) |
|--------|---------------------|---------------------|
| Spinup wall time | 35 min | 77 min |
| Nature/free run wall | 7 min | 38 min |
| ETKF wall time | 46 min | 102 min |
| Total GPU-hours | 2.4 | 17.0 |
| Points/GPU | 5.6M | 5.6M |
| Throughput (iter/s) | ~10 | ~9.5 |

Wall times are ~2× higher than ideal weak scaling due to:
- Julia precompilation overhead per srun launch (~10 min each)
- MPI communication for halo exchange and observation gathering
- Separate Julia processes (not reusing compiled code across steps)

### ETKF Diagnostics

| Cycle | Day | Forecast RMSE (C) | Analysis RMSE (C) | T Spread (C) |
|-------|-----|-------------------|-------------------|---------------|
| 1 | 1.0 | 0.3547 | 0.3547 | 0.135 |
| 2 | 2.0 | 0.3284 | 0.3285 | 0.022 |
| 5 | 5.0 | 0.2931 | 0.2931 | 0.009 |
| 10 | 10.0 | 0.2585 | 0.2587 | 0.005 |
| 14 | 14.0 | 0.2411 | 0.2413 | 0.008 |

### Analysis

**RMSE improvement**: SST RMSE decreased steadily from 0.355 to 0.241 C over 14 cycles (32% reduction). This is mostly due to the ensemble members converging toward the truth through the forecast model dynamics, since the analysis corrections are tiny.

**Ensemble collapse persists**: T spread drops from 0.135 to 0.008 C despite 1.5 multiplicative inflation. The analysis update is negligible (forecast RMSE ≈ analysis RMSE). With 56K thinned obs and 10 members, the problem is still severely rank-deficient. However, the spread stabilizes rather than going to zero, and even shows slight recovery in later cycles.

**Observation thinning effect**: Thinning by factor 4 (28K obs per field) reduces the observation count by 16× vs the single-GPU setup, but the obs-to-ensemble ratio is still 5,600:1. For effective ETKF, this ratio should be O(1-10).

### Improvements for Next Iteration

1. **Additive inflation** or localization to prevent collapse
2. **Fewer observations**: Thin more aggressively (every 10-20 points)
3. **Larger ensemble**: 50+ members if memory permits
4. **Combined Julia process**: Load code once, run all steps to eliminate ~40 min of precompilation overhead
5. **Variational method**: The differentiable workflow (4D-Var) would avoid ensemble collapse entirely
