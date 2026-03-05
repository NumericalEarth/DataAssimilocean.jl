# DataAssimilocean.jl — Preliminary Report

## Reactantanigans NESAP for Doudna Project

**Team:** Greg Wagner (PI), Roman Lee, Nestor Demeure (NERSC liaisons)

**Date:** March 5, 2026

---

## 1. Overview

DataAssimilocean.jl implements ensemble data assimilation for ocean state estimation
using ClimaOcean.jl and Oceananigans.jl. The project solves a *perfect model twin experiment*:
a "nature run" generates synthetic truth, and an ensemble Kalman filter recovers the ocean
state from sparse Argo-like observations by cycling between forecast integration and analysis updates.

The computational workflow has three phases:

1. **Spinup** — Initialize the ocean from climatology and integrate to a quasi-equilibrium state
2. **Nature run** — Generate synthetic truth data (full 3D tracer fields + surface observations)
3. **Ensemble DA** — Cycle between parallel member forecast runs and LETKF analysis updates

The domain is the Atlantic Ocean (100°W–0°E, 10°S–60°N) with 50 vertical levels, forced by
JRA55-do reanalysis atmosphere. Resolution scales from 1/4° (single GPU) to 1/32° (64 GPUs)
using weak scaling, maintaining a local grid of ~400×280×50 per GPU.

---

## 2. Development History

### 2.1 Single-GPU Prototype (1 GPU, 1/4°)

The first implementation was a monolithic Julia script (`single_gpu_da/ensemble_da.jl`)
running the complete DA cycle — initialization, forecast, and analysis — sequentially
on a single GPU. This established the baseline workflow:

- **Grid:** 400×280×50 (1/4° Atlantic)
- **Ensemble:** 10 members with depth-dependent perturbations (maximum at thermocline, ~150m)
- **Observations:** 76 Argo-like profiles per cycle, sampling the upper 1000m
- **Analysis window:** 1 day
- **Algorithm:** Global ETKF (Ensemble Transform Kalman Filter)

This prototype revealed the first major numerical challenge: **ensemble collapse**. With only
10 members and O(2500) observations, the ensemble spread collapsed rapidly, making the filter
unable to correct forecast errors after a few cycles.

### 2.2 Multi-GPU Weak Scaling (4 GPUs, 1/8°)

We scaled to 4 GPUs on a single Perlmutter node using MPI with a 2×2 domain decomposition.
The resolution doubled to 1/8° (800×560×50 global grid), maintaining the same ~400×280×50
local grid per GPU — a classic weak-scaling configuration.

Key technical challenges solved:
- **CUDA memory pool incompatibility:** Cray MPICH uses `cuIpcGetMemHandle` for intra-node
  GPU-to-GPU transfers, which is incompatible with CUDA.jl's `cudaMallocAsync` stream-ordered
  pool. Fix: `export JULIA_CUDA_MEMORY_POOL=none`.
- **Distributed I/O:** JLD2 output writers auto-append `_rankN` suffixes; bathymetry caching
  must be per-rank to avoid race conditions.
- **`construct_global_array` API:** Requires 3-tuples `(Nx, Ny, 1)` even for 2D surface data.

### 2.3 64-GPU Scaling (16 nodes, 1/32°)

Scaling to 64 GPUs across 16 nodes (Partition 8×8) at 1/32° resolution introduced a
critical multi-node bug:

**Julia ENV iteration bug:** On multi-node `srun`, Cray MPICH injects environment entries
without `=` separators into the process environment. Julia's `Base.iterate(::EnvDict)` enters
an infinite loop when encountering these malformed entries (the loop index `i` is not
incremented on `continue`). This caused all 64 ranks to hang indefinitely, producing
gigabytes of "malformed environment entry" warnings.

**Fix:** A monkey-patch (`fix_multinode_env.jl`) overrides `Base.iterate(::EnvDict)` to
increment the index on malformed entries. This must be included *before* `using MPI` in
all multi-node scripts. The fix was verified on 4 nodes (grid created in 22 seconds
vs. infinite hang without the patch).

### 2.4 Distributed Ensemble DA Architecture

The monolithic DA script was refactored into a distributed SLURM job pipeline:

```
┌─────────────┐
│  init_job    │  Initialize 10 perturbed ensemble members from spinup state
└──────┬──────┘
       │
       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ member_job   │  │ member_job   │  │ member_job   │  Parallel forecast
│ (members 1-4)│  │ (members 5-8)│  │ (members 9-10)│ (1 GPU each)
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │   afterok       │
                ▼
       ┌────────────────┐
       │ analysis_job   │  LETKF analysis, save updated states
       └────────┬───────┘
                │
                ▼  ... repeat for N cycles ...
```

Jobs are connected by SLURM `afterok` dependency chains, with member states exchanged
through JLD2 checkpoint files on the parallel filesystem (`$SCRATCH`). A single
`submit_da.sh` script submits all jobs upfront (57 jobs for 14 cycles × 10 members).

### 2.5 Numerical Challenges and Solutions

#### Inflation Bug (Critical)

The initial ETKF implementation applied multiplicative inflation to the *full* weight matrix:

```julia
# BUG: inflates both mean shift and perturbations
ensemble[m] = forecast_mean + inflation * Σ W[n,m] * (forecast[n] - forecast_mean)
```

With `inflation = 1.5`, this overshoots the analysis mean by 50%, causing the analysis RMSE
to be *worse* than the forecast (0.585 → 1.679 at cycle 1). The correct approach inflates
perturbations around the *analysis* mean:

```julia
# CORRECT: apply weights first, then inflate perturbations
analysis[m] = forecast_mean + Σ W[n,m] * (forecast[n] - forecast_mean)
analysis_mean = mean(analysis)
analysis[m] = analysis_mean + inflation * (analysis[m] - analysis_mean)
```

#### Observation Error Specification

Initial observation errors (σ_T = 0.1°C, σ_S = 0.01 psu) reflected instrument precision
but ignored representativeness error — the mismatch between a point observation and a
grid-cell average at 1/4° (~28 km). Correct values include representativeness error:
σ_T = 0.5°C, σ_S = 0.05 psu.

#### Overdetermination

With 76 profiles × 31 vertical levels × 2 variables (T, S) = 4,712 observations per cycle,
even the LETKF with 1000 km localization produced ~124 local observations per grid column
vs. a rank-9 ensemble (10 members - 1). This massive overdetermination caused spurious
correlations that degraded the analysis.

**Fix:** Vertical subsampling — observe every 4th level, reducing to ~8 levels and ~32 local
observations per column, better matched to the ensemble rank.

#### LETKF Localization

The global ETKF was replaced with the Local ETKF (LETKF) using Gaspari-Cohn localization
with a 1000 km cutoff radius. For each grid column, only observations from nearby profiles
contribute to the analysis, with their influence tapered by the Gaspari-Cohn correlation
function. This prevents spurious long-range correlations that arise from the rank-deficient
ensemble covariance.

### 2.6 Current DA Results (14 cycles, 10 members)

With all numerical fixes applied, the DA system runs stably through 14 cycles without
blowup. However, the ensemble spread (~0.07–0.09°C) remains too small relative to the
actual forecast errors (~0.8–1.3°C), limiting the filter's ability to correct the state:

| Cycle | Free Run RMSE | Forecast RMSE | Analysis RMSE | Spread |
|-------|--------------|---------------|---------------|--------|
| 1     | 0.988        | 0.585         | 0.602         | 0.068  |
| 2     | 0.974        | 0.618         | 0.738         | 0.066  |
| 5     | 0.934        | 0.806         | 0.821         | 0.070  |
| 10    | 0.876        | 1.017         | 1.045         | 0.087  |
| 14    | 0.843        | 0.955         | 0.957         | 0.094  |

The DA keeps the simulation stable and forecast errors comparable to the free run, but
does not yet achieve significant error reduction. The root cause is insufficient ensemble
size — 10 members cannot adequately sample the error covariance of a 400×280×50 ocean state.
The next step is to scale to **100 ensemble members**, which requires the distributed
architecture described above operating at scale.

---

## 3. Performance and Scaling

### 3.1 Measured Performance

All measurements use the Atlantic domain (100°W–0°E, 10°S–60°N, 50 vertical levels) with
OceanSeaIceModel coupled to JRA55-do atmosphere forcing. Weak scaling maintains a local
grid of ~400×280×50 per GPU.

| GPUs | Nodes | Resolution | Global Grid      | Δt    | Substeps | s/step | SYPD  | Job ID   |
|------|-------|-----------|------------------|-------|----------|--------|-------|----------|
| 1    | shared| 1/4°      | 400×280×50       | 5 min | 30       | 0.061  | 20.5  | 48948679 |
| 4    | 1     | 1/8°      | 800×560×50       | 5 min | 60       | 0.103  | 12.3  | 49011223 |
| 64   | 16    | 1/32°     | 3200×2240×50     | 2.5 min| 240     | TBD*   | TBD   | pending  |

*The 64-GPU run (job 49655390) produced timing of 3.17 s/step but experienced a numerical
instability (NaN) that may inflate this measurement. Clean benchmark jobs are pending
(8-GPU: job 49688114, 64-GPU: job 49687138).

**Platform:** NERSC Perlmutter, NVIDIA A100 GPUs, Cray MPICH (GPU-aware, no NCCL).

**Software:** Julia 1.11.7, Oceananigans.jl, ClimaOcean.jl.

### 3.2 Scaling Analysis

The weak-scaling efficiency from 1 → 4 GPUs is:

- **Ideal s/step scaling:** Substeps double (30 → 60), so expected 2× increase in s/step
- **Measured:** 0.061 → 0.103 s/step = 1.69× increase
- **Efficiency:** 2.0/1.69 = **118%** (super-linear due to cache effects with smaller local domain)

The 4-GPU measurement is intra-node only (NVLink), so it does not reflect inter-node
communication costs. The pending 8-GPU benchmark (2 nodes) will provide the first
inter-node data point using Cray MPICH over libfabric (Slingshot-11).

### 3.3 Extrapolated 64-GPU Performance

From the 4-GPU baseline, the expected 64-GPU scaling is:

- Substeps: 60 → 240 (4× increase from 1/8° → 1/32°)
- Communication: 1 node → 16 nodes (inter-node overhead TBD from 8-GPU benchmark)
- Estimated communication overhead factor: 1.3–2.0×

**Estimated 64-GPU s/step:** 0.103 × 4 × (1.3 to 2.0) = **0.54 to 0.82 s/step**

This will be validated by the pending 8-GPU and 64-GPU benchmarks.

---

## 4. Computational Cost of Ensemble DA at Scale

### 4.1 Target Configuration

- **Resolution:** 1/32° (3200×2240×50)
- **Ensemble size:** 100 members (required for adequate covariance sampling)
- **GPUs per member forecast:** 64 (16 nodes)
- **Analysis window:** 1 day
- **Timestep:** Δt = 2.5 min → 576 steps per day
- **Number of DA cycles:** 14 (14-day experiment)

### 4.2 Cost Per DA Cycle

**Forecast step** (dominates cost):

| Scenario | s/step | Wall time/member | GPU-hrs/member | GPU-hrs/cycle (×100) |
|----------|--------|-----------------|----------------|---------------------|
| Optimistic (0.54 s/step) | 0.54 | 5.2 min | 5.5 | 550 |
| Mid-range (0.82 s/step)  | 0.82 | 7.9 min | 8.4 | 840 |
| Conservative (3.17 s/step)| 3.17 | 30.4 min | 32.4 | 3,240 |

**Analysis step** (LETKF): ~3 min on a single GPU — negligible relative to forecast.

### 4.3 Total Cost (14-cycle experiment)

| Scenario | GPU-hours | Node-hours (4 GPU/node) | Wall time @ 6,400 GPUs |
|----------|-----------|------------------------|----------------------|
| Optimistic  | 7,700     | 1,925                  | ~1.2 hours           |
| Mid-range   | 11,800    | 2,950                  | ~1.8 hours           |
| Conservative| 45,400    | 11,350                 | ~7.1 hours           |

Additional overhead (spinup + nature run): ~500–1,000 GPU-hours.

### 4.4 Scheduling Strategy

With 100 members × 64 GPUs = 6,400 GPUs for fully parallel execution:

- **Fully parallel:** All members forecast simultaneously (6,400 GPUs, minimum wall time)
- **Batched (10×):** 10 concurrent members (640 GPUs), 10 sequential batches per cycle
- **Batched (4×):** 25 concurrent members (1,600 GPUs), 4 sequential batches per cycle

The distributed SLURM architecture already supports arbitrary batching through the
`submit_da.sh` orchestration script.

---

## 5. Next Steps

1. **Complete scaling benchmarks** — Clean 8-GPU and 64-GPU timing to anchor the
   inter-node scaling curve
2. **Increase ensemble size** — Scale from 10 to 100 members using the distributed
   SLURM job architecture
3. **Improve spread maintenance** — Implement RTPS (Relaxation to Prior Spread) or
   additive inflation to maintain adequate ensemble spread with 100 members
4. **Validate DA convergence** — Demonstrate RMSE reduction below the free run baseline
   with the larger ensemble
5. **Differentiable workflows** — Integrate Reactant.jl and Enzyme for adjoint-based
   optimization alongside ensemble methods

---

## Appendix: Software and Reproducibility

- **Repository:** DataAssimilocean.jl (`argo-observations-and-research` branch)
- **Julia version:** 1.11.7
- **Key dependencies:** Oceananigans.jl, ClimaOcean.jl, JLD2.jl, MPI.jl, CUDA.jl
- **Platform:** NERSC Perlmutter (NVIDIA A100 80GB, Slingshot-11, Cray MPICH)
- **Account:** m5176_g
- **Output location:** `$SCRATCH/DataAssimilocean/`
