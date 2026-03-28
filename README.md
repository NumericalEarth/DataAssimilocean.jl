# DataAssimilocean.jl

Ensemble data assimilation for ocean state estimation using
[ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl) and
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
on NERSC Perlmutter GPUs.

**Team Reactantanigans** — NESAP for Doudna project

## What this does

Solves a *perfect model twin experiment*: a "nature run" generates synthetic ocean truth,
and a Local Ensemble Transform Kalman Filter (LETKF) recovers the ocean state from sparse
Argo-like observations by cycling between parallel ensemble forecast integration and
localized analysis updates.

## Structure

```
experiments/          Shared utilities (grid, model, atmosphere, I/O)
single_gpu_da/        1-GPU DA experiment (1/4°, 400×280×50)
  distributed_da/     Distributed SLURM job pipeline for ensemble DA
    da_utils.jl       ETKF/LETKF algorithm, localization, I/O
    initialize_ensemble.jl
    ensemble_member.jl
    etkf_analysis.jl
    slurm/            Job scripts and orchestration
multi_gpu_da/         4-GPU MPI experiment (1/8°, 800×560×50)
64_gpu_da/            64-GPU MPI experiment (1/32°, 3200×2240×50)
  benchmark.jl        Scaling benchmark (64 GPUs)
  benchmark_8gpu.jl   Scaling benchmark (8 GPUs)
  fix_multinode_env.jl  Workaround for Julia multi-node ENV bug
```

## Quick start

```bash
# On Perlmutter
module load julia/1.11.7

# Single-GPU DA (submits 57 SLURM jobs for 14 cycles, 10 members)
cd single_gpu_da/distributed_da/slurm
bash submit_da.sh 14 10 4

# 8-GPU benchmark
sbatch 64_gpu_da/slurm/benchmark_8gpu.sh

# 64-GPU benchmark
sbatch 64_gpu_da/slurm/benchmark.sh
```

## Progress

See [progress.md](progress.md) for the full development report including
scaling measurements, numerical challenges, and computational cost estimates.

## Key technical notes

- **`JULIA_CUDA_MEMORY_POOL=none`** is required for GPU-aware MPI on Perlmutter
  (CUDA.jl's stream-ordered pool breaks `cuIpcGetMemHandle`)
- **Multi-node Julia bug:** `fix_multinode_env.jl` must be included before `using MPI`
  to work around an infinite loop in `Base.iterate(::EnvDict)` caused by malformed
  environment entries injected by Cray MPICH on remote nodes
- **NERSC account:** m5176_g
