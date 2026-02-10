# DataAssimilocean.jl

## Team: Reactantanigans

**NERSC team liaison:** Roman Lee, Nestor Demeure

## Science Problem

Oceananigans.jl and ClimaOcean.jl are Julia packages for simulating ocean and sea ice dynamics at all scales in spherical and limited-area rectangular domains, coupled to prescribed or prognostic atmosphere models. They use finite volume numerical methods with high-order advection schemes.

Our aim is to enable performant differentiable workflows at scale to support data assimilation, model calibration, and development of hybrid physics/AI model configurations. We solve a **"perfect model state estimation problem"** using a variational method.

## Workflow

### Step 1: Nature Run

Perform a "nature run" simulation and save data to disk (the entire initial state, plus copies of the state at different points in time).

- At high resolution: requires **100-1000 GPUs** and **O(100 GB - 1 TB)** storage.

### Step 2: State Recovery via Differentiable Optimization

Recover the state of the nature run independently by optimizing a cost function with gradient-based methods involving a forward simulation.

- The cost function must be evaluable in parallel from saved data (data does not fit in the memory of a single node).
- Differentiating the cost function (which involves a forward simulation) requires orchestrating **MPI, Reactant, Enzyme, Oceananigans, and ClimaOcean**.

## Software Stack

- **Language:** Julia
- **Core dependencies:** ClimaOcean.jl, Oceananigans.jl, Reactant.jl, Enzyme, XLA, LLVM/MLIR, MPI, Julia

## I/O Patterns

- A ~1/20 degree ocean simulation uses ~1 TB of GPU memory and is checkpointed in `.jld2` (HDF5-compatible) files, with each rank writing its own restart file.
- Daily output: ~100 GB (five 20 GB 3D fields). At ~1 SYPD this translates to ~100 GB written every ~4 minutes of wall time.
- Forcing data (~3 GB per GPU) is reloaded every 15 simulated days (about once per hour of wall time). This read pattern does not scale with the number of ranks.
- Overall: large, frequent, structured writes and periodic moderate reads.

## Performance Track

Improved algorithms: enabling automatic differentiation within an Oceananigans simulation allows state estimation and data assimilation far more efficiently than ensemble-based methods or manually implemented adjoints.

## Capabilities Track

- Seamless end-to-end differentiable AI training workflow
- Potentially AI-surrogate models
- Portable containerized programming environment (containers, cloud use)

## Skeleton / Mini-Application

The representative skeleton is the full workflow: running a large nature run simulation, then reconstructing its state via a parallelizable, differentiable cost-function-driven optimization. Next steps:

1. Specify a concrete science problem for the nature run.
2. Define a performance FOM (Figure of Merit).
