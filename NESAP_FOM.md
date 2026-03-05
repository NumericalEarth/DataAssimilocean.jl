# NESAP FOM Measurements

## Figure of Merit: Simulated Years Per Day (SYPD)

Ocean simulation throughput measured as simulated years per wall-clock day.
SYPD = (simulated_time_days / wall_time_days)

## Problem Definition

Atlantic ocean simulation (100°W–0°E, 10°S–60°N) with 50 vertical levels,
forced by JRA55 atmosphere. Uses OceanSeaIceModel with split explicit free surface.
Weak scaling: same local grid size per GPU (~400×280×50), increasing total resolution.

## Existing Timing Data (from production runs)

### 1 GPU — 1/4° resolution (400×280×50)
- Job: 48948679 (single_gpu_da pipeline)
- Wall per 100 steps: ~6.1 seconds (after warmup, Iter 1000-2000 range)
- Δt = 5 minutes, substeps = 30
- **s/step = 0.061**
- **SYPD = (5 min / 0.061 s) × (1 day / 1440 min) × (365 days / 1 day) ≈ 20.5 SYPD**

### 4 GPU — 1/8° resolution (800×560×50, Partition 2×2)
- Job: 49011223 (multi_gpu_da pipeline), 1 node
- Wall per 100 steps: ~10.34 seconds (Iter 1000+ range)
- Δt = 5 minutes, substeps = 60
- **s/step = 0.1034**
- **SYPD = (5 min / 0.1034 s) × (1/1440) × 365 ≈ 12.3 SYPD**

### 64 GPU — 1/32° resolution (3200×2240×50, Partition 8×8)
- Job: 49655390 ran but experienced NaN blowup — **timing unreliable**
  (NaN propagation distorts per-step compute time)
- Target Δt = 1 minute (4× finer grid than 1/8° requires ~4× smaller timestep)
- Substeps = 240
- Clean benchmark pending (job 49687138)

### Pending benchmarks
- 8 GPU — 1/16° (job 49688114, regular queue, Δt=1min)
- 64 GPU — 1/32° (job 49687138, regular queue, 100 steps with Δt=1min)

## NESAP Form Fields

- **Project:** Reactantanigans (DataAssimilocean.jl)
- **FOM Description:** Simulated Years Per Day (SYPD) — ocean simulation throughput
- **Problem Definition:** Atlantic ocean (100W-0E, 10S-60N), 50 vertical levels, OceanSeaIceModel with JRA55 atmosphere forcing. Weak scaling from 1/4° to 1/32° resolution.
- **Software:** ClimaOcean.jl + Oceananigans.jl, Julia 1.11.7
- **Platform:** Perlmutter GPU nodes (NVIDIA A100)
- **Reproducible environment:** Julia Project.toml with pinned dependencies
