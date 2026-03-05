# 8-GPU benchmark at 1/16 degree resolution
#
# Runs 100 time steps with Δt=1 minute for scaling/performance numbers.
# No output writers, no state saving — pure compute benchmark.
#
# Usage:
#   srun -n 8 julia --project 64_gpu_da/benchmark_8gpu.jl

include(joinpath(@__DIR__, "fix_multinode_env.jl"))

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

N_steps = 100
Δt = 1minutes
substeps = 120

# Dates (just need valid range for atmosphere forcing)
start_date = DateTime(2017, 5, 27)
end_date = start_date + Day(30)

resolution = 1/16

# MPI decomposition: 2×4 for 8 GPUs
@assert nranks == 8 "Expected 8 MPI ranks, got $nranks"
arch = Distributed(GPU(); partition=Partition(2, 4))

rank == 0 && @info "8-GPU Benchmark (1/16 deg, $N_steps steps, Δt=$Δt)"
rank == 0 && @info "  MPI ranks: $nranks (Partition 2×4)"

# ============================================================================
#                           GRID AND MODEL
# ============================================================================

rank == 0 && @info "Creating distributed grid..."
grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)
rank == 0 && @info "  Local grid size: ($Nx, $Ny, $Nz)"
rank == 0 && @info "  Global grid size: $(Nx*2) × $(Ny*4) × $Nz"

# Bathymetry
rank == 0 && @info "Computing bathymetry..."
bottom_height = regrid_bathymetry(grid;
    height_above_water=1, minimum_depth=10,
    interpolation_passes=5, major_basins=1)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

rank == 0 && @info "Creating ocean simulation..."
ocean = create_ocean_simulation(grid, start_date, end_date;
                                 with_restoring=true, substeps=substeps)

# Initial conditions from EN4 climatology
rank == 0 && @info "Setting initial conditions from EN4..."
set!(ocean.model,
     T = Metadatum(:temperature; date=start_date, dataset=EN4Monthly()),
     S = Metadatum(:salinity;    date=start_date, dataset=EN4Monthly()))

rank == 0 && @info "Setting up atmosphere..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_iteration=N_steps)

# Progress callback
add_progress_callback!(simulation, start_date; interval=10)

# ============================================================================
#                           BENCHMARK
# ============================================================================

# Warmup: 10 steps
rank == 0 && @info "Warmup (10 steps)..."
simulation.stop_iteration = 10
run!(simulation)
MPI.Barrier(comm)

# Reset and time 100 steps
rank == 0 && @info "Starting benchmark ($N_steps steps)..."
simulation.stop_iteration = 10 + N_steps

wall_start = time()
run!(simulation)
MPI.Barrier(comm)
wall_time = time() - wall_start

# Performance stats
sim_time_seconds = N_steps * 60  # Δt = 1 minute
sim_time_days = sim_time_seconds / 86400
sypd = sim_time_days / (wall_time / 86400)

rank == 0 && @info "=========================================="
rank == 0 && @info "Benchmark complete!"
rank == 0 && @info "  Resolution: 1/16 deg"
rank == 0 && @info "  GPUs:      8"
rank == 0 && @info "  Steps:     $N_steps"
rank == 0 && @info "  Δt:        1 minute"
rank == 0 && @info "  Sim time:  $(round(sim_time_days, digits=4)) days"
rank == 0 && @info "  Wall time: $(round(wall_time, digits=1)) s ($(round(wall_time/60, digits=1)) min)"
rank == 0 && @info "  SYPD:      $(round(sypd, digits=2))"
rank == 0 && @info "  s/step:    $(round(wall_time/N_steps, digits=3))"
rank == 0 && @info "=========================================="
