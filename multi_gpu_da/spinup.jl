# Multi-GPU spinup at 1/8 degree resolution
#
# Runs a 90-day spinup using 4 GPUs via MPI (Partition 2×2).
# Saves per-rank final state and surface output.
#
# Usage:
#   srun -n 4 julia --project multi_gpu_da/spinup.jl OUTPUT_DIR [DAYS]

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations
using CairoMakie

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

output_dir = ARGS[1]
spinup_days = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 90
mkpath(output_dir)

# ============================================================================
#                           CONFIGURATION
# ============================================================================

resolution = 1/8
Δt = 5minutes
substeps = 60   # 2× more substeps than 1/4 deg (grid is 2× finer)

# Dates
start_date = DateTime(2017, 5, 27)
end_date = start_date + Day(spinup_days)

# MPI decomposition: 2×2 for 4 GPUs
@assert nranks == 4 "Expected 4 MPI ranks, got $nranks"
arch = Distributed(GPU(); partition=Partition(2, 2))

rank == 0 && @info "Multi-GPU Spinup Configuration:"
rank == 0 && @info "  Resolution: $(resolution) degree"
rank == 0 && @info "  MPI ranks: $nranks (Partition 2×2)"
rank == 0 && @info "  Spinup: $spinup_days days ($start_date to $end_date)"
rank == 0 && @info "  Δt = $(Δt), substeps = $substeps"

# ============================================================================
#                           GRID AND MODEL
# ============================================================================

rank == 0 && @info "Creating distributed grid..."
grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)
rank == 0 && @info "  Local grid size: ($Nx, $Ny, $Nz)"

# Rank-aware bathymetry caching (irma_utils version doesn't handle distributed grids)
rank == 0 && @info "Computing bathymetry..."
Nx_local, Ny_local, _ = size(grid)
bathy_file = joinpath(output_dir, "bathymetry_$(Nx_local)x$(Ny_local)_rank$(rank).jld2")
if isfile(bathy_file)
    rank == 0 && @info "  Loading cached bathymetry (per-rank)..."
    bh_data = jldopen(bathy_file)["bottom_height"]
    bottom_height = Field{Center, Center, Nothing}(grid)
    set!(bottom_height, bh_data)
else
    bottom_height = regrid_bathymetry(grid;
        height_above_water=1, minimum_depth=10,
        interpolation_passes=5, major_basins=1)
    jldsave(bathy_file; bottom_height=Array(interior(bottom_height)))
    rank == 0 && @info "  Saved bathymetry to $bathy_file"
end
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
simulation = Simulation(coupled_model; Δt, stop_time=spinup_days * days)

# Output writers
setup_output_writers!(ocean, coupled_model, output_dir;
                       surface_interval=6hour, flux_interval=6hour)

# Progress callback
add_progress_callback!(simulation, start_date; interval=100)

# ============================================================================
#                           RUN
# ============================================================================

rank == 0 && @info "Starting spinup..."
wall_start = time()
run!(simulation)
wall_time = time() - wall_start
rank == 0 && @info "Spinup complete! Wall time: $(round(wall_time / 60, digits=1)) minutes"

# ============================================================================
#                           SAVE STATE (per-rank)
# ============================================================================

rank == 0 && @info "Saving final state (per-rank)..."
model = ocean.model

T = Array(interior(model.tracers.T))
S = Array(interior(model.tracers.S))
e = Array(interior(model.tracers.e))
u = Array(interior(model.velocities.u))
v = Array(interior(model.velocities.v))
w = Array(interior(model.velocities.w))
η = Array(interior(model.free_surface.η))

state_file = joinpath(output_dir, "final_state_rank$(rank).jld2")
jldsave(state_file; T, S, e, u, v, w, η,
        local_size=(Nx, Ny, Nz),
        rank=rank, nranks=nranks)

rank == 0 && @info "Rank $rank saved state to $state_file"
rank == 0 && @info "  Local T range: ($(minimum(T)), $(maximum(T)))"
rank == 0 && @info "  Local S range: ($(minimum(S)), $(maximum(S)))"

MPI.Barrier(comm)
rank == 0 && @info "All ranks saved state."
rank == 0 && @info "Spinup wall time: $(round(wall_time / 60, digits=1)) min"
