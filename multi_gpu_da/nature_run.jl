# Multi-GPU nature run (14 days from spinup end state)
#
# Loads per-rank spinup state and continues for 14 days.
# This is the "ground truth" for the DA experiment.
#
# Usage:
#   srun -n 4 julia --project multi_gpu_da/nature_run.jl SPINUP_DIR OUTPUT_DIR

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations
using CairoMakie
using Oceananigans.OutputReaders: FieldTimeSeries

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

spinup_dir = ARGS[1]
output_dir = ARGS[2]
mkpath(output_dir)

# ============================================================================
#                           CONFIGURATION
# ============================================================================

resolution = 1/8
Δt = 5minutes
substeps = 60
run_days = 14

start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(run_days)

@assert nranks == 4 "Expected 4 MPI ranks, got $nranks"
arch = Distributed(GPU(); partition=Partition(2, 2))

rank == 0 && @info "Nature Run (Multi-GPU): $run_days days at $(resolution) deg"

# ============================================================================
#                           GRID AND MODEL
# ============================================================================

grid = create_grid(arch, resolution)
# Rank-aware bathymetry (cached from spinup)
Nx, Ny, Nz = size(grid)
bathy_file = joinpath(spinup_dir, "bathymetry_$(Nx)x$(Ny)_rank$(rank).jld2")
if isfile(bathy_file)
    bh_data = jldopen(bathy_file)["bottom_height"]
    bottom_height = Field{Center, Center, Nothing}(grid)
    set!(bottom_height, bh_data)
else
    bottom_height = regrid_bathymetry(grid;
        height_above_water=1, minimum_depth=10,
        interpolation_passes=5, major_basins=1)
    jldsave(bathy_file; bottom_height=Array(interior(bottom_height)))
end
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

ocean = create_ocean_simulation(grid, start_date, end_date;
                                 with_restoring=true, substeps=substeps)

# Load per-rank spinup state
rank == 0 && @info "Loading spinup state..."
state_file = joinpath(spinup_dir, "final_state_rank$(rank).jld2")
state = jldopen(state_file) do f
    (T=f["T"], S=f["S"], e=f["e"], u=f["u"], v=f["v"], w=f["w"], η=f["η"])
end

set!(ocean.model, T=state.T, S=state.S, e=state.e, u=state.u, v=state.v)
copyto!(interior(ocean.model.velocities.w), state.w)
copyto!(interior(ocean.model.free_surface.η), state.η)

rank == 0 && @info "  Rank 0 local T range: ($(minimum(state.T)), $(maximum(state.T)))"

# Atmosphere and coupled model
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=run_days * days)

# Output writers
setup_output_writers!(ocean, coupled_model, output_dir;
                       surface_interval=6hour, flux_interval=6hour)
add_progress_callback!(simulation, start_date; interval=100)

# ============================================================================
#                           RUN
# ============================================================================

rank == 0 && @info "Starting nature run..."
wall_start = time()
run!(simulation)
wall_time = time() - wall_start
rank == 0 && @info "Nature run complete! Wall time: $(round(wall_time / 60, digits=1)) min"

# Save final state (per-rank)
rank == 0 && @info "Saving final state..."
model = ocean.model
T = Array(interior(model.tracers.T))
S = Array(interior(model.tracers.S))
e = Array(interior(model.tracers.e))
u = Array(interior(model.velocities.u))
v = Array(interior(model.velocities.v))
w = Array(interior(model.velocities.w))
η = Array(interior(model.free_surface.η))

jldsave(joinpath(output_dir, "final_state_rank$(rank).jld2");
        T, S, e, u, v, w, η, local_size=(Nx, Ny, Nz))

# Save global surface fields on rank 0 for DA observations
# Use reconstruct_global_field to gather distributed data
rank == 0 && @info "Gathering global surface fields for DA observations..."

using Oceananigans.DistributedComputations: construct_global_array

T_local_surf = Array(interior(model.tracers.T, :, :, Nz))
S_local_surf = Array(interior(model.tracers.S, :, :, Nz))

T_global_surf = Array(construct_global_array(T_local_surf, arch, (Nx, Ny, 1)))
S_global_surf = Array(construct_global_array(S_local_surf, arch, (Nx, Ny, 1)))

if rank == 0
    jldsave(joinpath(output_dir, "final_surface_global.jld2");
            T_surf=T_global_surf, S_surf=S_global_surf)
    @info "Global surface saved."
end

MPI.Barrier(comm)
rank == 0 && @info "Nature run wall time: $(round(wall_time / 60, digits=1)) min"

# ============================================================================
#                           ANIMATION (rank 0 only)
# ============================================================================

if rank == 0
    try
        @info "Generating animation (rank 0)..."
        # Load rank 0 surface fields for a partial animation
        surface_file = joinpath(output_dir, "surface_fields_rank0.jld2")
        if isfile(surface_file)
            Tt = FieldTimeSeries(surface_file, "T")
            times = Tt.times
            Nt = length(times)

            fig = Figure(size=(800, 600))
            ax = Axis(fig[1, 1], title="SST (rank 0 portion)", xlabel="i", ylabel="j")
            T_data = interior(Tt[1], :, :, 1)
            hm = heatmap!(ax, T_data, colormap=:thermal, colorrange=(10, 32))
            Colorbar(fig[1, 2], hm, label="C")

            record(fig, joinpath(output_dir, "nature_run_rank0.mp4"), 1:Nt; framerate=8) do n
                T_data .= interior(Tt[n], :, :, 1)
            end
            @info "Animation saved."
        end
    catch e
        @warn "Animation failed (non-fatal)" exception=(e, catch_backtrace())
    end
end
