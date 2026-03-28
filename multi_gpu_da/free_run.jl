# Multi-GPU free run (14 days from perturbed IC, no DA)
#
# Runs from the same subsurface-biased initial conditions as the ensemble mean
# (depth-dependent bias, zero at surface, max at thermocline ~150m).
# No DA corrections — serves as baseline for demonstrating DA value.
#
# Usage:
#   srun -n 4 julia --project multi_gpu_da/free_run.jl SPINUP_DIR OUTPUT_DIR

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: construct_global_array
using CairoMakie
using Random
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: znodes

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

# Perturbation parameters: depth-dependent, subsurface-focused
# Bias only (no noise) — represents ensemble mean trajectory
T_bias_max = 1.5   # C, maximum at thermocline (~150m depth)
S_bias_max = 0.1   # PSU, maximum at thermocline
z_peak = -150.0    # depth of maximum perturbation (m)

@assert nranks == 4 "Expected 4 MPI ranks, got $nranks"
arch = Distributed(GPU(); partition=Partition(2, 2))

rank == 0 && @info "Free Run (Multi-GPU): $run_days days at $(resolution) deg"
rank == 0 && @info "  Perturbation: T bias_max=$(T_bias_max) C at z=$(z_peak) m (subsurface)"

"""
Depth-dependent perturbation profile. Maximum at z_peak, zero at surface,
decaying below. Returns value in [0, 1].
"""
function perturbation_profile(z; z_peak=-150.0)
    z >= 0 && return 0.0
    d = -z / (-z_peak)
    return d * exp(1 - d)
end

# ============================================================================
#                           GRID AND MODEL
# ============================================================================

grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)

# Compute z-centers for depth-dependent perturbation (before ImmersedBoundaryGrid)
z_faces = znodes(grid, Face())
z_centers = [0.5 * (z_faces[k] + z_faces[k+1]) for k in 1:Nz]

# Rank-aware bathymetry (cached from spinup)
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
state_file = joinpath(spinup_dir, "final_state_rank$(rank).jld2")
state = jldopen(state_file) do f
    (T=f["T"], S=f["S"], e=f["e"], u=f["u"], v=f["v"], w=f["w"], η=f["η"])
end

# Add depth-dependent bias to T and S (no noise — represents ensemble mean)
T_perturbed = copy(state.T)
S_perturbed = copy(state.S)
Nz_local = size(T_perturbed, 3)

for k in 1:Nz_local
    α = perturbation_profile(z_centers[k]; z_peak)
    T_perturbed[:, :, k] .+= T_bias_max * α
    S_perturbed[:, :, k] .+= S_bias_max * α
end

set!(ocean.model, T=T_perturbed, S=S_perturbed, e=state.e, u=state.u, v=state.v)
copyto!(interior(ocean.model.velocities.w), state.w)
copyto!(interior(ocean.model.free_surface.η), state.η)

rank == 0 && @info "  Rank 0 perturbed T range: ($(minimum(T_perturbed)), $(maximum(T_perturbed)))"

# Atmosphere and coupled model
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=run_days * days)

setup_output_writers!(ocean, coupled_model, output_dir;
                       surface_interval=6hour, flux_interval=6hour)
add_progress_callback!(simulation, start_date; interval=100)

# ============================================================================
#                           RUN
# ============================================================================

rank == 0 && @info "Starting free run..."
wall_start = time()
run!(simulation)
wall_time = time() - wall_start
rank == 0 && @info "Free run complete! Wall time: $(round(wall_time / 60, digits=1)) min"

# Save final state (per-rank)
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

# Save global surface for comparison
T_local_surf = Array(interior(model.tracers.T, :, :, Nz))
S_local_surf = Array(interior(model.tracers.S, :, :, Nz))
T_global_surf = Array(construct_global_array(T_local_surf, arch, (Nx, Ny, 1)))
S_global_surf = Array(construct_global_array(S_local_surf, arch, (Nx, Ny, 1)))

if rank == 0
    jldsave(joinpath(output_dir, "final_surface_global.jld2");
            T_surf=T_global_surf, S_surf=S_global_surf)
end

MPI.Barrier(comm)
rank == 0 && @info "Free run wall time: $(round(wall_time / 60, digits=1)) min"
