# 64-GPU free run (14 days from perturbed IC, no DA)
#
# Runs from biased+gradient-perturbed initial conditions.
# No DA corrections — serves as baseline for demonstrating DA value.
#
# Perturbation: depth-dependent bias + gradient component
#   T += T_bias * p(z) + T_grad * g(z)
#   S += S_bias * p(z) + S_grad * g(z)
#
# where p(z) peaks at z_peak (subsurface bias) and g(z) changes sign at z_peak
# (weakening stratification above, strengthening below).
#
# Usage:
#   srun -n 64 julia --project 64_gpu_da/free_run.jl SPINUP_DIR OUTPUT_DIR

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: construct_global_array
using CairoMakie
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

resolution = 1/32
Δt = 5minutes
substeps = 240
run_days = 14

start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(run_days)

# Perturbation parameters: bias + gradient components
# Bias component: subsurface warming/salinification peaking at z_peak
T_bias = 1.5    # C, maximum at z_peak
S_bias = 0.1    # psu, maximum at z_peak
# Gradient component: weakens stratification (warms above z_peak, cools below)
T_grad = 0.8    # C
S_grad = -0.05  # psu
z_peak = -150.0 # depth of maximum bias perturbation (m)

@assert nranks == 64 "Expected 64 MPI ranks, got $nranks"
arch = Distributed(GPU(); partition=Partition(8, 8))

rank == 0 && @info "Free Run (64-GPU): $run_days days at $(resolution) deg"
rank == 0 && @info "  Perturbation: T_bias=$(T_bias) C, T_grad=$(T_grad) C at z=$(z_peak) m"

"""
Depth-dependent bias profile. Maximum at z_peak, zero at surface, decaying below.
    p(z) = d * exp(1 - d)  where d = -z / (-z_peak)
Returns value in [0, 1].
"""
function bias_profile(z; z_peak=-150.0)
    z >= 0 && return 0.0
    d = -z / (-z_peak)
    return d * exp(1 - d)
end

"""
Depth-dependent gradient profile. Changes sign at z_peak: positive above, negative below.
Weakens stratification when multiplied by a positive temperature perturbation.
    g(z) = (1 - d) * exp(-d)  where d = -z / (-z_peak)
"""
function gradient_profile(z; z_peak=-150.0)
    z >= 0 && return 1.0
    d = -z / (-z_peak)
    return (1 - d) * exp(-d)
end

# ============================================================================
#                           GRID AND MODEL
# ============================================================================

grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)

# Compute z-centers for depth-dependent perturbation (before ImmersedBoundaryGrid)
# Use Array() to avoid GPU scalar indexing
z_faces = Array(znodes(grid, Face()))
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

# Add depth-dependent bias + gradient perturbation to T and S
T_perturbed = copy(state.T)
S_perturbed = copy(state.S)
Nz_local = size(T_perturbed, 3)

for k in 1:Nz_local
    z = z_centers[k]
    p = bias_profile(z; z_peak)
    g = gradient_profile(z; z_peak)
    T_perturbed[:, :, k] .+= T_bias * p + T_grad * g
    S_perturbed[:, :, k] .+= S_bias * p + S_grad * g
end

set!(ocean.model, T=T_perturbed, S=S_perturbed, e=state.e, u=state.u, v=state.v)
copyto!(interior(ocean.model.velocities.w), state.w)
copyto!(interior(ocean.model.free_surface.η), state.η)

rank == 0 && @info "  Rank 0 perturbed T range: ($(minimum(T_perturbed)), $(maximum(T_perturbed)))"
rank == 0 && @info "  Rank 0 perturbed S range: ($(minimum(S_perturbed)), $(maximum(S_perturbed)))"

# Atmosphere and coupled model
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=run_days * days)

setup_output_writers!(ocean, coupled_model, output_dir;
                       surface_interval=6hour, flux_interval=6hour)

# Daily 3D tracer output for comparison with nature run
ocean.output_writers[:tracers_3d] = JLD2Writer(ocean.model,
    (; T=ocean.model.tracers.T, S=ocean.model.tracers.S);
    schedule = TimeInterval(1days),
    filename = joinpath(output_dir, "tracers_3d"),
    overwrite_existing = true)

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
