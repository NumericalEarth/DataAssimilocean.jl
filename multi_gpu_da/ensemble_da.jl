# Ensemble Transform Kalman Filter (ETKF) for ocean state estimation (Multi-GPU)
#
# MPI-aware ETKF with Argo-like observation network:
#   1. Initialize ensemble from perturbed spinup state (per-rank)
#   2. Forecast: run each member forward 1 day (4 GPUs via MPI, members sequential)
#   3. Analyze: extract 3D observations at random profile locations (upper 1000m),
#      gather via MPI, compute ETKF weights
#   4. Update: apply weights to local 3D T and S fields
#   5. Repeat for N analysis cycles
#
# Observation network mimics Argo floats:
#   - ~770 floats in domain (~1 per 3-degree box)
#   - Each profiles every ~10 days → ~76 profiles per daily cycle
#   - Each profile observes all grid levels in the upper 1000m (~31 levels)
#   - Total: ~4,700 obs (T+S) per cycle
#
# Usage:
#   srun -n 4 julia --project multi_gpu_da/ensemble_da.jl \
#       SPINUP_DIR NATURE_DIR FREE_DIR OUTPUT_DIR [N_CYCLES]

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: construct_global_array
using Oceananigans.Grids: znodes
using Random
using Statistics
using LinearAlgebra
using CairoMakie

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

spinup_dir = ARGS[1]
nature_dir = ARGS[2]
free_dir = ARGS[3]
output_dir = ARGS[4]
mkpath(output_dir)

cli_cycles = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing

# ============================================================================
#                           CONFIGURATION
# ============================================================================

resolution = 1/8
Δt = 5minutes
substeps = 60

@assert nranks == 4 "Expected 4 MPI ranks, got $nranks"
arch = Distributed(GPU(); partition=Partition(2, 2))

# DA parameters
N_ens = 10
N_cycles = isnothing(cli_cycles) ? 14 : cli_cycles
analysis_window = 1days
inflation = 1.5

# Observation errors
sigma_T = 0.1    # C
sigma_S = 0.01   # psu

# Argo-like observation network
# Domain: 100W-0E (100 deg) x 10S-60N (70 deg)
# Argo density: ~1 float per 3-degree box → ~33*23 = 759 floats
# Each profiles every ~10 days → ~76 profiles per daily DA cycle
# Each profile: all grid levels in upper 1000m (~31 levels)
n_profiles_per_cycle = 76
obs_depth_limit = -1000.0   # meters: observe upper 1000m only
obs_seed = 12345            # base seed for reproducible profile locations

# Perturbation parameters: depth-dependent, subsurface-focused
T_noise_std = 0.5   # C, ensemble spread (scaled by depth profile)
S_noise_std = 0.05  # PSU, ensemble spread
T_bias_max = 1.5    # C, bias at thermocline (~150m depth)
S_bias_max = 0.1    # PSU, bias at thermocline
z_peak = -150.0     # depth of maximum perturbation (m)

# Dates
start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(N_cycles)

rank == 0 && @info "ETKF Configuration (Multi-GPU, Argo-like obs):"
rank == 0 && @info "  Ensemble size: $N_ens"
rank == 0 && @info "  Analysis cycles: $N_cycles"
rank == 0 && @info "  Inflation: $inflation"
rank == 0 && @info "  Obs error: T=$(sigma_T) C, S=$(sigma_S) psu"
rank == 0 && @info "  Profiles per cycle: $n_profiles_per_cycle"
rank == 0 && @info "  Obs depth limit: $(obs_depth_limit) m"
rank == 0 && @info "  MPI ranks: $nranks"

# ============================================================================
#                           ETKF ALGORITHM
# ============================================================================

"""
    compute_etkf_weights(Y_b, y_obs, R_inv_diag)

Compute ETKF weight matrix following Hunt et al. (2007).
Y_b: (n_obs, N_ens), y_obs: (n_obs,), R_inv_diag: (n_obs,)
Returns W: (N_ens, N_ens)
"""
function compute_etkf_weights(Y_b, y_obs, R_inv_diag)
    N = size(Y_b, 2)
    y_mean = vec(mean(Y_b, dims=2))
    S = Y_b .- y_mean
    d = y_obs .- y_mean

    RinvS = R_inv_diag .* S
    C = S' * RinvS
    invTTt = (N - 1) * I + C

    F = eigen(Symmetric(invTTt))
    inv_sigma = 1.0 ./ F.values
    U = F.vectors

    RinvD = R_inv_diag .* d
    w_mean = U * (inv_sigma .* (U' * (S' * RinvD)))
    W_pert = U * Diagonal(sqrt.((N - 1) .* inv_sigma)) * U'
    W = w_mean * ones(N)' + W_pert

    return W
end

"""
    apply_etkf_update!(ensemble_field, W, inflation)

Apply ETKF weight matrix to update a 3D field for all ensemble members.
"""
function apply_etkf_update!(ensemble_field, W, inflation)
    N = length(ensemble_field)
    forecast = [copy(f) for f in ensemble_field]
    field_mean = mean(forecast)

    for m in 1:N
        ensemble_field[m] .= field_mean
        for n in 1:N
            ensemble_field[m] .+= inflation * W[n, m] .* (forecast[n] .- field_mean)
        end
    end
end

"""
    extract_profile_observations!(obs_vector, field_3d, i_global, j_global, k_obs,
                                   Nx_local, Ny_local, i_offset, j_offset)

Extract observations at Argo-like profile locations from a 3D field.
Only fills entries for profiles local to this MPI rank (others remain zero).
Use MPI.Allreduce(SUM) after calling this to combine across ranks.

Layout: obs_vector[(p-1)*n_levels + l] = field_3d[i_local, j_local, k_obs[l]]
"""
function extract_profile_observations!(obs_vector, field_3d, i_global, j_global, k_obs,
                                        Nx_local, Ny_local, i_offset, j_offset)
    n_profiles = length(i_global)
    n_levels = length(k_obs)
    fill!(obs_vector, 0.0)

    for p in 1:n_profiles
        il = i_global[p] - i_offset
        jl = j_global[p] - j_offset
        if 1 <= il <= Nx_local && 1 <= jl <= Ny_local
            for (li, k) in enumerate(k_obs)
                obs_vector[(p-1)*n_levels + li] = field_3d[il, jl, k]
            end
        end
    end
end

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
#                           SETUP
# ============================================================================

rank == 0 && @info "Creating distributed grid..."
grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)  # LOCAL size

# Compute vertical observation levels (upper 1000m) on the regular grid
# z is the same on all ranks since only x,y are partitioned
# For ExponentialDiscretization(50, -5000, 0; scale=1250):
#   k=1 is deepest (~-4950m), k=Nz=50 is surface (~-4m)
z_faces = znodes(grid, Face())
k_obs = Int[]
for k in 1:Nz
    z_center = 0.5 * (z_faces[k] + z_faces[k+1])
    if z_center >= obs_depth_limit
        push!(k_obs, k)
    end
end
n_obs_levels = length(k_obs)
rank == 0 && @info "  Observation levels: $n_obs_levels (k=$(k_obs[1]):$(k_obs[end]), depth $(obs_depth_limit)m to surface)"

# Compute z-centers for depth-dependent perturbations
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

rank == 0 && @info "  Local grid: ($Nx, $Ny, $Nz)"

# Global grid dimensions and MPI rank offsets
# Partition(2,2): Rx=2, Ry=2
Nx_global = 2 * Nx
Ny_global = 2 * Ny
Ry = 2  # from Partition(2, 2)
i_rank = div(rank, Ry) + 1   # 1-indexed
j_rank = mod(rank, Ry) + 1   # 1-indexed
i_offset = (i_rank - 1) * Nx
j_offset = (j_rank - 1) * Ny

rank == 0 && @info "  Global grid: ($Nx_global, $Ny_global, $Nz)"
rank == 0 && @info "  Rank $rank: i_offset=$i_offset, j_offset=$j_offset"

# Load spinup state (per-rank)
rank == 0 && @info "Loading spinup state..."
state_file = joinpath(spinup_dir, "final_state_rank$(rank).jld2")
spinup_state = jldopen(state_file) do f
    (T=f["T"], S=f["S"], e=f["e"], u=f["u"], v=f["v"], w=f["w"], η=f["η"])
end

# Load nature run 3D tracer snapshots (daily, for ETKF observations)
rank == 0 && @info "Loading nature run 3D tracers..."
nature_3d_file = joinpath(nature_dir, "tracers_3d_rank$(rank).jld2")
nature_3d_fh = jldopen(nature_3d_file)
nature_3d_tkeys = sort(parse.(Int, keys(nature_3d_fh["timeseries/t"])))
nature_3d_times = [nature_3d_fh["timeseries/t/$(k)"] for k in nature_3d_tkeys]
rank == 0 && @info "  Nature 3D snapshots: $(length(nature_3d_times)) times"

# Also load surface FieldTimeSeries for diagnostics (RMSE of SST/SSS)
nature_surface_file = joinpath(nature_dir, "surface_fields_rank$(rank).jld2")
nature_Tt = FieldTimeSeries(nature_surface_file, "T")
nature_St = FieldTimeSeries(nature_surface_file, "S")
nature_surface_times = nature_Tt.times

# Load free run surface fields for baseline comparison
rank == 0 && @info "Loading free run surface fields..."
free_surface_file = joinpath(free_dir, "surface_fields_rank$(rank).jld2")
free_Tt = FieldTimeSeries(free_surface_file, "T")
free_St = FieldTimeSeries(free_surface_file, "S")
free_surface_times = free_Tt.times

# ============================================================================
#                           INITIALIZE ENSEMBLE
# ============================================================================

rank == 0 && @info "Initializing $N_ens ensemble members..."
ensemble_T = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_S = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_e = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_u = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_v = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_w = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_η = Vector{Any}(undef, N_ens)

Nz_local = size(spinup_state.T, 3)

for i in 1:N_ens
    rng = MersenneTwister(42 + i + rank * 1000)

    T_i = copy(spinup_state.T)
    S_i = copy(spinup_state.S)

    for k in 1:Nz_local
        α = perturbation_profile(z_centers[k]; z_peak)
        T_i[:, :, k] .+= α .* (T_noise_std .* randn(rng, Nx, Ny) .+ T_bias_max)
        S_i[:, :, k] .+= α .* (S_noise_std .* randn(rng, Nx, Ny) .+ S_bias_max)
    end

    ensemble_T[i] = T_i
    ensemble_S[i] = S_i
    ensemble_e[i] = copy(spinup_state.e)
    ensemble_u[i] = copy(spinup_state.u)
    ensemble_v[i] = copy(spinup_state.v)
    ensemble_w[i] = copy(spinup_state.w)
    ensemble_η[i] = copy(spinup_state.η)
end

rank == 0 && @info "Ensemble initialized."

# ============================================================================
#                           CREATE MODEL
# ============================================================================

rank == 0 && @info "Creating ocean model..."
ocean = create_ocean_simulation(grid, start_date, end_date;
                                 with_restoring=true, substeps=substeps)

atmosphere, radiation = create_atmosphere(arch, start_date, end_date)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=N_cycles * analysis_window)

# Remove output writers for ensemble forecast runs
empty!(ocean.output_writers)

# Warm-up run
rank == 0 && @info "Compiling model (warm-up run)..."
set!(ocean.model, T=ensemble_T[1], S=ensemble_S[1], e=ensemble_e[1],
     u=ensemble_u[1], v=ensemble_v[1])
copyto!(interior(ocean.model.velocities.w), ensemble_w[1])
copyto!(interior(ocean.model.free_surface.η), ensemble_η[1])

simulation.model.clock.time = 0.0
simulation.model.clock.iteration = 0
ocean.model.clock.time = 0.0
ocean.model.clock.iteration = 0
simulation.stop_time = Float64(Δt)
run!(simulation)

rank == 0 && @info "Warm-up complete. Starting DA loop."

# ============================================================================
#                           DA LOOP
# ============================================================================

# Observation counts
n_obs_per_field = n_profiles_per_cycle * n_obs_levels
n_obs_total = 2 * n_obs_per_field  # T + S

rank == 0 && @info "Observation network:"
rank == 0 && @info "  Profiles/cycle: $n_profiles_per_cycle"
rank == 0 && @info "  Levels/profile: $n_obs_levels"
rank == 0 && @info "  Obs per field: $n_obs_per_field"
rank == 0 && @info "  Total obs (T+S): $n_obs_total"
rank == 0 && @info "  Obs-to-ensemble ratio: $(n_obs_total / N_ens)"

# R inverse (observation error covariance inverse, diagonal)
R_inv = vcat(fill(1.0 / sigma_T^2, n_obs_per_field),
             fill(1.0 / sigma_S^2, n_obs_per_field))

# Diagnostics storage
rmse_T_before = Float64[]
rmse_T_after = Float64[]
rmse_T_free = Float64[]
rmse_S_before = Float64[]
rmse_S_after = Float64[]
rmse_S_free = Float64[]
spread_T = Float64[]
spread_S = Float64[]
analysis_days = Float64[]
n_valid_obs_log = Int[]

# Pre-allocate observation vectors
obs_T_local = zeros(n_obs_per_field)
obs_S_local = zeros(n_obs_per_field)
Y_b_local = zeros(n_obs_total, N_ens)

wall_start = time()

for cycle in 1:N_cycles
    cycle_start = time()
    t_start = (cycle - 1) * Float64(analysis_window)
    t_end = cycle * Float64(analysis_window)
    day_num = t_end / (24 * 3600)

    rank == 0 && @info "=== DA Cycle $cycle/$N_cycles (day $day_num) ==="

    # --- Forecast: run each member forward ---
    for i in 1:N_ens
        set!(ocean.model, T=ensemble_T[i], S=ensemble_S[i], e=ensemble_e[i],
             u=ensemble_u[i], v=ensemble_v[i])
        copyto!(interior(ocean.model.velocities.w), ensemble_w[i])
        copyto!(interior(ocean.model.free_surface.η), ensemble_η[i])

        simulation.model.clock.time = t_start
        simulation.model.clock.iteration = 0
        ocean.model.clock.time = t_start
        ocean.model.clock.iteration = 0
        simulation.stop_time = t_end
        run!(simulation)

        ensemble_T[i] = Array(interior(ocean.model.tracers.T))
        ensemble_S[i] = Array(interior(ocean.model.tracers.S))
        ensemble_e[i] = Array(interior(ocean.model.tracers.e))
        ensemble_u[i] = Array(interior(ocean.model.velocities.u))
        ensemble_v[i] = Array(interior(ocean.model.velocities.v))
        ensemble_w[i] = Array(interior(ocean.model.velocities.w))
        ensemble_η[i] = Array(interior(ocean.model.free_surface.η))
    end

    forecast_time = time() - cycle_start

    # --- Generate Argo-like profile locations for this cycle ---
    profile_rng = MersenneTwister(obs_seed + cycle)
    i_profiles = rand(profile_rng, 1:Nx_global, n_profiles_per_cycle)
    j_profiles = rand(profile_rng, 1:Ny_global, n_profiles_per_cycle)

    # --- Load nature run 3D state at analysis time ---
    _, obs_3d_idx = findmin(abs.(nature_3d_times .- t_end))
    tkey = nature_3d_tkeys[obs_3d_idx]
    nature_T_3d = nature_3d_fh["timeseries/T/$(tkey)"]
    nature_S_3d = nature_3d_fh["timeseries/S/$(tkey)"]

    # --- Extract truth observations at profile locations ---
    extract_profile_observations!(obs_T_local, nature_T_3d, i_profiles, j_profiles,
                                   k_obs, Nx, Ny, i_offset, j_offset)
    extract_profile_observations!(obs_S_local, nature_S_3d, i_profiles, j_profiles,
                                   k_obs, Nx, Ny, i_offset, j_offset)

    # Combine across ranks (each rank contributes its local profiles, others are zero)
    obs_T_truth = MPI.Allreduce(obs_T_local, MPI.SUM, comm)
    obs_S_truth = MPI.Allreduce(obs_S_local, MPI.SUM, comm)
    y_obs = vcat(obs_T_truth, obs_S_truth)

    # --- Filter out land observations ---
    # Land cells have T=0 in the nature run. Check the surface level of each profile.
    # A profile is "ocean" if the truth T at the surface level (k_obs[end]) is > 1.
    valid_mask = trues(n_obs_total)
    for p in 1:n_profiles_per_cycle
        surf_idx = (p - 1) * n_obs_levels + n_obs_levels  # last level = surface
        if abs(obs_T_truth[surf_idx]) < 1.0
            # This profile is on land - mark all its levels as invalid
            for l in 1:n_obs_levels
                valid_mask[(p - 1) * n_obs_levels + l] = false                   # T obs
                valid_mask[n_obs_per_field + (p - 1) * n_obs_levels + l] = false  # S obs
            end
        end
    end
    n_valid = sum(valid_mask)

    # --- Compute surface forecast diagnostics ---
    # (Uses surface FieldTimeSeries, same metric as before for comparison)
    _, surf_idx = findmin(abs.(nature_surface_times .- t_end))
    T_truth_surf = interior(nature_Tt[surf_idx], :, :, 1)
    S_truth_surf = interior(nature_St[surf_idx], :, :, 1)

    T_ens_surf = [e[:, :, Nz_local] for e in ensemble_T]
    S_ens_surf = [e[:, :, Nz_local] for e in ensemble_S]
    T_mean_surf = mean(T_ens_surf)
    S_mean_surf = mean(S_ens_surf)

    local_T_sse = sum((T_mean_surf .- T_truth_surf) .^ 2)
    local_S_sse = sum((S_mean_surf .- S_truth_surf) .^ 2)
    local_n = length(T_mean_surf)

    global_T_sse = MPI.Allreduce(local_T_sse, MPI.SUM, comm)
    global_S_sse = MPI.Allreduce(local_S_sse, MPI.SUM, comm)
    global_n = MPI.Allreduce(Float64(local_n), MPI.SUM, comm)

    rmse_T_fc = sqrt(global_T_sse / global_n)
    rmse_S_fc = sqrt(global_S_sse / global_n)

    # Ensemble spread
    local_T_spread_sse = sum(mean((t .- T_mean_surf).^2 for t in T_ens_surf))
    local_S_spread_sse = sum(mean((s .- S_mean_surf).^2 for s in S_ens_surf))
    global_T_spread_sse = MPI.Allreduce(local_T_spread_sse, MPI.SUM, comm)
    global_S_spread_sse = MPI.Allreduce(local_S_spread_sse, MPI.SUM, comm)
    T_sprd = sqrt(global_T_spread_sse / global_n)
    S_sprd = sqrt(global_S_spread_sse / global_n)

    # Free run baseline RMSE
    _, free_idx = findmin(abs.(free_surface_times .- t_end))
    T_free_surf = interior(free_Tt[free_idx], :, :, 1)
    S_free_surf = interior(free_St[free_idx], :, :, 1)
    local_T_sse_free = sum((T_free_surf .- T_truth_surf) .^ 2)
    local_S_sse_free = sum((S_free_surf .- S_truth_surf) .^ 2)
    global_T_sse_free = MPI.Allreduce(local_T_sse_free, MPI.SUM, comm)
    global_S_sse_free = MPI.Allreduce(local_S_sse_free, MPI.SUM, comm)
    rmse_T_free_val = sqrt(global_T_sse_free / global_n)
    rmse_S_free_val = sqrt(global_S_sse_free / global_n)

    push!(rmse_T_before, rmse_T_fc)
    push!(rmse_S_before, rmse_S_fc)
    push!(rmse_T_free, rmse_T_free_val)
    push!(rmse_S_free, rmse_S_free_val)
    push!(spread_T, T_sprd)
    push!(spread_S, S_sprd)
    push!(analysis_days, day_num)
    push!(n_valid_obs_log, n_valid)

    rank == 0 && @info "  Free run SST RMSE=$(round(rmse_T_free_val, digits=4)) C"
    rank == 0 && @info "  DA forecast SST RMSE=$(round(rmse_T_fc, digits=4)) C, spread=$(round(T_sprd, digits=4)) C"
    rank == 0 && @info "  Valid obs: $n_valid / $n_obs_total ($(n_profiles_per_cycle) profiles)"

    # --- ETKF Analysis ---
    if n_valid > 0
        # Build ensemble observation matrix (all members, all obs)
        fill!(Y_b_local, 0.0)
        for i in 1:N_ens
            extract_profile_observations!(@view(Y_b_local[1:n_obs_per_field, i]),
                                           ensemble_T[i], i_profiles, j_profiles,
                                           k_obs, Nx, Ny, i_offset, j_offset)
            extract_profile_observations!(@view(Y_b_local[n_obs_per_field+1:end, i]),
                                           ensemble_S[i], i_profiles, j_profiles,
                                           k_obs, Nx, Ny, i_offset, j_offset)
        end

        # Single Allreduce for the full ensemble-obs matrix
        Y_b = MPI.Allreduce(Y_b_local, MPI.SUM, comm)

        # Apply land mask: only use valid (ocean) observations
        y_obs_valid = y_obs[valid_mask]
        Y_b_valid = Y_b[valid_mask, :]
        R_inv_valid = R_inv[valid_mask]

        # Compute weight matrix (deterministic, same on all ranks)
        W = compute_etkf_weights(Y_b_valid, y_obs_valid, R_inv_valid)

        # Apply weights to local T and S (full 3D)
        apply_etkf_update!(ensemble_T, W, inflation)
        apply_etkf_update!(ensemble_S, W, inflation)
    else
        rank == 0 && @warn "  No valid observations this cycle - skipping analysis"
    end

    # --- Analysis diagnostics ---
    T_mean_a = mean([e[:, :, Nz_local] for e in ensemble_T])
    S_mean_a = mean([e[:, :, Nz_local] for e in ensemble_S])
    local_T_sse_a = sum((T_mean_a .- T_truth_surf) .^ 2)
    local_S_sse_a = sum((S_mean_a .- S_truth_surf) .^ 2)
    global_T_sse_a = MPI.Allreduce(local_T_sse_a, MPI.SUM, comm)
    global_S_sse_a = MPI.Allreduce(local_S_sse_a, MPI.SUM, comm)
    rmse_T_an = sqrt(global_T_sse_a / global_n)
    rmse_S_an = sqrt(global_S_sse_a / global_n)

    push!(rmse_T_after, rmse_T_an)
    push!(rmse_S_after, rmse_S_an)

    cycle_time = time() - cycle_start
    rank == 0 && @info "  Analysis: SST RMSE=$(round(rmse_T_an, digits=4)) C ($(round(forecast_time, digits=1))s forecast, $(round(cycle_time, digits=1))s total)"
end

close(nature_3d_fh)

total_wall = time() - wall_start
rank == 0 && @info "DA complete! Total wall time: $(round(total_wall / 60, digits=1)) minutes"

# ============================================================================
#                           SAVE DIAGNOSTICS (rank 0)
# ============================================================================

if rank == 0
    @info "Saving diagnostics..."
    jldsave(joinpath(output_dir, "ensemble_diagnostics.jld2");
            rmse_T_before, rmse_T_after, rmse_T_free,
            rmse_S_before, rmse_S_after, rmse_S_free,
            spread_T, spread_S,
            analysis_days, n_valid_obs_log,
            N_ens, inflation, sigma_T, sigma_S,
            n_profiles_per_cycle, obs_depth_limit, n_obs_levels,
            total_wall_seconds=total_wall)
end

# Gather final ensemble mean surface to rank 0
T_mean_surf_local = mean(ensemble_T)[:, :, Nz_local]
S_mean_surf_local = mean(ensemble_S)[:, :, Nz_local]
T_mean_surf_global = Array(construct_global_array(T_mean_surf_local, arch, (Nx, Ny, 1)))
S_mean_surf_global = Array(construct_global_array(S_mean_surf_local, arch, (Nx, Ny, 1)))

if rank == 0
    jldsave(joinpath(output_dir, "da_final_mean_surface.jld2");
            T_surf=T_mean_surf_global, S_surf=S_mean_surf_global)
end

# ============================================================================
#                           VISUALIZATION (rank 0 only)
# ============================================================================

if rank == 0
    @info "Generating DA diagnostic plots..."

    try
        # RMSE comparison plot
        fig = Figure(size=(1000, 500))

        ax1 = Axis(fig[1, 1], xlabel="Day", ylabel="RMSE (C)",
                    title="SST RMSE: DA vs Free Run")
        lines!(ax1, analysis_days, rmse_T_free, linewidth=2, color=:red,
               label="Free run (no DA)")
        lines!(ax1, analysis_days, rmse_T_before, linewidth=2, color=:orange,
               label="DA forecast", linestyle=:dash)
        lines!(ax1, analysis_days, rmse_T_after, linewidth=2, color=:green,
               label="DA analysis")
        axislegend(ax1, position=:lt)

        ax2 = Axis(fig[1, 2], xlabel="Day", ylabel="Spread / RMSE (C)",
                    title="Ensemble Spread vs RMSE")
        lines!(ax2, analysis_days, rmse_T_free, linewidth=2, color=:red, label="Free run RMSE")
        lines!(ax2, analysis_days, rmse_T_before, linewidth=2, color=:orange, label="Forecast RMSE")
        lines!(ax2, analysis_days, spread_T, linewidth=2, color=:purple, label="Spread")
        axislegend(ax2, position=:lt)

        save(joinpath(output_dir, "rmse_comparison.png"), fig, px_per_unit=2)
        @info "RMSE comparison plot saved"

        # Ensemble spread plot
        fig3 = Figure(size=(800, 400))
        ax = Axis(fig3[1, 1], xlabel="Day", ylabel="Value",
                  title="Ensemble Diagnostics (Argo-like, $(n_profiles_per_cycle) profiles x $(n_obs_levels) levels)")
        lines!(ax, analysis_days, spread_T, linewidth=2, color=:purple, label="T spread (C)")
        lines!(ax, analysis_days, 10 .* spread_S, linewidth=2, color=:blue,
               label="S spread (x10, psu)")
        lines!(ax, analysis_days, rmse_T_after, linewidth=2, color=:green,
               label="T analysis RMSE (C)", linestyle=:dash)
        axislegend(ax)

        save(joinpath(output_dir, "ensemble_spread.png"), fig3, px_per_unit=2)
        @info "Ensemble spread plot saved"
    catch e
        @warn "Visualization failed (non-fatal)" exception=(e, catch_backtrace())
    end
end

MPI.Barrier(comm)
rank == 0 && @info "Ensemble DA experiment complete!"
rank == 0 && @info "  Final free run SST RMSE: $(round(rmse_T_free[end], digits=4)) C"
rank == 0 && @info "  Final forecast SST RMSE: $(round(rmse_T_before[end], digits=4)) C"
rank == 0 && @info "  Final analysis SST RMSE: $(round(rmse_T_after[end], digits=4)) C"
rank == 0 && @info "  Output: $output_dir"
