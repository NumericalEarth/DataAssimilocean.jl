# Ensemble Transform Kalman Filter (ETKF) for ocean state estimation (Multi-GPU)
#
# MPI-aware ETKF with observation thinning:
#   1. Initialize ensemble from perturbed spinup state (per-rank)
#   2. Forecast: run each member forward 1 day (4 GPUs via MPI, members sequential)
#   3. Analyze: gather thinned surface obs, compute ETKF weights
#   4. Update: apply weights to local 3D T and S fields
#   5. Repeat for 14 analysis cycles
#
# Usage:
#   srun -n 4 julia --project multi_gpu_da/ensemble_da.jl \
#       SPINUP_DIR NATURE_DIR FREE_DIR OUTPUT_DIR [N_CYCLES]

using MPI
MPI.Init()

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: construct_global_array
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
inflation = 1.5             # Stronger inflation (was 1.05 in single-GPU)

# Observation errors
sigma_T = 0.1
sigma_S = 0.01

# Observation thinning: take every thin_factor-th point in each direction
# At 1/8 deg: local 400×280, thinned by 4 → 100×70 = 7K per rank, 28K global per field
thin_factor = 4

# Perturbation parameters
T_noise_std = 0.5
S_noise_std = 0.05
T_bias = 0.3
S_bias = 0.02
noise_depth_levels = 20

# Dates
start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(N_cycles)

rank == 0 && @info "ETKF Configuration (Multi-GPU):"
rank == 0 && @info "  Ensemble size: $N_ens"
rank == 0 && @info "  Analysis cycles: $N_cycles"
rank == 0 && @info "  Inflation: $inflation"
rank == 0 && @info "  Obs error: T=$(sigma_T) C, S=$(sigma_S) psu"
rank == 0 && @info "  Obs thinning factor: $thin_factor"
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
    thin_surface(field_2d, thin_factor)

Take every thin_factor-th point from a 2D array.
Returns a smaller array.
"""
function thin_surface(field_2d, thin_factor)
    return field_2d[1:thin_factor:end, 1:thin_factor:end]
end

# ============================================================================
#                           SETUP
# ============================================================================

rank == 0 && @info "Creating distributed grid..."
grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)  # LOCAL size

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

# Compute thinned dimensions
Nx_thin = length(1:thin_factor:Nx)
Ny_thin = length(1:thin_factor:Ny)
rank == 0 && @info "  Thinned local surface: ($Nx_thin, $Ny_thin)"

# Load spinup state (per-rank)
rank == 0 && @info "Loading spinup state..."
state_file = joinpath(spinup_dir, "final_state_rank$(rank).jld2")
spinup_state = jldopen(state_file) do f
    (T=f["T"], S=f["S"], e=f["e"], u=f["u"], v=f["v"], w=f["w"], η=f["η"])
end

# Load nature run final surface for truth observations
# (Global surface saved by nature_run.jl)
rank == 0 && @info "Loading nature run observations..."

# For observations, we load the per-rank nature run surface fields at each cycle.
# Since nature_run.jl saves per-rank surface_fields, we load rank-specific files.
# But we also saved final_surface_global.jld2 - for simplicity, use the
# per-rank JLD2Writer output which has time series.
nature_surface_file = joinpath(nature_dir, "surface_fields_rank$(rank).jld2")
nature_Tt = FieldTimeSeries(nature_surface_file, "T")
nature_St = FieldTimeSeries(nature_surface_file, "S")
nature_times = nature_Tt.times

rank == 0 && @info "  Nature run times: $(length(nature_times)) snapshots"

# Load free run RMSE for comparison
free_rmse_T = nothing
free_rmse_days = nothing
free_rmse_file = joinpath(dirname(free_dir), "plots", "rmse_data.jld2")
if rank == 0 && isfile(free_rmse_file)
    @info "Loading free run RMSE data..."
    free_data = jldopen(free_rmse_file)
    free_rmse_T = free_data["rmse_T_final"]
    close(free_data)
end

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
k_start = max(1, Nz_local - noise_depth_levels + 1)

for i in 1:N_ens
    # Seed depends on member AND rank for unique perturbation per rank
    rng = MersenneTwister(42 + i + rank * 1000)

    T_i = copy(spinup_state.T)
    S_i = copy(spinup_state.S)

    for k in k_start:Nz_local
        T_i[:, :, k] .+= T_noise_std .* randn(rng, Nx, Ny) .+ T_bias
        S_i[:, :, k] .+= S_noise_std .* randn(rng, Nx, Ny) .+ S_bias
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

# Diagnostics storage (all ranks compute, rank 0 saves)
rmse_T_before = Float64[]
rmse_T_after = Float64[]
rmse_S_before = Float64[]
rmse_S_after = Float64[]
spread_T = Float64[]
spread_S = Float64[]
analysis_days = Float64[]

# Thinned observation error inverse
# Total thinned obs = nranks * Nx_thin * Ny_thin per field
n_T_obs_local = Nx_thin * Ny_thin
n_S_obs_local = Nx_thin * Ny_thin
n_obs_local = n_T_obs_local + n_S_obs_local
n_T_obs_global = nranks * n_T_obs_local
n_S_obs_global = nranks * n_S_obs_local
n_obs_global = n_T_obs_global + n_S_obs_global

R_inv_global = vcat(fill(1.0 / sigma_T^2, n_T_obs_global),
                     fill(1.0 / sigma_S^2, n_S_obs_global))

rank == 0 && @info "Thinned obs per field: $n_T_obs_global ($(n_T_obs_local) per rank)"
rank == 0 && @info "Total obs (T+S): $n_obs_global"

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

        # Reset both clocks
        simulation.model.clock.time = t_start
        simulation.model.clock.iteration = 0
        ocean.model.clock.time = t_start
        ocean.model.clock.iteration = 0
        simulation.stop_time = t_end
        run!(simulation)

        # Extract forecasted state (local)
        ensemble_T[i] = Array(interior(ocean.model.tracers.T))
        ensemble_S[i] = Array(interior(ocean.model.tracers.S))
        ensemble_e[i] = Array(interior(ocean.model.tracers.e))
        ensemble_u[i] = Array(interior(ocean.model.velocities.u))
        ensemble_v[i] = Array(interior(ocean.model.velocities.v))
        ensemble_w[i] = Array(interior(ocean.model.velocities.w))
        ensemble_η[i] = Array(interior(ocean.model.free_surface.η))
    end

    forecast_time = time() - cycle_start

    # --- Get truth observations at analysis time ---
    _, obs_idx = findmin(abs.(nature_times .- t_end))

    # Local truth surface (from per-rank nature run output)
    T_truth_local = interior(nature_Tt[obs_idx], :, :, 1)
    S_truth_local = interior(nature_St[obs_idx], :, :, 1)

    # Thin local observations
    T_truth_thin = vec(thin_surface(T_truth_local, thin_factor))
    S_truth_thin = vec(thin_surface(S_truth_local, thin_factor))

    # Gather thinned truth from all ranks
    T_truth_global = MPI.Allgather(T_truth_thin, comm)
    S_truth_global = MPI.Allgather(S_truth_thin, comm)
    y_obs = vcat(T_truth_global, S_truth_global)

    # --- Compute forecast diagnostics ---
    # Local ensemble mean surface
    T_ens_local = [e[:, :, Nz_local] for e in ensemble_T]
    S_ens_local = [e[:, :, Nz_local] for e in ensemble_S]
    T_mean_local = mean(T_ens_local)
    S_mean_local = mean(S_ens_local)

    # Local RMSE contribution (sum of squared differences)
    local_T_sse = sum((T_mean_local .- T_truth_local) .^ 2)
    local_S_sse = sum((S_mean_local .- S_truth_local) .^ 2)
    local_n = length(T_mean_local)

    # Global RMSE via MPI reduction
    global_T_sse = MPI.Allreduce(local_T_sse, MPI.SUM, comm)
    global_S_sse = MPI.Allreduce(local_S_sse, MPI.SUM, comm)
    global_n = MPI.Allreduce(Float64(local_n), MPI.SUM, comm)

    rmse_T_fc = sqrt(global_T_sse / global_n)
    rmse_S_fc = sqrt(global_S_sse / global_n)

    # Ensemble spread (local contribution)
    local_T_spread_sse = sum(mean((t .- T_mean_local).^2 for t in T_ens_local))
    local_S_spread_sse = sum(mean((s .- S_mean_local).^2 for s in S_ens_local))
    global_T_spread_sse = MPI.Allreduce(local_T_spread_sse, MPI.SUM, comm)
    global_S_spread_sse = MPI.Allreduce(local_S_spread_sse, MPI.SUM, comm)
    T_sprd = sqrt(global_T_spread_sse / global_n)
    S_sprd = sqrt(global_S_spread_sse / global_n)

    push!(rmse_T_before, rmse_T_fc)
    push!(rmse_S_before, rmse_S_fc)
    push!(spread_T, T_sprd)
    push!(spread_S, S_sprd)
    push!(analysis_days, day_num)

    rank == 0 && @info "  Forecast: SST RMSE=$(round(rmse_T_fc, digits=4)) C, spread=$(round(T_sprd, digits=4)) C"

    # --- ETKF Analysis ---
    # Build thinned observation-space ensemble matrix
    Y_b = zeros(n_obs_global, N_ens)
    for i in 1:N_ens
        T_thin_local = vec(thin_surface(ensemble_T[i][:, :, Nz_local], thin_factor))
        S_thin_local = vec(thin_surface(ensemble_S[i][:, :, Nz_local], thin_factor))

        # Gather across ranks
        T_thin_global = MPI.Allgather(T_thin_local, comm)
        S_thin_global = MPI.Allgather(S_thin_local, comm)

        Y_b[:, i] = vcat(T_thin_global, S_thin_global)
    end

    # Compute weight matrix (same on all ranks - deterministic)
    W = compute_etkf_weights(Y_b, y_obs, R_inv_global)

    # Apply weights to local T and S (full 3D)
    apply_etkf_update!(ensemble_T, W, inflation)
    apply_etkf_update!(ensemble_S, W, inflation)

    # --- Analysis diagnostics ---
    T_mean_a = mean([e[:, :, Nz_local] for e in ensemble_T])
    S_mean_a = mean([e[:, :, Nz_local] for e in ensemble_S])
    local_T_sse_a = sum((T_mean_a .- T_truth_local) .^ 2)
    local_S_sse_a = sum((S_mean_a .- S_truth_local) .^ 2)
    global_T_sse_a = MPI.Allreduce(local_T_sse_a, MPI.SUM, comm)
    global_S_sse_a = MPI.Allreduce(local_S_sse_a, MPI.SUM, comm)
    rmse_T_an = sqrt(global_T_sse_a / global_n)
    rmse_S_an = sqrt(global_S_sse_a / global_n)

    push!(rmse_T_after, rmse_T_an)
    push!(rmse_S_after, rmse_S_an)

    cycle_time = time() - cycle_start
    rank == 0 && @info "  Analysis: SST RMSE=$(round(rmse_T_an, digits=4)) C ($(round(forecast_time, digits=1))s forecast, $(round(cycle_time, digits=1))s total)"
end

total_wall = time() - wall_start
rank == 0 && @info "DA complete! Total wall time: $(round(total_wall / 60, digits=1)) minutes"

# ============================================================================
#                           SAVE DIAGNOSTICS (rank 0)
# ============================================================================

if rank == 0
    @info "Saving diagnostics..."
    jldsave(joinpath(output_dir, "ensemble_diagnostics.jld2");
            rmse_T_before, rmse_T_after,
            rmse_S_before, rmse_S_after,
            spread_T, spread_S,
            analysis_days,
            N_ens, inflation, sigma_T, sigma_S, thin_factor,
            total_wall_seconds=total_wall)

    # Save final ensemble mean (global, gathered from all ranks)
    T_final_locals = mean(ensemble_T)
    S_final_locals = mean(ensemble_S)
    T_final_surf = T_final_locals[:, :, Nz_local]
    S_final_surf = S_final_locals[:, :, Nz_local]
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
        lines!(ax1, analysis_days, rmse_T_before, linewidth=2, color=:orange,
               label="DA forecast", linestyle=:dash)
        lines!(ax1, analysis_days, rmse_T_after, linewidth=2, color=:green,
               label="DA analysis")
        axislegend(ax1, position=:lt)

        ax2 = Axis(fig[1, 2], xlabel="Day", ylabel="Spread / RMSE (C)",
                    title="Ensemble Spread vs RMSE")
        lines!(ax2, analysis_days, rmse_T_before, linewidth=2, color=:orange, label="Forecast RMSE")
        lines!(ax2, analysis_days, spread_T, linewidth=2, color=:purple, label="Spread")
        axislegend(ax2, position=:lt)

        save(joinpath(output_dir, "rmse_comparison.png"), fig, px_per_unit=2)
        @info "RMSE comparison plot saved"

        # Final state comparison (using gathered global surface)
        nature_final_file = joinpath(nature_dir, "final_surface_global.jld2")
        if isfile(nature_final_file)
            nature_final = jldopen(nature_final_file)
            nature_T_final = nature_final["T_surf"]
            close(nature_final)

            fig2 = Figure(size=(1400, 500))
            Label(fig2[0, :], "Final SST Comparison (Day $(N_cycles))", fontsize=18, tellwidth=false)

            ax1 = Axis(fig2[1, 1], title="Nature Run (truth)")
            hm1 = heatmap!(ax1, nature_T_final, colormap=:thermal, colorrange=(10, 32))

            ax2 = Axis(fig2[1, 2], title="DA Ensemble Mean")
            hm2 = heatmap!(ax2, T_mean_surf_global, colormap=:thermal, colorrange=(10, 32))

            ax3 = Axis(fig2[1, 3], title="DA Mean - Nature")
            T_da_diff = T_mean_surf_global .- nature_T_final
            hm3 = heatmap!(ax3, T_da_diff, colormap=:balance, colorrange=(-1, 1))

            Colorbar(fig2[1, 4], hm1, label="C")

            save(joinpath(output_dir, "final_comparison.png"), fig2, px_per_unit=2)
            @info "Final comparison plot saved"
        end

        # Ensemble spread plot
        fig3 = Figure(size=(800, 400))
        ax = Axis(fig3[1, 1], xlabel="Day", ylabel="Value",
                  title="Ensemble Diagnostics Over Time")
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
rank == 0 && @info "  Final forecast SST RMSE: $(round(rmse_T_before[end], digits=4)) C"
rank == 0 && @info "  Final analysis SST RMSE: $(round(rmse_T_after[end], digits=4)) C"
rank == 0 && @info "  Output: $output_dir"
