# Ensemble Transform Kalman Filter (ETKF) for ocean state estimation
#
# Implements a 10-member ETKF following Manteia.jl / Hunt et al. (2007):
#   1. Initialize ensemble from perturbed spinup state
#   2. Forecast: run each member forward 1 day (sequentially on single GPU)
#   3. Analyze: compute ETKF weights from surface T, S observations
#   4. Update: apply weights to full 3D T and S fields
#   5. Repeat for 14 analysis cycles
#
# Usage:
#   julia --project single_gpu_da/ensemble_da.jl SPINUP_DIR NATURE_DIR FREE_DIR OUTPUT_DIR

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Random
using Statistics
using LinearAlgebra
using CairoMakie
using Oceananigans.OutputReaders: FieldTimeSeries

spinup_dir = ARGS[1]
nature_dir = ARGS[2]
free_dir = ARGS[3]
output_dir = ARGS[4]
mkpath(output_dir)

# Optional: override N_cycles from command line (for debug runs)
cli_cycles = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing

# ============================================================================
#                           CONFIGURATION
# ============================================================================

arch = GPU()
resolution = 0.25

# DA parameters
N_ens = 10                  # Ensemble members
N_cycles = isnothing(cli_cycles) ? 14 : cli_cycles  # Number of analysis cycles
analysis_window = 1days     # Forecast window between analyses
Δt = 5minutes               # Model time step
inflation = 1.05            # Multiplicative inflation factor

# Observation errors
sigma_T = 0.1               # SST observation error (°C)
sigma_S = 0.01              # SSS observation error (psu)

# Perturbation parameters (same as free run for consistency)
T_noise_std = 0.5
S_noise_std = 0.05
T_bias = 0.3
S_bias = 0.02
noise_depth_levels = 20

# Dates
start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(N_cycles)

@info "ETKF Configuration:"
@info "  Ensemble size: $N_ens"
@info "  Analysis cycles: $N_cycles"
@info "  Analysis window: $(analysis_window / days) days"
@info "  Inflation: $inflation"
@info "  Obs error: T=$(sigma_T) C, S=$(sigma_S) psu"

# ============================================================================
#                           ETKF ALGORITHM
# ============================================================================

"""
    compute_etkf_weights(Y_b, y_obs, R_inv_diag)

Compute ETKF weight matrix following Hunt et al. (2007).

Arguments:
- Y_b: (n_obs, N_ens) observation-space forecast ensemble
- y_obs: (n_obs,) truth observations
- R_inv_diag: (n_obs,) diagonal of R⁻¹

Returns W: (N_ens, N_ens) weight matrix
"""
function compute_etkf_weights(Y_b, y_obs, R_inv_diag)
    N = size(Y_b, 2)

    # Ensemble mean and perturbations in observation space
    y_mean = vec(mean(Y_b, dims=2))
    S = Y_b .- y_mean  # (n_obs, N)

    # Innovation
    d = y_obs .- y_mean  # (n_obs,)

    # C = S' R⁻¹ S  (N × N)
    RinvS = R_inv_diag .* S  # (n_obs, N)
    C = S' * RinvS           # (N, N)

    # (N-1)I + C
    invTTt = (N - 1) * I + C

    # Eigendecomposition
    F = eigen(Symmetric(invTTt))
    inv_sigma = 1.0 ./ F.values
    U = F.vectors

    # Mean weight: w_mean = U diag(inv_sigma) U' S' R⁻¹ d
    RinvD = R_inv_diag .* d
    w_mean = U * (inv_sigma .* (U' * (S' * RinvD)))

    # Perturbation weight: W_pert = U diag(sqrt((N-1) inv_sigma)) U'
    W_pert = U * Diagonal(sqrt.((N - 1) .* inv_sigma)) * U'

    # Full weight: W = w_mean * 1' + W_pert  (outer product + perturbation)
    W = w_mean * ones(N)' + W_pert

    return W
end

"""
    apply_etkf_update!(ensemble_field, W, inflation)

Apply ETKF weight matrix to update a 3D field for all ensemble members.

Arguments:
- ensemble_field: Vector of N 3D arrays (one per member)
- W: (N, N) weight matrix
- inflation: multiplicative inflation factor

Modifies ensemble_field in place.
"""
function apply_etkf_update!(ensemble_field, W, inflation)
    N = length(ensemble_field)

    # Save forecast (we overwrite in place)
    forecast = [copy(f) for f in ensemble_field]
    field_mean = mean(forecast)

    for m in 1:N
        ensemble_field[m] .= field_mean
        for n in 1:N
            ensemble_field[m] .+= inflation * W[n, m] .* (forecast[n] .- field_mean)
        end
    end

    return nothing
end

# ============================================================================
#                           SETUP
# ============================================================================

@info "Loading nature run surface observations..."
nature_surface_file = joinpath(nature_dir, "surface_fields.jld2")
nature_Tt = FieldTimeSeries(nature_surface_file, "T")
nature_St = FieldTimeSeries(nature_surface_file, "S")
nature_times = nature_Tt.times

@info "Loading spinup state..."
spinup_state = load_model_state(joinpath(spinup_dir, "final_state.jld2"))
Nx_state, Ny_state, Nz_state = size(spinup_state.T)

# Load free run RMSE for comparison (if available)
free_rmse_T = nothing
free_rmse_days = nothing
free_rmse_file = joinpath(dirname(free_dir), "plots", "rmse_data.jld2")
if isfile(free_rmse_file)
    @info "Loading free run RMSE data..."
    free_data = jldopen(free_rmse_file)
    free_rmse_T = free_data["rmse_T"]
    free_rmse_days = free_data["days"]
    close(free_data)
end

# ============================================================================
#                           INITIALIZE ENSEMBLE
# ============================================================================

@info "Initializing $N_ens ensemble members..."
ensemble_T = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_S = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_e = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_u = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_v = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_w = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_η = Vector{Any}(undef, N_ens)  # η may be 2D or 3D depending on Oceananigans version

k_start = max(1, Nz_state - noise_depth_levels + 1)

for i in 1:N_ens
    rng = MersenneTwister(42 + i)  # Different seed per member

    T_i = copy(spinup_state.T)
    S_i = copy(spinup_state.S)

    for k in k_start:Nz_state
        T_i[:, :, k] .+= T_noise_std .* randn(rng, Nx_state, Ny_state) .+ T_bias
        S_i[:, :, k] .+= S_noise_std .* randn(rng, Nx_state, Ny_state) .+ S_bias
    end

    ensemble_T[i] = T_i
    ensemble_S[i] = S_i
    ensemble_e[i] = copy(spinup_state.e)
    ensemble_u[i] = copy(spinup_state.u)
    ensemble_v[i] = copy(spinup_state.v)
    ensemble_w[i] = copy(spinup_state.w)
    ensemble_η[i] = copy(spinup_state.η)
end

@info "Ensemble initialized. Member 1 T range: ($(minimum(ensemble_T[1])), $(maximum(ensemble_T[1])))"

# ============================================================================
#                           CREATE MODEL (single instance)
# ============================================================================

@info "Creating ocean model..."
grid = create_grid(arch, resolution)
grid = load_or_compute_bathymetry(grid, spinup_dir)
Nx, Ny, Nz = size(grid)

ocean = create_ocean_simulation(grid, start_date, end_date; with_restoring=true)

@info "Setting up atmosphere..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=N_cycles * analysis_window)

# Remove output writers for ensemble forecast runs
empty!(ocean.output_writers)

# Single initialization run to compile kernels
@info "Compiling model (warm-up run)..."
set!(ocean.model, T=ensemble_T[1], S=ensemble_S[1], e=ensemble_e[1],
     u=ensemble_u[1], v=ensemble_v[1])
copyto!(interior(ocean.model.velocities.w), ensemble_w[1])
copyto!(interior(ocean.model.free_surface.η), ensemble_η[1])

# Reset both the coupled model clock and the ocean model clock
simulation.model.clock.time = 0.0
simulation.model.clock.iteration = 0
ocean.model.clock.time = 0.0
ocean.model.clock.iteration = 0
simulation.stop_time = Float64(Δt)

run!(simulation)

# Extract back (discard warm-up)
@info "Warm-up complete. Starting DA loop."

# ============================================================================
#                           DA LOOP
# ============================================================================

# Diagnostics storage
rmse_T_before = Float64[]  # Forecast RMSE (ensemble mean vs truth)
rmse_T_after = Float64[]   # Analysis RMSE
rmse_S_before = Float64[]
rmse_S_after = Float64[]
spread_T = Float64[]       # Ensemble spread
spread_S = Float64[]
analysis_days = Float64[]

# Observation error inverse (diagonal)
n_T_obs = Nx_state * Ny_state
n_S_obs = Nx_state * Ny_state
n_obs = n_T_obs + n_S_obs
R_inv = vcat(fill(1.0 / sigma_T^2, n_T_obs), fill(1.0 / sigma_S^2, n_S_obs))

wall_start = time()

for cycle in 1:N_cycles
    cycle_start = time()
    t_start = (cycle - 1) * Float64(analysis_window)
    t_end = cycle * Float64(analysis_window)
    day_num = t_end / (24 * 3600)

    @info "=== DA Cycle $cycle/$N_cycles (day $day_num) ==="

    # --- Forecast: run each member forward one analysis window ---
    for i in 1:N_ens
        # Set member state
        set!(ocean.model, T=ensemble_T[i], S=ensemble_S[i], e=ensemble_e[i],
             u=ensemble_u[i], v=ensemble_v[i])
        copyto!(interior(ocean.model.velocities.w), ensemble_w[i])
        copyto!(interior(ocean.model.free_surface.η), ensemble_η[i])

        # Reset both clocks and run
        simulation.model.clock.time = t_start
        simulation.model.clock.iteration = 0
        ocean.model.clock.time = t_start
        ocean.model.clock.iteration = 0
        simulation.stop_time = t_end
        run!(simulation)

        # Extract forecasted state
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
    # Find the nature run snapshot closest to t_end
    _, obs_idx = findmin(abs.(nature_times .- t_end))
    T_truth_surf = interior(nature_Tt[obs_idx], :, :, 1)
    S_truth_surf = interior(nature_St[obs_idx], :, :, 1)
    y_obs = vcat(vec(T_truth_surf), vec(S_truth_surf))

    # --- Compute forecast diagnostics ---
    T_ens_mean = mean(ensemble_T)
    S_ens_mean = mean(ensemble_S)
    T_mean_surf = T_ens_mean[:, :, Nz_state]
    S_mean_surf = S_ens_mean[:, :, Nz_state]

    rmse_T_fc = sqrt(mean((T_mean_surf .- T_truth_surf) .^ 2))
    rmse_S_fc = sqrt(mean((S_mean_surf .- S_truth_surf) .^ 2))

    # Ensemble spread (RMS of per-point standard deviation)
    T_surfaces = [e[:, :, Nz_state] for e in ensemble_T]
    S_surfaces = [e[:, :, Nz_state] for e in ensemble_S]
    T_spread = sqrt(mean(mean((t .- T_mean_surf).^2 for t in T_surfaces)))
    S_spread = sqrt(mean(mean((s .- S_mean_surf).^2 for s in S_surfaces)))

    push!(rmse_T_before, rmse_T_fc)
    push!(rmse_S_before, rmse_S_fc)
    push!(spread_T, T_spread)
    push!(spread_S, S_spread)
    push!(analysis_days, day_num)

    @info "  Forecast: SST RMSE=$(round(rmse_T_fc, digits=4))°C, spread=$(round(T_spread, digits=4))°C"

    # --- ETKF Analysis ---
    # Build observation-space ensemble matrix
    Y_b = zeros(n_obs, N_ens)
    for i in 1:N_ens
        Y_b[1:n_T_obs, i] = vec(ensemble_T[i][:, :, Nz_state])
        Y_b[n_T_obs+1:end, i] = vec(ensemble_S[i][:, :, Nz_state])
    end

    # Compute weight matrix
    W = compute_etkf_weights(Y_b, y_obs, R_inv)

    # Apply weights to T and S (full 3D)
    apply_etkf_update!(ensemble_T, W, inflation)
    apply_etkf_update!(ensemble_S, W, inflation)

    # --- Analysis diagnostics ---
    T_ens_mean_a = mean(ensemble_T)
    S_ens_mean_a = mean(ensemble_S)
    rmse_T_an = sqrt(mean((T_ens_mean_a[:, :, Nz_state] .- T_truth_surf) .^ 2))
    rmse_S_an = sqrt(mean((S_ens_mean_a[:, :, Nz_state] .- S_truth_surf) .^ 2))

    push!(rmse_T_after, rmse_T_an)
    push!(rmse_S_after, rmse_S_an)

    cycle_time = time() - cycle_start
    @info "  Analysis: SST RMSE=$(round(rmse_T_an, digits=4))°C ($(round(forecast_time, digits=1))s forecast, $(round(cycle_time, digits=1))s total)"
end

total_wall = time() - wall_start
@info "DA complete! Total wall time: $(round(total_wall / 60, digits=1)) minutes"

# ============================================================================
#                           SAVE DIAGNOSTICS
# ============================================================================

@info "Saving diagnostics..."
jldsave(joinpath(output_dir, "ensemble_diagnostics.jld2");
        rmse_T_before, rmse_T_after,
        rmse_S_before, rmse_S_after,
        spread_T, spread_S,
        analysis_days,
        N_ens, inflation, sigma_T, sigma_S,
        total_wall_seconds=total_wall)

# Save final ensemble mean state
T_final_mean = mean(ensemble_T)
S_final_mean = mean(ensemble_S)
jldsave(joinpath(output_dir, "da_final_mean_state.jld2");
        T=T_final_mean, S=S_final_mean)

# ============================================================================
#                           VISUALIZATION
# ============================================================================

@info "Generating DA diagnostic plots..."

try
    # --- RMSE Comparison Plot ---
    fig = Figure(size=(1000, 500))

    ax1 = Axis(fig[1, 1], xlabel="Day", ylabel="RMSE (°C)",
               title="SST RMSE: DA vs Free Run")
    lines!(ax1, analysis_days, rmse_T_before, linewidth=2, color=:orange,
           label="DA forecast", linestyle=:dash)
    lines!(ax1, analysis_days, rmse_T_after, linewidth=2, color=:green,
           label="DA analysis")
    if !isnothing(free_rmse_T)
        lines!(ax1, free_rmse_days, free_rmse_T, linewidth=2, color=:red,
               label="Free run (no DA)")
    end
    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[1, 2], xlabel="Day", ylabel="Spread / RMSE (°C)",
               title="Ensemble Spread vs RMSE")
    lines!(ax2, analysis_days, rmse_T_before, linewidth=2, color=:orange, label="Forecast RMSE")
    lines!(ax2, analysis_days, spread_T, linewidth=2, color=:purple, label="Spread")
    axislegend(ax2, position=:lt)

    save(joinpath(output_dir, "rmse_comparison.png"), fig, px_per_unit=2)
    @info "RMSE comparison plot saved"

    # --- Final State Comparison ---
    # Load nature and free run final surface T
    nature_T_final = interior(nature_Tt[length(nature_times)], :, :, 1)

    fig2 = Figure(size=(1400, 500))
    Label(fig2[0, :], "Final SST Comparison (Day 14)", fontsize=18, tellwidth=false)

    ax1 = Axis(fig2[1, 1], title="Nature Run (truth)")
    hm1 = heatmap!(ax1, nature_T_final, colormap=:thermal, colorrange=(10, 32))

    ax2 = Axis(fig2[1, 2], title="DA Ensemble Mean")
    hm2 = heatmap!(ax2, T_final_mean[:, :, Nz_state], colormap=:thermal, colorrange=(10, 32))

    ax3 = Axis(fig2[1, 3], title="DA Mean - Nature")
    T_da_diff = T_final_mean[:, :, Nz_state] .- nature_T_final
    hm3 = heatmap!(ax3, T_da_diff, colormap=:balance, colorrange=(-1, 1))

    Colorbar(fig2[1, 4], hm1, label="°C")

    save(joinpath(output_dir, "final_comparison.png"), fig2, px_per_unit=2)
    @info "Final comparison plot saved"

    # --- Ensemble Spread Plot ---
    fig3 = Figure(size=(800, 400))
    ax = Axis(fig3[1, 1], xlabel="Day", ylabel="Value",
              title="Ensemble Diagnostics Over Time")
    lines!(ax, analysis_days, spread_T, linewidth=2, color=:purple, label="T spread (°C)")
    lines!(ax, analysis_days, 10 .* spread_S, linewidth=2, color=:blue,
           label="S spread (×10, psu)")
    lines!(ax, analysis_days, rmse_T_after, linewidth=2, color=:green,
           label="T analysis RMSE (°C)", linestyle=:dash)
    axislegend(ax)

    save(joinpath(output_dir, "ensemble_spread.png"), fig3, px_per_unit=2)
    @info "Ensemble spread plot saved"
catch e
    @warn "Visualization failed (non-fatal)" exception=(e, catch_backtrace())
end

@info "Ensemble DA experiment complete!"
@info "  Final forecast SST RMSE: $(round(rmse_T_before[end], digits=4))°C"
@info "  Final analysis SST RMSE: $(round(rmse_T_after[end], digits=4))°C"
@info "  Output: $output_dir"
