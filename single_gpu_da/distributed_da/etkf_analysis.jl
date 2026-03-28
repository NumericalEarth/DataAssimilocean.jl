# ETKF analysis step: load forecasts, compute weights, apply update, save diagnostics
#
# Usage:
#   julia --project single_gpu_da/distributed_da/etkf_analysis.jl \
#       SPINUP_DIR NATURE_DIR ENSEMBLE_DIR OUTPUT_DIR CYCLE [FREE_DIR] [DIAG_DIR] [N_CYCLES]
#
# ENSEMBLE_DIR: base directory containing forecast_XX/ and cycle_XX/
# OUTPUT_DIR:   same base directory (writes cycle_XX/ for updated states)
# CYCLE:        1-based cycle number
# FREE_DIR:     optional free run directory for baseline RMSE
# DIAG_DIR:     optional diagnostics output directory
# N_CYCLES:     optional override for total number of cycles

include(joinpath(@__DIR__, "..", "..", "experiments", "irma_utils.jl"))
include(joinpath(@__DIR__, "da_utils.jl"))

using Oceananigans.Grids: znodes, xnodes, ynodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.OutputReaders: FieldTimeSeries
using Printf
using Dates
using CairoMakie

spinup_dir   = ARGS[1]
nature_dir   = ARGS[2]
ensemble_dir = ARGS[3]
output_dir   = ARGS[4]
cycle        = parse(Int, ARGS[5])
free_dir     = length(ARGS) >= 6 ? ARGS[6] : nothing
diag_dir     = length(ARGS) >= 7 ? ARGS[7] : joinpath(output_dir, "diagnostics")
N_cycles     = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : DA_N_CYCLES
N_ens        = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : DA_N_ENS

mkpath(diag_dir)

@info "LETKF Analysis (localized)"
@info "  Cycle: $cycle / $N_cycles"
@info "  Ensemble size: $N_ens"
@info "  Ensemble dir: $ensemble_dir"
@info "  Output dir: $output_dir"
@info "  Diagnostics dir: $diag_dir"

# --- Time ---
analysis_window = DA_ANALYSIS_WINDOW_SECONDS
t_end = cycle * analysis_window
day_num = t_end / 86400.0

@info "  Analysis time: day $day_num"

# --- Setup grid (needed for vertical levels) ---
arch = GPU()
grid = create_grid(arch, DA_RESOLUTION)
Nx, Ny, Nz = size(grid)

z_faces = Array(znodes(grid, Face()))
k_obs_all = Int[]
for k in 1:Nz
    z_center = 0.5 * (z_faces[k] + z_faces[k+1])
    if z_center >= DA_OBS_DEPTH_LIMIT
        push!(k_obs_all, k)
    end
end
# Subsample vertical levels to reduce overdetermination (31 levels → ~8)
k_obs = k_obs_all[1:DA_OBS_VERTICAL_SKIP:end]
n_obs_levels = length(k_obs)
@info "  Observation levels: $n_obs_levels (of $(length(k_obs_all)) above $(DA_OBS_DEPTH_LIMIT)m)"

# Add bathymetry (needed for grid size verification)
grid = load_or_compute_bathymetry(grid, spinup_dir)

# --- Load all member forecast states ---
fc_dir = forecast_dirname(ensemble_dir, cycle)
@info "Loading $N_ens member forecasts from $fc_dir..."

ensemble_T = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_S = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_e = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_u = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_v = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_w = Vector{Array{Float64, 3}}(undef, N_ens)
ensemble_eta = Vector{Any}(undef, N_ens)

for m in 1:N_ens
    state = load_member_state(member_filename(fc_dir, m))
    ensemble_T[m] = state.T
    ensemble_S[m] = state.S
    ensemble_e[m] = state.e
    ensemble_u[m] = state.u
    ensemble_v[m] = state.v
    ensemble_w[m] = state.w
    ensemble_eta[m] = state.eta
end

Nx_state, Ny_state, Nz_state = size(ensemble_T[1])

# --- Generate Argo-like profile locations (deterministic, same seed as member jobs) ---
profile_rng = MersenneTwister(DA_OBS_SEED + cycle)
i_profiles = rand(profile_rng, 1:Nx_state, DA_N_PROFILES_PER_CYCLE)
j_profiles = rand(profile_rng, 1:Ny_state, DA_N_PROFILES_PER_CYCLE)

# --- Load nature run 3D state at analysis time ---
@info "Loading nature run 3D tracers..."
nature_3d_file = joinpath(nature_dir, "tracers_3d.jld2")
nature_3d_fh = jldopen(nature_3d_file)
nature_3d_tkeys = sort(parse.(Int, keys(nature_3d_fh["timeseries/t"])))
nature_3d_times = [nature_3d_fh["timeseries/t/$(k)"] for k in nature_3d_tkeys]

_, obs_3d_idx = findmin(abs.(nature_3d_times .- t_end))
tkey = nature_3d_tkeys[obs_3d_idx]
nature_T_3d = nature_3d_fh["timeseries/T/$(tkey)"]
nature_S_3d = nature_3d_fh["timeseries/S/$(tkey)"]
close(nature_3d_fh)

# --- Extract truth observations at profile locations ---
n_obs_per_field = DA_N_PROFILES_PER_CYCLE * n_obs_levels
n_obs_total = 2 * n_obs_per_field  # T + S

obs_T = zeros(n_obs_per_field)
obs_S = zeros(n_obs_per_field)
extract_profile_observations!(obs_T, nature_T_3d, i_profiles, j_profiles, k_obs)
extract_profile_observations!(obs_S, nature_S_3d, i_profiles, j_profiles, k_obs)
y_obs = vcat(obs_T, obs_S)

# --- Filter out land observations ---
valid_mask = trues(n_obs_total)
for p in 1:DA_N_PROFILES_PER_CYCLE
    surf_idx = (p - 1) * n_obs_levels + n_obs_levels  # last level = surface
    if abs(obs_T[surf_idx]) < 1.0
        for l in 1:n_obs_levels
            valid_mask[(p - 1) * n_obs_levels + l] = false                   # T obs
            valid_mask[n_obs_per_field + (p - 1) * n_obs_levels + l] = false  # S obs
        end
    end
end
n_valid = sum(valid_mask)

# --- Compute surface forecast diagnostics ---
@info "Loading nature surface observations..."
nature_surface_file = joinpath(nature_dir, "surface_fields.jld2")
nature_Tt = FieldTimeSeries(nature_surface_file, "T")
nature_St = FieldTimeSeries(nature_surface_file, "S")
nature_surface_times = nature_Tt.times

_, surf_idx = findmin(abs.(nature_surface_times .- t_end))
T_truth_surf = interior(nature_Tt[surf_idx], :, :, 1)
S_truth_surf = interior(nature_St[surf_idx], :, :, 1)

T_ens_mean = mean(ensemble_T)
S_ens_mean = mean(ensemble_S)
T_mean_surf = T_ens_mean[:, :, Nz_state]
S_mean_surf = S_ens_mean[:, :, Nz_state]

rmse_T_fc = sqrt(mean((T_mean_surf .- T_truth_surf) .^ 2))
rmse_S_fc = sqrt(mean((S_mean_surf .- S_truth_surf) .^ 2))

# Ensemble spread
T_surfaces = [e[:, :, Nz_state] for e in ensemble_T]
S_surfaces = [e[:, :, Nz_state] for e in ensemble_S]
T_sprd = sqrt(mean(mean((t .- T_mean_surf).^2 for t in T_surfaces)))
S_sprd = sqrt(mean(mean((s .- S_mean_surf).^2 for s in S_surfaces)))

# Free run baseline RMSE
rmse_T_free_val = NaN
rmse_S_free_val = NaN
if !isnothing(free_dir)
    @info "Loading free run surface fields..."
    free_surface_file = joinpath(free_dir, "surface_fields.jld2")
    if isfile(free_surface_file)
        free_Tt = FieldTimeSeries(free_surface_file, "T")
        free_St = FieldTimeSeries(free_surface_file, "S")
        free_surface_times = free_Tt.times
        _, free_idx = findmin(abs.(free_surface_times .- t_end))
        T_free_surf = interior(free_Tt[free_idx], :, :, 1)
        S_free_surf = interior(free_St[free_idx], :, :, 1)
        rmse_T_free_val = sqrt(mean((T_free_surf .- T_truth_surf) .^ 2))
        rmse_S_free_val = sqrt(mean((S_free_surf .- S_truth_surf) .^ 2))
    end
end

@info "  Free run SST RMSE=$(round(rmse_T_free_val, digits=4)) C"
@info "  DA forecast SST RMSE=$(round(rmse_T_fc, digits=4)) C, spread=$(round(T_sprd, digits=4)) C"
@info "  Valid obs: $n_valid / $n_obs_total ($DA_N_PROFILES_PER_CYCLE profiles)"

# --- LETKF Analysis (localized) ---
if n_valid > 0
    # Build ensemble observation matrix (full, used by LETKF to extract local subsets)
    Y_b = zeros(n_obs_total, N_ens)
    member_obs_T = zeros(n_obs_per_field)
    member_obs_S = zeros(n_obs_per_field)

    for m in 1:N_ens
        extract_profile_observations!(member_obs_T, ensemble_T[m], i_profiles, j_profiles, k_obs)
        extract_profile_observations!(member_obs_S, ensemble_S[m], i_profiles, j_profiles, k_obs)
        Y_b[1:n_obs_per_field, m] .= member_obs_T
        Y_b[n_obs_per_field+1:end, m] .= member_obs_S
    end

    # R inverse (observation error covariance inverse, diagonal)
    R_inv = vcat(fill(1.0 / DA_SIGMA_T^2, n_obs_per_field),
                 fill(1.0 / DA_SIGMA_S^2, n_obs_per_field))

    # Compute grid lon/lat centers for localization
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lon_centers = Array(xnodes(underlying, Center(), Center(), Center()))
    lat_centers = Array(ynodes(underlying, Center(), Center(), Center()))

    # Compute profile lon/lat from grid indices
    lon_profiles = [lon_centers[i_profiles[p]] for p in 1:DA_N_PROFILES_PER_CYCLE]
    lat_profiles = [lat_centers[j_profiles[p]] for p in 1:DA_N_PROFILES_PER_CYCLE]

    @info "  LETKF with $(DA_LOCALIZATION_RADIUS) km localization radius"
    letkf_start = time()

    apply_letkf_update!(ensemble_T, ensemble_S, Y_b, y_obs, R_inv, valid_mask,
                         i_profiles, j_profiles, k_obs, n_obs_levels,
                         lon_centers, lat_centers, lon_profiles, lat_profiles,
                         DA_INFLATION, DA_LOCALIZATION_RADIUS)

    letkf_time = time() - letkf_start
    @info "  LETKF completed in $(round(letkf_time, digits=1))s"
else
    @warn "  No valid observations this cycle - skipping analysis"
end

# --- Analysis diagnostics ---
T_ens_mean_a = mean(ensemble_T)
S_ens_mean_a = mean(ensemble_S)
rmse_T_an = sqrt(mean((T_ens_mean_a[:, :, Nz_state] .- T_truth_surf) .^ 2))
rmse_S_an = sqrt(mean((S_ens_mean_a[:, :, Nz_state] .- S_truth_surf) .^ 2))

@info "  Analysis: SST RMSE=$(round(rmse_T_an, digits=4)) C"

# --- Save analysis states ---
analysis_cycle_dir = cycle_dirname(output_dir, cycle)
mkpath(analysis_cycle_dir)

for m in 1:N_ens
    save_member_state(member_filename(analysis_cycle_dir, m);
        T   = ensemble_T[m],
        S   = ensemble_S[m],
        e   = ensemble_e[m],
        u   = ensemble_u[m],
        v   = ensemble_v[m],
        w   = ensemble_w[m],
        eta = ensemble_eta[m])
end

@info "Analysis states saved to $analysis_cycle_dir"

# --- Save cycle diagnostics ---
diag_file = joinpath(diag_dir, @sprintf("cycle_%02d_diagnostics.jld2", cycle))
jldsave(diag_file;
    cycle, day_num,
    rmse_T_before = rmse_T_fc,
    rmse_T_after  = rmse_T_an,
    rmse_T_free   = rmse_T_free_val,
    rmse_S_before = rmse_S_fc,
    rmse_S_after  = rmse_S_an,
    rmse_S_free   = rmse_S_free_val,
    spread_T = T_sprd,
    spread_S = S_sprd,
    n_valid_obs = n_valid,
    N_ens, inflation = DA_INFLATION,
    sigma_T = DA_SIGMA_T, sigma_S = DA_SIGMA_S)

@info "Cycle diagnostics saved to $diag_file"

# --- On final cycle, aggregate diagnostics and generate plots ---
if cycle == N_cycles
    @info "Final cycle! Aggregating diagnostics..."

    rmse_T_before_all = Float64[]
    rmse_T_after_all  = Float64[]
    rmse_T_free_all   = Float64[]
    rmse_S_before_all = Float64[]
    rmse_S_after_all  = Float64[]
    rmse_S_free_all   = Float64[]
    spread_T_all      = Float64[]
    spread_S_all      = Float64[]
    analysis_days_all = Float64[]
    n_valid_obs_all   = Int[]

    for c in 1:N_cycles
        df = joinpath(diag_dir, @sprintf("cycle_%02d_diagnostics.jld2", c))
        if isfile(df)
            d = jldopen(df)
            push!(rmse_T_before_all, d["rmse_T_before"])
            push!(rmse_T_after_all,  d["rmse_T_after"])
            push!(rmse_T_free_all,   d["rmse_T_free"])
            push!(rmse_S_before_all, d["rmse_S_before"])
            push!(rmse_S_after_all,  d["rmse_S_after"])
            push!(rmse_S_free_all,   d["rmse_S_free"])
            push!(spread_T_all,      d["spread_T"])
            push!(spread_S_all,      d["spread_S"])
            push!(analysis_days_all, d["day_num"])
            push!(n_valid_obs_all,   d["n_valid_obs"])
            close(d)
        else
            @warn "Missing diagnostics for cycle $c"
        end
    end

    # Save aggregated diagnostics (same format as monolithic version)
    jldsave(joinpath(diag_dir, "ensemble_diagnostics.jld2");
            rmse_T_before = rmse_T_before_all,
            rmse_T_after  = rmse_T_after_all,
            rmse_T_free   = rmse_T_free_all,
            rmse_S_before = rmse_S_before_all,
            rmse_S_after  = rmse_S_after_all,
            rmse_S_free   = rmse_S_free_all,
            spread_T = spread_T_all,
            spread_S = spread_S_all,
            analysis_days = analysis_days_all,
            n_valid_obs_log = n_valid_obs_all,
            N_ens, inflation = DA_INFLATION,
            sigma_T = DA_SIGMA_T, sigma_S = DA_SIGMA_S,
            n_profiles_per_cycle = DA_N_PROFILES_PER_CYCLE,
            obs_depth_limit = DA_OBS_DEPTH_LIMIT,
            n_obs_levels)

    # Save final ensemble mean state
    T_final_mean = mean(ensemble_T)
    S_final_mean = mean(ensemble_S)
    jldsave(joinpath(diag_dir, "da_final_mean_state.jld2");
            T=T_final_mean, S=S_final_mean)

    @info "Aggregated diagnostics saved."

    # --- Visualization ---
    try
        @info "Generating DA diagnostic plots..."

        fig = Figure(size=(1000, 500))

        ax1 = Axis(fig[1, 1], xlabel="Day", ylabel="RMSE (C)",
                    title="SST RMSE: DA vs Free Run")
        lines!(ax1, analysis_days_all, rmse_T_free_all, linewidth=2, color=:red,
               label="Free run (no DA)")
        lines!(ax1, analysis_days_all, rmse_T_before_all, linewidth=2, color=:orange,
               label="DA forecast", linestyle=:dash)
        lines!(ax1, analysis_days_all, rmse_T_after_all, linewidth=2, color=:green,
               label="DA analysis")
        axislegend(ax1, position=:lt)

        ax2 = Axis(fig[1, 2], xlabel="Day", ylabel="Spread / RMSE (C)",
                    title="Ensemble Spread vs RMSE")
        lines!(ax2, analysis_days_all, rmse_T_free_all, linewidth=2, color=:red, label="Free run RMSE")
        lines!(ax2, analysis_days_all, rmse_T_before_all, linewidth=2, color=:orange, label="Forecast RMSE")
        lines!(ax2, analysis_days_all, spread_T_all, linewidth=2, color=:purple, label="Spread")
        axislegend(ax2, position=:lt)

        save(joinpath(diag_dir, "rmse_comparison.png"), fig, px_per_unit=2)
        @info "RMSE comparison plot saved"

        # Ensemble spread plot
        fig3 = Figure(size=(800, 400))
        ax = Axis(fig3[1, 1], xlabel="Day", ylabel="Value",
                  title="Ensemble Diagnostics (Argo-like, $(DA_N_PROFILES_PER_CYCLE) profiles x $(n_obs_levels) levels)")
        lines!(ax, analysis_days_all, spread_T_all, linewidth=2, color=:purple, label="T spread (C)")
        lines!(ax, analysis_days_all, 10 .* spread_S_all, linewidth=2, color=:blue,
               label="S spread (x10, psu)")
        lines!(ax, analysis_days_all, rmse_T_after_all, linewidth=2, color=:green,
               label="T analysis RMSE (C)", linestyle=:dash)
        axislegend(ax)

        save(joinpath(diag_dir, "ensemble_spread.png"), fig3, px_per_unit=2)
        @info "Ensemble spread plot saved"
    catch e
        @warn "Visualization failed (non-fatal)" exception=(e, catch_backtrace())
    end

    @info "Distributed DA complete!"
    @info "  Final free run SST RMSE: $(round(rmse_T_free_all[end], digits=4)) C"
    @info "  Final forecast SST RMSE: $(round(rmse_T_before_all[end], digits=4)) C"
    @info "  Final analysis SST RMSE: $(round(rmse_T_after_all[end], digits=4)) C"
end

@info "Analysis cycle $cycle complete."
