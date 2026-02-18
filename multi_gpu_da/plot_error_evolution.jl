# Compare error evolution: Free Run vs DA Ensemble Mean
#
# Loads per-rank surface time series from nature and free runs,
# computes RMSE(t) for the free run, and overlays with DA cycle RMSE.
#
# Usage:
#   julia --project multi_gpu_da/plot_error_evolution.jl OUTPUT_BASE

using JLD2
using CairoMakie
using Statistics

output_base = ARGS[1]
nature_dir = joinpath(output_base, "nature_run")
free_dir = joinpath(output_base, "free_run")
da_dir = joinpath(output_base, "da_run")
plot_dir = joinpath(output_base, "plots")
mkpath(plot_dir)

# ============================================================================
#                    LOAD PER-RANK SURFACE TIME SERIES
# ============================================================================

# We load per-rank data and compute local RMSE contributions,
# then sum across ranks for the global RMSE.
# This avoids needing MPI or construct_global_array.

@info "Loading per-rank surface time series..."

# Load nature and free run surface fields for all 4 ranks
# JLD2Writer output structure: timeseries/T/0, timeseries/T/1, ...
# and timeseries/t/0, timeseries/t/1, ... for times

function load_surface_timeseries(dir, rank)
    file = joinpath(dir, "surface_fields_rank$(rank).jld2")
    f = jldopen(file)

    # Get time indices
    t_keys = sort(parse.(Int, keys(f["timeseries/t"])))
    times = [f["timeseries/t/$(k)"] for k in t_keys]

    # Load T surface at each time
    T_series = [f["timeseries/T/$(k)"][:, :, 1] for k in t_keys]
    S_series = [f["timeseries/S/$(k)"][:, :, 1] for k in t_keys]

    close(f)
    return times, T_series, S_series
end

# Load all ranks
nature_data = [load_surface_timeseries(nature_dir, r) for r in 0:3]
free_data = [load_surface_timeseries(free_dir, r) for r in 0:3]

times = nature_data[1][1]  # Same for all ranks
n_times = length(times)
days = times ./ (24 * 3600)

@info "  $(n_times) time snapshots over $(round(days[end], digits=1)) days"

# ============================================================================
#                    COMPUTE FREE RUN RMSE(t)
# ============================================================================

@info "Computing free run RMSE(t)..."

free_rmse_T = Float64[]
free_rmse_S = Float64[]

for ti in 1:n_times
    global_sse_T = 0.0
    global_sse_S = 0.0
    global_n = 0

    for r in 1:4
        nature_T = nature_data[r][2][ti]
        nature_S = nature_data[r][3][ti]
        free_T = free_data[r][2][ti]
        free_S = free_data[r][3][ti]

        # Ocean mask (exclude land where T ≈ 0)
        mask = abs.(nature_T) .> 1.0

        global_sse_T += sum((free_T[mask] .- nature_T[mask]) .^ 2)
        global_sse_S += sum((free_S[mask] .- nature_S[mask]) .^ 2)
        global_n += sum(mask)
    end

    push!(free_rmse_T, sqrt(global_sse_T / global_n))
    push!(free_rmse_S, sqrt(global_sse_S / global_n))
end

@info "  Free run SST RMSE: $(round(free_rmse_T[1], digits=4)) → $(round(free_rmse_T[end], digits=4)) C"

# ============================================================================
#                    LOAD DA DIAGNOSTICS
# ============================================================================

@info "Loading DA diagnostics..."

da_diag = jldopen(joinpath(da_dir, "ensemble_diagnostics.jld2"))
da_rmse_T_before = da_diag["rmse_T_before"]
da_rmse_T_after = da_diag["rmse_T_after"]
da_rmse_S_before = da_diag["rmse_S_before"]
da_rmse_S_after = da_diag["rmse_S_after"]
da_spread_T = da_diag["spread_T"]
da_days = da_diag["analysis_days"]
close(da_diag)

@info "  DA SST RMSE: $(round(da_rmse_T_before[1], digits=4)) → $(round(da_rmse_T_before[end], digits=4)) C"

# ============================================================================
#                    PLOT: ERROR EVOLUTION COMPARISON
# ============================================================================

@info "Generating plots..."

# Main comparison plot
fig = Figure(size=(1200, 800))

# Panel 1: SST RMSE over time
ax1 = Axis(fig[1, 1],
    xlabel="Day",
    ylabel="SST RMSE (C)",
    title="SST Error Evolution: Free Run vs Data Assimilation (1/8 deg, 4 GPUs)")

# Free run RMSE (continuous, every 6 hours)
lines!(ax1, days, free_rmse_T, linewidth=2.5, color=:red, label="Free run (no DA)")

# DA forecast RMSE (daily points)
scatter!(ax1, da_days, da_rmse_T_before, markersize=10, color=:dodgerblue, label="DA forecast RMSE")
lines!(ax1, da_days, da_rmse_T_before, linewidth=1.5, color=:dodgerblue, linestyle=:dash)

# DA analysis RMSE (daily points)
scatter!(ax1, da_days, da_rmse_T_after, markersize=8, color=:green, marker=:diamond, label="DA analysis RMSE")

axislegend(ax1, position=:rt)

# Panel 2: SSS RMSE
ax2 = Axis(fig[2, 1],
    xlabel="Day",
    ylabel="SSS RMSE (psu)",
    title="SSS Error Evolution")

lines!(ax2, days, free_rmse_S, linewidth=2.5, color=:red, label="Free run")
scatter!(ax2, da_days, da_rmse_S_before, markersize=10, color=:dodgerblue, label="DA forecast")
lines!(ax2, da_days, da_rmse_S_before, linewidth=1.5, color=:dodgerblue, linestyle=:dash)

axislegend(ax2, position=:rt)

save(joinpath(plot_dir, "error_evolution.png"), fig, px_per_unit=2)
@info "Saved error_evolution.png"

# Panel 3: Combined with spread
fig2 = Figure(size=(1000, 500))

ax = Axis(fig2[1, 1],
    xlabel="Day",
    ylabel="SST RMSE or Spread (C)",
    title="SST: Free Run Error vs DA Error vs Ensemble Spread")

lines!(ax, days, free_rmse_T, linewidth=3, color=:red, label="Free run RMSE")
lines!(ax, da_days, da_rmse_T_before, linewidth=2.5, color=:dodgerblue, label="DA forecast RMSE")
lines!(ax, da_days, da_spread_T, linewidth=2, color=:purple, linestyle=:dash, label="DA ensemble spread")

axislegend(ax, position=:rt)

save(joinpath(plot_dir, "free_vs_da_summary.png"), fig2, px_per_unit=2)
@info "Saved free_vs_da_summary.png"

@info "Done! Plots in: $plot_dir"
