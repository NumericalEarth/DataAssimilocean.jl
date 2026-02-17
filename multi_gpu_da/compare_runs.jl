# Compare nature run and free run (serial post-processing)
#
# Loads global final surface fields from nature and free runs,
# computes RMSE, and generates diagnostic plots.
#
# Usage:
#   julia --project multi_gpu_da/compare_runs.jl NATURE_DIR FREE_DIR PLOT_DIR

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using CairoMakie
using Statistics

nature_dir = ARGS[1]
free_dir = ARGS[2]
plot_dir = ARGS[3]
mkpath(plot_dir)

# ============================================================================
#                           LOAD GLOBAL SURFACE FIELDS
# ============================================================================

@info "Loading global surface fields..."
nature_data = jldopen(joinpath(nature_dir, "final_surface_global.jld2"))
free_data = jldopen(joinpath(free_dir, "final_surface_global.jld2"))

nature_T = nature_data["T_surf"]
nature_S = nature_data["S_surf"]
free_T = free_data["T_surf"]
free_S = free_data["S_surf"]

close(nature_data)
close(free_data)

@info "  Nature T range: ($(minimum(nature_T)), $(maximum(nature_T)))"
@info "  Free T range: ($(minimum(free_T)), $(maximum(free_T)))"

# ============================================================================
#                           COMPUTE RMSE
# ============================================================================

T_diff = free_T .- nature_T
S_diff = free_S .- nature_S

# Mask out land (where T ≈ 0 in both)
ocean_mask = abs.(nature_T) .> 1.0

n_ocean = sum(ocean_mask)
rmse_T = sqrt(sum(T_diff[ocean_mask] .^ 2) / n_ocean)
rmse_S = sqrt(sum(S_diff[ocean_mask] .^ 2) / n_ocean)

@info "Final RMSE (day 14, ocean points only):"
@info "  SST RMSE: $(round(rmse_T, digits=4)) C"
@info "  SSS RMSE: $(round(rmse_S, digits=4)) psu"
@info "  Ocean grid points: $n_ocean / $(length(ocean_mask))"

# Save RMSE data
jldsave(joinpath(plot_dir, "rmse_data.jld2");
        rmse_T_final=rmse_T, rmse_S_final=rmse_S, n_ocean=n_ocean)

# ============================================================================
#                           PLOTS
# ============================================================================

@info "Generating comparison plots..."

try
    # Final state difference maps
    fig = Figure(size=(1400, 800))
    Label(fig[0, :], "Final State Differences (Free Run - Nature Run, Day 14)",
          fontsize=18, tellwidth=false)

    ax1 = Axis(fig[1, 1], title="Nature Run SST")
    hm1 = heatmap!(ax1, nature_T, colormap=:thermal, colorrange=(10, 32))
    Colorbar(fig[1, 2], hm1, label="C")

    ax2 = Axis(fig[1, 3], title="Free Run SST")
    hm2 = heatmap!(ax2, free_T, colormap=:thermal, colorrange=(10, 32))
    Colorbar(fig[1, 4], hm2, label="C")

    T_clim = max(0.5, 3 * std(T_diff[ocean_mask]))
    ax3 = Axis(fig[2, 1], title="SST Difference")
    hm3 = heatmap!(ax3, T_diff, colormap=:balance, colorrange=(-T_clim, T_clim))
    Colorbar(fig[2, 2], hm3, label="C")

    S_clim = max(0.05, 3 * std(S_diff[ocean_mask]))
    ax4 = Axis(fig[2, 3], title="SSS Difference")
    hm4 = heatmap!(ax4, S_diff, colormap=:balance, colorrange=(-S_clim, S_clim))
    Colorbar(fig[2, 4], hm4, label="psu")

    save(joinpath(plot_dir, "final_difference.png"), fig, px_per_unit=2)
    @info "Final difference plot saved"

catch e
    @warn "Plot generation failed" exception=(e, catch_backtrace())
end

@info "Comparison complete!"
@info "  SST RMSE: $(round(rmse_T, digits=4)) C"
@info "  SSS RMSE: $(round(rmse_S, digits=4)) psu"
