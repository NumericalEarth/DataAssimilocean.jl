# Post-processing: Assemble global surfaces from per-rank data and generate plots
#
# Serial (no MPI needed) - loads per-rank JLD2 files and stitches them together.
#
# Usage:
#   julia --project multi_gpu_da/postprocess.jl OUTPUT_BASE

using JLD2
using CairoMakie
using Statistics

output_base = ARGS[1]
nature_dir = joinpath(output_base, "nature_run")
free_dir = joinpath(output_base, "free_run")
da_dir = joinpath(output_base, "da_run")
plot_dir = joinpath(output_base, "plots")
mkpath(plot_dir)

# Rank-to-grid mapping for Partition(2,2) with Rx=2, Ry=2, Rz=1:
# Uses rank2index: i=div(r, Ry*Rz), j=div(r % (Ry*Rz), Rz), k=r % Rz
# rank 0 → local_index (1,1) → x[1:Nx_local], y[1:Ny_local]
# rank 1 → local_index (1,2) → x[1:Nx_local], y[Ny_local+1:end]
# rank 2 → local_index (2,1) → x[Nx_local+1:end], y[1:Ny_local]
# rank 3 → local_index (2,2) → x[Nx_local+1:end], y[Ny_local+1:end]

function assemble_global_surface(dir, field_name="T")
    data = [jldopen(joinpath(dir, "final_state_rank$(r).jld2")) do f
        f[field_name]
    end for r in 0:3]

    # Each rank's data is (Nx_local, Ny_local, Nz)
    Nz = size(data[1], 3)

    # Extract surface (k=Nz)
    s = [d[:, :, Nz] for d in data]

    Nx_local, Ny_local = size(s[1])

    # Assemble global surface following Oceananigans rank2index mapping
    global_surf = zeros(2 * Nx_local, 2 * Ny_local)
    global_surf[1:Nx_local,    1:Ny_local]    .= s[1]  # rank 0 → (1,1)
    global_surf[1:Nx_local,    Ny_local+1:end] .= s[2]  # rank 1 → (1,2)
    global_surf[Nx_local+1:end, 1:Ny_local]    .= s[3]  # rank 2 → (2,1)
    global_surf[Nx_local+1:end, Ny_local+1:end] .= s[4]  # rank 3 → (2,2)

    return global_surf
end

@info "Assembling global surfaces..."

# Nature run
nature_T = assemble_global_surface(nature_dir, "T")
nature_S = assemble_global_surface(nature_dir, "S")
@info "  Nature SST range: ($(minimum(nature_T)), $(maximum(nature_T)))"

# Free run
free_T = assemble_global_surface(free_dir, "T")
free_S = assemble_global_surface(free_dir, "S")
@info "  Free SST range: ($(minimum(free_T)), $(maximum(free_T)))"

# DA ensemble mean (already global in da_final_mean_surface.jld2)
da_file = joinpath(da_dir, "da_final_mean_surface.jld2")
da_T = nothing
if isfile(da_file)
    da_data = jldopen(da_file)
    da_T = da_data["T_surf"]
    da_S = da_data["S_surf"]
    close(da_data)
    @info "  DA mean SST range: ($(minimum(da_T)), $(maximum(da_T)))"
end

# Ocean mask (where nature T > 1)
ocean_mask = abs.(nature_T) .> 1.0
n_ocean = sum(ocean_mask)

# ============================================================================
#                           RMSE
# ============================================================================

T_diff = free_T .- nature_T
S_diff = free_S .- nature_S
rmse_T = sqrt(sum(T_diff[ocean_mask] .^ 2) / n_ocean)
rmse_S = sqrt(sum(S_diff[ocean_mask] .^ 2) / n_ocean)

@info "Free Run vs Nature (day 14):"
@info "  SST RMSE: $(round(rmse_T, digits=4)) C"
@info "  SSS RMSE: $(round(rmse_S, digits=4)) psu"

if !isnothing(da_T)
    da_T_diff = da_T .- nature_T
    da_rmse_T = sqrt(sum(da_T_diff[ocean_mask] .^ 2) / n_ocean)
    @info "DA Mean vs Nature (day 14):"
    @info "  SST RMSE: $(round(da_rmse_T, digits=4)) C"
end

# ============================================================================
#                           PLOTS
# ============================================================================

@info "Generating plots..."

# 1. Final state comparison (Nature vs Free vs DA)
fig = Figure(size=(1800, 600))
Label(fig[0, :], "Final SST Comparison (Day 14, 1/8 deg, 4 GPUs)", fontsize=18, tellwidth=false)

ax1 = Axis(fig[1, 1], title="Nature Run (truth)")
hm1 = heatmap!(ax1, nature_T, colormap=:thermal, colorrange=(10, 32))

ax2 = Axis(fig[1, 2], title="Free Run (no DA)")
hm2 = heatmap!(ax2, free_T, colormap=:thermal, colorrange=(10, 32))

if !isnothing(da_T)
    ax3 = Axis(fig[1, 3], title="DA Ensemble Mean")
    hm3 = heatmap!(ax3, da_T, colormap=:thermal, colorrange=(10, 32))
end

Colorbar(fig[1, 4], hm1, label="C")

save(joinpath(plot_dir, "final_comparison.png"), fig, px_per_unit=2)
@info "Saved final_comparison.png"

# 2. Difference maps
fig2 = Figure(size=(1400, 600))
Label(fig2[0, :], "SST Differences from Nature Run (Day 14)", fontsize=18, tellwidth=false)

T_clim = max(0.5, 3 * std(T_diff[ocean_mask]))

ax1 = Axis(fig2[1, 1], title="Free Run - Nature (RMSE=$(round(rmse_T, digits=3)) C)")
hm1 = heatmap!(ax1, T_diff, colormap=:balance, colorrange=(-T_clim, T_clim))

if !isnothing(da_T)
    ax2 = Axis(fig2[1, 2], title="DA Mean - Nature (RMSE=$(round(da_rmse_T, digits=3)) C)")
    hm2 = heatmap!(ax2, da_T_diff, colormap=:balance, colorrange=(-T_clim, T_clim))
end

Colorbar(fig2[1, 3], hm1, label="C")

save(joinpath(plot_dir, "final_difference.png"), fig2, px_per_unit=2)
@info "Saved final_difference.png"

@info "Post-processing complete!"
@info "Plots saved to: $plot_dir"
