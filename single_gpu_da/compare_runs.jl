# Compare nature run vs free run
#
# Generates:
#   1. Side-by-side SST animation (nature vs free)
#   2. RMSE time series plot
#   3. Final-time difference maps
#
# Usage:
#   julia --project single_gpu_da/compare_runs.jl NATURE_DIR FREE_DIR OUTPUT_DIR

using CairoMakie
using JLD2
using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries
using Statistics
using Dates
using Printf

nature_dir = ARGS[1]
free_dir = ARGS[2]
output_dir = ARGS[3]
mkpath(output_dir)

start_date = DateTime(2017, 8, 25)

@info "Loading surface fields..."

nature_T = FieldTimeSeries(joinpath(nature_dir, "surface_fields.jld2"), "T")
nature_S = FieldTimeSeries(joinpath(nature_dir, "surface_fields.jld2"), "S")
free_T = FieldTimeSeries(joinpath(free_dir, "surface_fields.jld2"), "T")
free_S = FieldTimeSeries(joinpath(free_dir, "surface_fields.jld2"), "S")

times = nature_T.times
Nt = length(times)
@info "Loaded $Nt time snapshots"

# === RMSE Time Series ===
@info "Computing RMSE time series..."

rmse_T = Float64[]
rmse_S = Float64[]
days_vec = Float64[]

for n in 1:Nt
    T_n = interior(nature_T[n], :, :, 1)
    T_f = interior(free_T[n], :, :, 1)
    S_n = interior(nature_S[n], :, :, 1)
    S_f = interior(free_S[n], :, :, 1)

    T_diff = T_n .- T_f
    S_diff = S_n .- S_f

    push!(rmse_T, sqrt(mean(T_diff .^ 2)))
    push!(rmse_S, sqrt(mean(S_diff .^ 2)))
    push!(days_vec, times[n] / (24 * 3600))
end

@info "Final SST RMSE: $(rmse_T[end]) °C"
@info "Final SSS RMSE: $(rmse_S[end]) psu"

# Save RMSE data
jldsave(joinpath(output_dir, "rmse_data.jld2");
        rmse_T, rmse_S, days=days_vec, times)

# RMSE time series plot
fig_rmse = Figure(size=(800, 400))
ax1 = Axis(fig_rmse[1, 1], xlabel="Day", ylabel="RMSE (°C)",
           title="SST RMSE: Free Run vs Nature Run")
lines!(ax1, days_vec, rmse_T, linewidth=2, color=:red, label="SST")
axislegend(ax1)

ax2 = Axis(fig_rmse[1, 2], xlabel="Day", ylabel="RMSE (psu)",
           title="SSS RMSE: Free Run vs Nature Run")
lines!(ax2, days_vec, rmse_S, linewidth=2, color=:blue, label="SSS")
axislegend(ax2)

save(joinpath(output_dir, "rmse_timeseries.png"), fig_rmse, px_per_unit=2)
@info "RMSE plot saved"

# === Final Difference Maps ===
@info "Generating final difference maps..."

T_nature_final = interior(nature_T[Nt], :, :, 1)
T_free_final = interior(free_T[Nt], :, :, 1)
S_nature_final = interior(nature_S[Nt], :, :, 1)
S_free_final = interior(free_S[Nt], :, :, 1)

T_diff_final = T_free_final .- T_nature_final
S_diff_final = S_free_final .- S_nature_final

fig_diff = Figure(size=(1200, 800))
Label(fig_diff[0, :], "Final State Differences (Free Run - Nature Run, Day 14)",
      fontsize=18, tellwidth=false)

ax1 = Axis(fig_diff[1, 1], title="Nature Run SST", xlabel="i", ylabel="j")
hm1 = heatmap!(ax1, T_nature_final, colormap=:thermal, colorrange=(10, 32))
Colorbar(fig_diff[1, 2], hm1, label="°C")

ax2 = Axis(fig_diff[1, 3], title="Free Run SST", xlabel="i", ylabel="j")
hm2 = heatmap!(ax2, T_free_final, colormap=:thermal, colorrange=(10, 32))
Colorbar(fig_diff[1, 4], hm2, label="°C")

ax3 = Axis(fig_diff[2, 1], title="SST Difference", xlabel="i", ylabel="j")
T_clim = max(abs(quantile(vec(T_diff_final), 0.01)), abs(quantile(vec(T_diff_final), 0.99)))
hm3 = heatmap!(ax3, T_diff_final, colormap=:balance, colorrange=(-T_clim, T_clim))
Colorbar(fig_diff[2, 2], hm3, label="°C")

ax4 = Axis(fig_diff[2, 3], title="SSS Difference", xlabel="i", ylabel="j")
S_clim = max(abs(quantile(vec(S_diff_final), 0.01)), abs(quantile(vec(S_diff_final), 0.99)))
hm4 = heatmap!(ax4, S_diff_final, colormap=:balance, colorrange=(-S_clim, S_clim))
Colorbar(fig_diff[2, 4], hm4, label="psu")

save(joinpath(output_dir, "final_difference.png"), fig_diff, px_per_unit=2)
@info "Difference maps saved"

# === Side-by-side Animation ===
@info "Generating comparison animation..."

try
    n = Observable(1)

    Tn_nature = @lift interior(nature_T[$n], :, :, 1)
    Tn_free = @lift interior(free_T[$n], :, :, 1)
    Tn_diff = @lift interior(free_T[$n], :, :, 1) .- interior(nature_T[$n], :, :, 1)

    title = @lift begin
        day_num = times[$n] / (24 * 3600)
        "Day $(round(day_num, digits=1))"
    end

    fig_anim = Figure(size=(1400, 500))
    Label(fig_anim[0, :], title, fontsize=18, tellwidth=false)

    ax1 = Axis(fig_anim[1, 1], title="Nature Run SST", xlabel="i", ylabel="j")
    hm1 = heatmap!(ax1, Tn_nature, colormap=:thermal, colorrange=(10, 32))
    Colorbar(fig_anim[1, 2], hm1, label="°C")

    ax2 = Axis(fig_anim[1, 3], title="Free Run SST", xlabel="i", ylabel="j")
    hm2 = heatmap!(ax2, Tn_free, colormap=:thermal, colorrange=(10, 32))
    Colorbar(fig_anim[1, 4], hm2, label="°C")

    ax3 = Axis(fig_anim[1, 5], title="Difference (Free - Nature)", xlabel="i", ylabel="j")
    hm3 = heatmap!(ax3, Tn_diff, colormap=:balance, colorrange=(-2, 2))
    Colorbar(fig_anim[1, 6], hm3, label="°C")

    CairoMakie.record(fig_anim, joinpath(output_dir, "comparison_animation.mp4"),
                       1:Nt, framerate=8) do nn
        n[] = nn
    end

    @info "Comparison animation saved"
catch e
    @warn "Animation failed (non-fatal)" exception=(e, catch_backtrace())
end

@info "Comparison complete!"
@info "  Final SST RMSE: $(rmse_T[end]) °C"
@info "  Final SSS RMSE: $(rmse_S[end]) psu"
