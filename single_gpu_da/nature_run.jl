# Nature run: 14-day forward simulation from spinup end state
#
# This produces the "truth" for the DA experiment.
# Saves surface fields at 6-hour intervals for observations and animation.
#
# Usage:
#   julia --project single_gpu_da/nature_run.jl SPINUP_DIR OUTPUT_DIR

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

spinup_dir = ARGS[1]
output_dir = ARGS[2]
mkpath(output_dir)

# Configuration
arch = GPU()
resolution = 0.25
nature_days = 14
Δt = 5minutes

# Nature run dates (starts where spinup ends)
start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(nature_days)

@info "Nature run: $start_date to $end_date ($nature_days days)"

# Grid (reuse cached bathymetry from spinup)
grid = create_grid(arch, resolution)
grid = load_or_compute_bathymetry(grid, spinup_dir)
Nx, Ny, Nz = size(grid)

# Ocean model
ocean = create_ocean_simulation(grid, start_date, end_date; with_restoring=true)

# Load spinup state
@info "Loading spinup state..."
state = load_model_state(joinpath(spinup_dir, "final_state.jld2"))
set!(ocean.model, T=state.T, S=state.S, e=state.e, u=state.u, v=state.v)

# Handle w and eta separately (different grid locations)
copyto!(interior(ocean.model.velocities.w), state.w)
copyto!(interior(ocean.model.free_surface.η), state.η)

# Atmosphere
@info "Setting up JRA55 atmosphere for nature run period..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

# Coupled model
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=nature_days * days)

# Progress
add_progress_callback!(simulation, start_date)

# Output: surface fields every 6 hours
setup_output_writers!(ocean, coupled_model, output_dir;
                      surface_interval=6hour, flux_interval=6hour)

# Run
@info "Running nature run..."
run!(simulation)

# Save final state
save_model_state(ocean, joinpath(output_dir, "final_state.jld2"))

# === Animation ===
@info "Generating nature run animation..."

using CairoMakie
using Oceananigans.OutputReaders: FieldTimeSeries

try
    surface_file = joinpath(output_dir, "surface_fields.jld2")
    Tt = FieldTimeSeries(surface_file, "T")
    times = Tt.times
    Nt = length(times)

    n = Observable(1)
    Tn = @lift interior(Tt[$n], :, :, 1)

    title = @lift begin
        day_num = times[$n] / (24 * 3600)
        current_date = start_date + Day(round(Int, day_num))
        "Nature Run SST — Day $(round(day_num, digits=1)) ($current_date)"
    end

    fig = Figure(size=(900, 500))
    Label(fig[0, :], title, fontsize=18, tellwidth=false)
    ax = Axis(fig[1, 1], xlabel="i", ylabel="j")
    hm = heatmap!(ax, Tn, colormap=:thermal, colorrange=(10, 32))
    Colorbar(fig[1, 2], hm, label="SST (°C)")

    CairoMakie.record(fig, joinpath(output_dir, "nature_run_animation.mp4"),
                       1:Nt, framerate=8) do nn
        n[] = nn
    end

    # Save final frame as PNG
    n[] = Nt
    save(joinpath(output_dir, "nature_run_final.png"), fig)
    @info "Animation saved to $(joinpath(output_dir, "nature_run_animation.mp4"))"
catch e
    @warn "Animation failed (non-fatal)" exception=(e, catch_backtrace())
end

@info "Nature run complete!"
