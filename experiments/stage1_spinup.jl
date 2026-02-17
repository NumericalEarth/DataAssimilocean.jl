# Stage 1: Coarse resolution spinup
# Runs for specified duration and saves final state for Stage 2
#
# Usage:
#   julia --project experiments/stage1_spinup.jl [options]
#
# Examples:
#   julia --project experiments/stage1_spinup.jl --test
#   julia --project experiments/stage1_spinup.jl --resolution 0.25 --days 360
#   julia --project experiments/stage1_spinup.jl --resolution 0.125 --days 720 --dt 5 --no-animation

include("irma_utils.jl")

using ArgParse

function parse_commandline()
    s = ArgParseSettings(description = "Stage 1: Coarse resolution ocean spinup simulation")

    @add_arg_table! s begin
        "--test"
            help = "Run a 1-day test instead of full spinup"
            action = :store_true
        "--resolution"
            help = "Grid resolution in degrees (e.g. 0.25 for 1/4°, 0.125 for 1/8°)"
            arg_type = Float64
            default = 0.125
        "--days"
            help = "Number of spinup days (ignored if --test is set)"
            arg_type = Int
            default = 720
        "--dt"
            help = "Time step in minutes"
            arg_type = Float64
            default = 5.0
        "--output-dir"
            help = "Output directory (default: auto-generated from resolution)"
            arg_type = String
            default = ""
        "--no-animation"
            help = "Skip the post-simulation animation"
            action = :store_true
        "--cpu"
            help = "Run on CPU instead of GPU"
            action = :store_true
    end

    return parse_args(s)
end

args = parse_commandline()

# Configuration
arch = args["cpu"] ? CPU() : GPU()
resolution = args["resolution"]

# Simulation timing
if args["test"]
    spinup_days = 1
    @info "TEST MODE: Running 1-day spinup"
else
    spinup_days = args["days"]
    @info "PRODUCTION: Running $(spinup_days)-day spinup"
end

# Hurricane Irma dates
# Full simulation: June 26 - Sept 8, 2017
# Stage 1 covers the spinup period before hurricane analysis
hurricane_start = DateTime(2017, 8, 25)  # When we want high-res to start
start_date = hurricane_start - Day(spinup_days)
end_date = hurricane_start

@info "Stage 1: Spinup from $start_date to $end_date"

# Output directory (includes resolution for traceability)
resolution_tag = @sprintf("%.4gdeg", resolution)  # e.g. "0.5deg" or "0.25deg"
output_dir = isempty(args["output-dir"]) ? "stage1_output_$(resolution_tag)" : args["output-dir"]
mkpath(output_dir)

# === Grid Setup ===
@info "Creating $(resolution)° grid..."
grid = create_grid(arch, resolution)
Nx, Ny, Nz = size(grid)
@info "Grid size: $(Nx) × $(Ny) × $(Nz)"

# Bathymetry
grid = load_or_compute_bathymetry(grid, output_dir)

# === Ocean Model ===
@info "Creating ocean simulation..."
ocean = create_ocean_simulation(grid, start_date, end_date; with_restoring=true)

# Initial conditions from EN4
@info "Setting initial conditions from EN4..."
set!(ocean.model,
     T = Metadatum(:temperature; date=start_date, dataset=EN4Monthly()),
     S = Metadatum(:salinity;    date=start_date, dataset=EN4Monthly()))

# === Atmosphere ===
@info "Setting up JRA55 atmosphere..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

# === Coupled Model ===
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# === Simulation ===
Δt = args["dt"] * minutes
stop_time = spinup_days * days
simulation = Simulation(coupled_model; Δt, stop_time)

# Progress callback
add_progress_callback!(simulation, start_date)

# Output writers (reduced frequency for spinup to save space)
setup_output_writers!(ocean, coupled_model, output_dir;
                      surface_interval = 6hour,  # Less frequent for spinup
                      flux_interval = 6hour)

# === Run ===
@info "Running Stage 1 spinup simulation..."
@info "Stop time: $(stop_time / days) days"

run!(simulation)

# === Save Final State ===
state_file = joinpath(output_dir, "final_state.jld2")
save_model_state(ocean, state_file)

# ## Animation of spinup

if args["no-animation"]
    @info "Skipping animation (--no-animation flag set)"
    @info "Stage 1 complete!"
    @info "Final state saved to: $state_file"
    @info "Ready for Stage 2 initialization"
    exit(0)
end

using CairoMakie
using Oceananigans.OutputReaders: FieldTimeSeries

surface_file = joinpath(output_dir, "surface_fields.jld2")

Tt = FieldTimeSeries(surface_file, "T")
St = FieldTimeSeries(surface_file, "S")
ut = FieldTimeSeries(surface_file, "u")
vt = FieldTimeSeries(surface_file, "v")
et = FieldTimeSeries(surface_file, "e")

times = Tt.times
Nt = length(times)

# Pick one frame per day (closest snapshot to each 24h mark)
daily_seconds = days .* (1:spinup_days)
frame_indices = Int[]
for target in daily_seconds
    _, idx = findmin(abs.(times .- target))
    push!(frame_indices, idx)
end
unique!(frame_indices)

@info "Animating $(length(frame_indices)) daily frames out of $Nt total snapshots"

# Observable frame index drives all panels via @lift
n = Observable(1)

Tn = @lift interior(Tt[$n], :, :, 1)
Sn = @lift interior(St[$n], :, :, 1)
en = @lift max.(interior(et[$n], :, :, 1), 0)

speed_n = @lift begin
    u_surf = interior(ut[$n], :, :, 1)
    v_surf = interior(vt[$n], :, :, 1)
    u_c = 0.5 .* (u_surf[1:end-1, :] .+ u_surf[2:end, :])
    v_c = 0.5 .* (v_surf[:, 1:end-1] .+ v_surf[:, 2:end])
    mi = min(size(u_c, 1), size(v_c, 1))
    mj = min(size(u_c, 2), size(v_c, 2))
    sqrt.(u_c[1:mi, 1:mj].^2 .+ v_c[1:mi, 1:mj].^2)
end

title = @lift begin
    day_num = round(Int, times[$n] / days)
    current_date = start_date + Day(day_num)
    "Stage 1 Spinup — Day $day_num ($(Dates.format(current_date, "yyyy-mm-dd")))"
end

# Coordinates
lons = collect(range(DOMAIN_LONGITUDE[1], DOMAIN_LONGITUDE[2], length=Nx))
lats = collect(range(DOMAIN_LATITUDE[1], DOMAIN_LATITUDE[2], length=Ny))

fig = Figure(size=(1200, 800))

fig[0, :] = Label(fig, title, fontsize=20, tellwidth=false)

ax1 = Axis(fig[1, 1], xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, lons, lats, Tn, colormap=:thermal, colorrange=(10, 32))
Colorbar(fig[1, 2], hm1, label="SST (°C)")

ax2 = Axis(fig[1, 3], xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, lons, lats, Sn, colormap=:viridis, colorrange=(33, 37))
Colorbar(fig[1, 4], hm2, label="SSS (psu)")

ax3 = Axis(fig[2, 1], xlabel="Longitude", ylabel="Latitude")
hm3 = heatmap!(ax3, lons[1:Nx-1], lats[1:Ny-1], speed_n, colormap=:speed, colorrange=(0, 1))
Colorbar(fig[2, 2], hm3, label="Speed (m/s)")

ax4 = Axis(fig[2, 3], xlabel="Longitude", ylabel="Latitude")
hm4 = heatmap!(ax4, lons, lats, en, colormap=:inferno, colorrange=(0, 1e-3))
Colorbar(fig[2, 4], hm4, label="TKE (m²/s²)")

CairoMakie.record(fig, joinpath(output_dir, "spinup_animation.mp4"), frame_indices, framerate=12) do nn
    n[] = nn
end

save(joinpath(output_dir, "final_state.png"), fig)

@info "Stage 1 complete!"
@info "Final state saved to: $state_file"
@info "Animation saved to: $(joinpath(output_dir, "spinup_animation.mp4"))"
@info "Ready for Stage 2 initialization"
