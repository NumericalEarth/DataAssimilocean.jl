# Stage 2: High resolution hurricane analysis (1/12°)
# Loads state from Stage 1 and runs high-resolution simulation
#
# Usage:
#   julia --project experiments/stage2_highres.jl [--test]
#
# The --test flag runs a 6-hour test instead of full 14-day analysis

include("irma_utils.jl")

using CairoMakie
using Dates: Hour

# Parse command line args
test_mode = "--test" in ARGS

# Configuration
arch = GPU()
source_resolution = 1/4    # degrees (from Stage 1)
target_resolution = 1/12   # degrees (high resolution)

# Simulation timing
if test_mode
    analysis_hours = 6  # Just a few hours for testing
    @info "TEST MODE: Running $(analysis_hours)-hour analysis"
else
    analysis_days = 14
    @info "PRODUCTION: Running $(analysis_days)-day analysis"
end

# Hurricane Irma dates
# Stage 2 starts where Stage 1 ends
hurricane_start = DateTime(2017, 8, 25)
start_date = hurricane_start
if test_mode
    end_date = hurricane_start + Hour(analysis_hours)
else
    end_date = hurricane_start + Day(analysis_days)
end

@info "Stage 2: High-res analysis from $start_date to $end_date"

# Directories
input_dir = "stage1_output"
output_dir = "stage2_output"
mkpath(output_dir)

# === Load Stage 1 State ===
state_file = joinpath(input_dir, "final_state.jld2")
if !isfile(state_file)
    error("Stage 1 output not found: $state_file. Run stage1_spinup.jl first.")
end
source_state = load_model_state(state_file)

# === Grid Setup ===
@info "Creating $(target_resolution)° grid..."
grid = create_grid(arch, target_resolution)
Nx, Ny, Nz = size(grid)
@info "Grid size: $(Nx) × $(Ny) × $(Nz)"

# Bathymetry (will be computed at new resolution if not cached)
grid = load_or_compute_bathymetry(grid, output_dir)

# === Ocean Model ===
# High resolution needs more substeps for barotropic stability
@info "Creating ocean simulation..."
ocean = create_ocean_simulation(grid, start_date, end_date; with_restoring=true, substeps=50)

# === Initial Conditions ===
# Use EN4 data directly at high resolution rather than interpolating from Stage 1
# This avoids land-ocean masking issues during interpolation
@info "Setting initial conditions from EN4 at high resolution..."
set!(ocean.model,
     T = Metadatum(:temperature; date=start_date, dataset=EN4Monthly()),
     S = Metadatum(:salinity;    date=start_date, dataset=EN4Monthly()))
@info "Initial conditions set from EN4"

# === Atmosphere ===
@info "Setting up JRA55 atmosphere..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

# === Coupled Model ===
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# === Simulation ===
# Smaller time step for high resolution (1/12° needs ~60s Δt for stability)
Δt = 60  # seconds
if test_mode
    stop_time = analysis_hours * hour
else
    stop_time = analysis_days * days
end
simulation = Simulation(coupled_model; Δt, stop_time)

# Progress callback
add_progress_callback!(simulation, start_date)

# Output writers (high frequency for analysis)
setup_output_writers!(ocean, coupled_model, output_dir;
                      surface_interval = 1hour,    # Higher frequency for analysis
                      flux_interval = 1hour)

# === Run ===
@info "Running Stage 2 high-resolution simulation..."
@info "Stop time: $(stop_time / days) days"

run!(simulation)

# === Save Final State ===
state_file = joinpath(output_dir, "final_state.jld2")
save_model_state(ocean, state_file)

# === Visualization ===
@info "Creating final state visualization..."

fig = Figure(size=(1200, 800))

# Get surface data
T_surf = Array(interior(ocean.model.tracers.T, :, :, Nz))
S_surf = Array(interior(ocean.model.tracers.S, :, :, Nz))
u_surf = Array(interior(ocean.model.velocities.u, :, :, Nz))
v_surf = Array(interior(ocean.model.velocities.v, :, :, Nz))

# Interpolate u,v to cell centers for speed
u_center = 0.5 * (u_surf[1:end-1, :] .+ u_surf[2:end, :])
v_center = 0.5 * (v_surf[:, 1:end-1] .+ v_surf[:, 2:end])
min_i = min(size(u_center, 1), size(v_center, 1))
min_j = min(size(u_center, 2), size(v_center, 2))
speed = sqrt.(u_center[1:min_i, 1:min_j].^2 .+ v_center[1:min_i, 1:min_j].^2)

# Coordinates
lons = range(DOMAIN_LONGITUDE[1], DOMAIN_LONGITUDE[2], length=Nx)
lats = range(DOMAIN_LATITUDE[1], DOMAIN_LATITUDE[2], length=Ny)

ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, collect(lons), collect(lats), T_surf; colormap=:thermal, colorrange=(10, 32))
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3]; title="SSS (psu)", xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, collect(lons), collect(lats), S_surf; colormap=:viridis, colorrange=(33, 37))
Colorbar(fig[1, 4], hm2)

ax3 = Axis(fig[2, 1]; title="Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude")
hm3 = heatmap!(ax3, collect(lons)[1:min_i], collect(lats)[1:min_j], speed; colormap=:speed, colorrange=(0, 2))
Colorbar(fig[2, 2], hm3)

# TKE
e_surf = Array(interior(ocean.model.tracers.e, :, :, Nz))
ax4 = Axis(fig[2, 3]; title="Surface TKE (m²/s²)", xlabel="Longitude", ylabel="Latitude")
hm4 = heatmap!(ax4, collect(lons), collect(lats), max.(e_surf, 0); colormap=:inferno, colorrange=(0, 1e-3))
Colorbar(fig[2, 4], hm4)

Label(fig[0, :], "Stage 2 Final State: $(Dates.format(end_date, "yyyy-mm-dd HH:MM")) (1/12° resolution)", fontsize=20)

save(joinpath(output_dir, "final_state.png"), fig)

@info "Stage 2 complete!"
@info "Final state saved to: $(joinpath(output_dir, "final_state.jld2"))"
