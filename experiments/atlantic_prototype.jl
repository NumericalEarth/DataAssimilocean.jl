# Atlantic Basin Prototype Simulation (1/4°, GPU)
# Hurricane Irma case study - EN4-initialized, JRA-55 forced
# Includes lateral restoring at north/south boundaries
# 60-day spinup + 14-day hurricane period = 74 days total
# Saves atmosphere wind and surface fluxes for animation

using Oceananigans
using Oceananigans.Units: minutes, days, meters
using Oceananigans.OutputReaders: Linear
using ClimaOcean
using ClimaOcean.EN4: EN4Monthly
using ClimaOcean.JRA55: MultiYearJRA55
using CairoMakie
using Printf
using Dates
using JLD2

# Define hour explicitly to avoid conflict between Dates.hour and Oceananigans.Units.hour
const hour = Oceananigans.Units.hour

arch = GPU()

# Domain: Atlantic 100°W–0°E, -10°S–60°N
longitude = (-100, 0)
latitude  = (-10, 60)

Lλ = longitude[2] - longitude[1]
Lφ = latitude[2] - latitude[1]

# 1/4° resolution
Δφ = 1/4
Nx = Int(Lλ / Δφ)
Ny = Int(Lφ / Δφ)
Nz = 50
depth = 5000meters

z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4)

grid = LatitudeLongitudeGrid(arch; latitude, longitude, z,
                             size = (Nx, Ny, Nz), halo = (7, 7, 7))   

# Output directory (create early for bathymetry cache)
output_dir = "atlantic_prototype_output"
mkpath(output_dir)

# Bathymetry - cache to avoid expensive recomputation
bathymetry_file = joinpath(output_dir, "bathymetry_$(Nx)x$(Ny).jld2")
if isfile(bathymetry_file)
    @info "Loading cached bathymetry from $bathymetry_file"
    bottom_height_data = jldopen(bathymetry_file)["bottom_height"]
    bottom_height = Field{Center, Center, Nothing}(grid)
    set!(bottom_height, bottom_height_data)
else
    @info "Computing bathymetry (this may take a few minutes)..."
bottom_height = regrid_bathymetry(grid;
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)
    @info "Saving bathymetry to $bathymetry_file"
    jldsave(bathymetry_file; bottom_height=Array(interior(bottom_height)))
end

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Lateral restoring at north and south boundaries (5° sponge width)
sponge_width = 5  # degrees
restoring_rate = 1 / (7days)  # 7-day timescale

restoring_mask = LinearlyTaperedPolarMask(;
    northern = (latitude[2] - sponge_width, latitude[2]),  # (55, 60)
    southern = (latitude[1], latitude[1] + sponge_width),  # (-10, -5)
    z = (-depth, 0))

# Hurricane Irma timeline
spinup_days = 60
analysis_days = 14
run_duration = spinup_days + analysis_days  # 74 days total

start_date = DateTime(2017, 6, 26)
end_date = start_date + Day(run_duration)
analysis_start = start_date + Day(spinup_days)

@info "Simulation period: $start_date to $end_date ($run_duration days)"
@info "Analysis period starts: $analysis_start"

# Create restoring forcing for T and S
T_restoring_metadata = Metadata(:temperature;
                                 dates = start_date:Month(1):end_date,
                                 dataset = EN4Monthly())

S_restoring_metadata = Metadata(:salinity;
                                 dates = start_date:Month(1):end_date,
                                 dataset = EN4Monthly())

T_restoring = DatasetRestoring(T_restoring_metadata, arch;
                               rate = restoring_rate,
                               mask = restoring_mask,
                               time_indexing = Linear())

S_restoring = DatasetRestoring(S_restoring_metadata, arch;
                               rate = restoring_rate,
                               mask = restoring_mask,
                               time_indexing = Linear())

# Ocean model with restoring forcing
free_surface = SplitExplicitFreeSurface(grid; substeps=30)
ocean = ocean_simulation(grid;
                         momentum_advection = WENOVectorInvariant(order=5),
                         tracer_advection = WENO(order=5),
                         free_surface,
                         forcing = (T = T_restoring, S = S_restoring))

# Initial conditions from EN4
set!(ocean.model,
     T = Metadatum(:temperature; date=start_date, dataset=EN4Monthly()),
     S = Metadatum(:salinity;    date=start_date, dataset=EN4Monthly()))

# Atmospheric forcing
radiation = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch;
                                       dataset = MultiYearJRA55(),
                                       start_date = start_date,
                                       end_date = end_date,
                                       time_indexing = Linear(),
                                       backend = JRA55NetCDFBackend(41),
                                       include_rivers_and_icebergs = false)

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# Simulation
Δt = 10minutes
stop_time = run_duration * days
simulation = Simulation(coupled_model; Δt, stop_time)

# Progress callback
wall_time = Ref(time_ns())

function progress(sim)
    ocean_model = sim.model.ocean
    u, v, w = ocean_model.model.velocities
    T, S = ocean_model.model.tracers.T, ocean_model.model.tracers.S
    elapsed = 1e-9 * (time_ns() - wall_time[])
    current_date = start_date + Dates.Second(round(Int, time(sim)))

    @info @sprintf("Iter %d, t=%s (%s), Δt=%s, wall=%s | u=(%.2e,%.2e,%.2e) | T=(%.1f,%.1f) S=(%.1f,%.1f)",
                   iteration(sim), prettytime(sim), current_date, prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), minimum(S), maximum(S))
    wall_time[] = time_ns()
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# === Output Writers ===

# Ocean surface fields (T, S, u, v, w, e)
surface_fields = merge(ocean.model.tracers, ocean.model.velocities)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, surface_fields;
                                            schedule = TimeInterval(1hour),
                                            filename = joinpath(output_dir, "surface_fields"),
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

# Sea surface height
ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η=ocean.model.free_surface.η);
                                                 schedule = TimeInterval(1hour),
                                                 filename = joinpath(output_dir, "sea_surface_height"),
                                                 overwrite_existing = true)

# Atmosphere-ocean interface fluxes (interpolated to ocean grid)
# Access via coupled_model.interfaces
interface = coupled_model.interfaces.atmosphere_ocean_interface
exchanger = coupled_model.interfaces.exchanger

# Atmosphere state on ocean grid
atm_state = exchanger.exchange_atmosphere_state
atm_u = atm_state.u  # Atmosphere u-velocity on ocean grid
atm_v = atm_state.v  # Atmosphere v-velocity on ocean grid

# Interface fluxes
fluxes = interface.fluxes
τx = fluxes.x_momentum      # x-momentum flux (wind stress)
τy = fluxes.y_momentum      # y-momentum flux
Qh_sensible = fluxes.sensible_heat
Qh_latent = fluxes.latent_heat
Qs = fluxes.downwelling_shortwave

# Create output writer for atmosphere/flux fields
flux_outputs = (;
    atm_u = atm_u,
    atm_v = atm_v,
    tau_x = τx,
    tau_y = τy,
    Qh_sensible = Qh_sensible,
    Qh_latent = Qh_latent,
    Qs = Qs
)

ocean.output_writers[:fluxes] = JLD2Writer(ocean.model, flux_outputs;
                                           schedule = TimeInterval(1hour),
                                           filename = joinpath(output_dir, "atmosphere_fluxes"),
                                           overwrite_existing = true)

# === Run ===
# For testing, uncomment the following line:
# simulation.stop_iteration = 100

@info "Running Atlantic prototype simulation for Hurricane Irma case..."
@info "Start date: $start_date"
@info "End date: $end_date"
@info "Duration: $run_duration days (spinup: $spinup_days, analysis: $analysis_days)"
@info "Lateral restoring: $(sponge_width)° sponge at N/S boundaries"

run!(simulation)
@info "Simulation complete! Iterations: $(iteration(simulation))"

# === 4-Panel Animation: SST, Surface Speed, Wind Speed, TKE ===

surface_file = joinpath(output_dir, "surface_fields.jld2")
flux_file = joinpath(output_dir, "atmosphere_fluxes.jld2")

@info "Creating animation from saved data..."

# Load surface data
file = jldopen(surface_file)
time_keys = sort([parse(Float64, k) for k in keys(file["timeseries/t"])])

# Build mapping from key to actual time value (in seconds)
key_to_time = Dict{Int, Float64}()
for k in time_keys
    key_to_time[Int(k)] = file["timeseries/t/$(Int(k))"]
end

# Get actual times
actual_times = [key_to_time[Int(k)] for k in time_keys]

@info "Time span: $(actual_times[1]/86400) to $(actual_times[end]/86400) days"
@info "Number of snapshots: $(length(time_keys))"

# Check if flux file exists
has_flux_file = isfile(flux_file)
@info "Flux file available: $has_flux_file"

# Get grid dimensions from first snapshot
first_key = Int(time_keys[1])
T_example = file["timeseries/T/$first_key"]
Nx_out, Ny_out = size(T_example)[1:2]

# Load flux file if available
flux_jld = has_flux_file ? jldopen(flux_file) : nothing

# Create coordinate arrays for proper axis labels
ocean_lons = range(longitude[1], longitude[2], length=Nx_out)
ocean_lats = range(latitude[1], latitude[2], length=Ny_out)

# Check for TKE
has_tke = haskey(file, "timeseries/e")
@info "TKE available in output: $has_tke"

# Create figure
fig = Figure(size=(1600, 1200), fontsize=14)

# Title
title_label = Label(fig[0, :], ""; fontsize=20, tellwidth=false)

# Panel 1: SST
ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm1 = heatmap!(ax1, collect(ocean_lons), collect(ocean_lats), zeros(Nx_out, Ny_out);
               colormap=:thermal, colorrange=(15, 32), nan_color=:gray)
Colorbar(fig[1, 2], hm1)

# Panel 2: Sea Surface Speed
ax2 = Axis(fig[1, 3]; title="Sea Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm2 = heatmap!(ax2, collect(ocean_lons), collect(ocean_lats), zeros(Nx_out, Ny_out);
               colormap=:viridis, colorrange=(0, 1.5), nan_color=:gray)
Colorbar(fig[1, 4], hm2)

# Panel 3: Wind Speed
ax3 = Axis(fig[2, 1]; title="Wind Speed (m/s)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm3 = heatmap!(ax3, collect(ocean_lons), collect(ocean_lats), zeros(Nx_out, Ny_out);
               colormap=:plasma, colorrange=(0, 25), nan_color=:gray)
Colorbar(fig[2, 2], hm3)

# Panel 4: Surface TKE (colorbar will be updated dynamically)
ax4 = Axis(fig[2, 3]; title="Surface TKE (m²/s²)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm4 = heatmap!(ax4, collect(ocean_lons), collect(ocean_lats), zeros(Nx_out, Ny_out);
               colormap=:inferno, colorrange=(0, 1e-3), nan_color=:gray)
Colorbar(fig[2, 4], hm4)

# Record animation
@info "Recording animation with $(length(time_keys)) frames..."
animation_file = joinpath(output_dir, "sst_speed_wind_tke_animation.mp4")

record(fig, animation_file, enumerate(time_keys); framerate=12) do (frame_idx, key)
    key_int = Int(key)
    actual_time = key_to_time[key_int]
    
    # Load ocean data
    T_data = file["timeseries/T/$key_int"][:, :, 1]
    u_raw = file["timeseries/u/$key_int"][:, :, 1]
    v_raw = file["timeseries/v/$key_int"][:, :, 1]
    
    # Interpolate staggered velocities to cell centers
    u_center = 0.5 * (u_raw[1:end-1, :] .+ u_raw[2:end, :])
    v_center = 0.5 * (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
    speed_data = sqrt.(u_center.^2 .+ v_center.^2)
    
    # TKE (clamp to non-negative)
    if has_tke
        e_data = file["timeseries/e/$key_int"][:, :, 1]
        tke_data = max.(e_data, 0.0f0)
    else
        tke_data = zeros(Float32, Nx_out, Ny_out)
    end
    
    # Wind speed from atmosphere FTS with correct time offset
    # The FTS times are absolute (seconds since JRA55 epoch 1958-01-01)
    jra55_epoch = DateTime(1958, 1, 1)
    time_offset = (start_date - jra55_epoch).value / 1000  # milliseconds to seconds
    atm_query_time = Oceananigans.OutputReaders.Time(time_offset + actual_time)
    
    ua_fts = atmosphere.velocities.u
    va_fts = atmosphere.velocities.v
    ua_snapshot = ua_fts[atm_query_time]
    va_snapshot = va_fts[atm_query_time]
    ua_atm = Array(interior(ua_snapshot, :, :, 1))
    va_atm = Array(interior(va_snapshot, :, :, 1))
    wind_atm = sqrt.(ua_atm.^2 .+ va_atm.^2)
    
    # Interpolate to ocean grid (nearest neighbor)
    atm_grid = ua_fts.grid
    atm_lons = Array(atm_grid.λᶜᵃᵃ[1:atm_grid.Nx])
    atm_lats = Array(atm_grid.φᵃᶜᵃ[1:atm_grid.Ny])
    
    wind_data = zeros(Float32, Nx_out, Ny_out)
    for j in 1:Ny_out
        ocean_lat = latitude[1] + (j - 0.5) * (latitude[2] - latitude[1]) / Ny_out
        aj = argmin(abs.(atm_lats .- ocean_lat))
        for i in 1:Nx_out
            ocean_lon = longitude[1] + (i - 0.5) * (longitude[2] - longitude[1]) / Nx_out
            ocean_lon_360 = ocean_lon < 0 ? ocean_lon + 360 : ocean_lon
            ai = argmin(abs.(atm_lons .- ocean_lon_360))
            wind_data[i, j] = wind_atm[ai, aj]
        end
    end
    
    # Replace NaN for plotting
    T_plot = replace(T_data, NaN => NaN32)
    speed_plot = replace(speed_data, NaN => NaN32)
    wind_plot = replace(wind_data, NaN => NaN32)
    tke_plot = replace(tke_data, NaN => NaN32)
    
    # Update heatmaps
    hm1[3] = T_plot
    hm2[3] = speed_plot
    hm3[3] = wind_plot
    hm4[3] = tke_plot
    
    # Update title with current date
    current_date = start_date + Dates.Second(round(Int, actual_time))
    day_num = round(actual_time / 86400, digits=1)
    title_label.text = "Hurricane Irma - $(Dates.format(current_date, "yyyy-mm-dd HH:MM")) (Day $day_num)"
    
    if frame_idx % 50 == 0
        @info "  Frame $frame_idx / $(length(time_keys))"
    end
end

close(file)
if flux_jld !== nothing
    close(flux_jld)
end

@info "Animation saved to: $animation_file"

# Save final state as PNG
T_cpu = Array(interior(ocean.model.tracers.T, :, :, grid.Nz))
u_raw = Array(interior(ocean.model.velocities.u, :, :, grid.Nz))
v_raw = Array(interior(ocean.model.velocities.v, :, :, grid.Nz))
η_cpu = dropdims(Array(interior(ocean.model.free_surface.η)); dims=3)

u_center = 0.5 * (u_raw[1:end-1, :] .+ u_raw[2:end, :])
v_center = 0.5 * (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
speed_cpu = sqrt.(u_center.^2 .+ v_center.^2)

fig_final = Figure(size=(1400, 500), fontsize=14)

ax1 = Axis(fig_final[1, 1]; title="Final SST (°C)", xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, collect(ocean_lons), collect(ocean_lats), T_cpu; colormap=:thermal, nan_color=:gray)
Colorbar(fig_final[1, 2], hm1)

ax2 = Axis(fig_final[1, 3]; title="Final Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, collect(ocean_lons)[1:size(speed_cpu,1)], collect(ocean_lats)[1:size(speed_cpu,2)], speed_cpu; 
               colormap=:viridis, nan_color=:gray)
Colorbar(fig_final[1, 4], hm2)

ax3 = Axis(fig_final[1, 5]; title="SSH (m)", xlabel="Longitude", ylabel="Latitude")
hm3 = heatmap!(ax3, collect(ocean_lons), collect(ocean_lats), η_cpu; colormap=:balance, nan_color=:gray)
Colorbar(fig_final[1, 6], hm3)

output_file = joinpath(output_dir, "final_state.png")
save(output_file, fig_final)
@info "Final state saved to: $output_file"
