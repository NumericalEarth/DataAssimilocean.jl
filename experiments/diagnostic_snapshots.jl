# Diagnostic script to verify atmosphere wind is evolving
# Saves PNG snapshots instead of animation for debugging

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

const hour = Oceananigans.Units.hour

arch = GPU()

# Domain
longitude = (-100, 0)
latitude  = (-10, 60)

Nx = Int(100 / 0.25)  # 400
Ny = Int(70 / 0.25)   # 280
Nz = 50
depth = 5000meters

z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4)

grid = LatitudeLongitudeGrid(arch; latitude, longitude, z,
                             size = (Nx, Ny, Nz), halo = (7, 7, 7))

# Output directory
output_dir = "diagnostic_output"
mkpath(output_dir)

# Bathymetry
bathymetry_file = "atlantic_prototype_output/bathymetry_$(Nx)x$(Ny).jld2"
if isfile(bathymetry_file)
    @info "Loading cached bathymetry"
    bottom_height_data = jldopen(bathymetry_file)["bottom_height"]
    bottom_height = Field{Center, Center, Nothing}(grid)
    set!(bottom_height, bottom_height_data)
else
    error("Bathymetry file not found: $bathymetry_file")
end

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Simulation dates - 7 day test run
start_date = DateTime(2017, 6, 26)
run_duration = 7  # days
end_date = start_date + Day(run_duration)

@info "Test run: $start_date to $end_date ($run_duration days)"

# Ocean model (no restoring for quick test)
free_surface = SplitExplicitFreeSurface(grid; substeps=30)
ocean = ocean_simulation(grid;
                         momentum_advection = WENOVectorInvariant(order=5),
                         tracer_advection = WENO(order=5),
                         free_surface)

# Initial conditions
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
    T = ocean_model.model.tracers.T
    elapsed = 1e-9 * (time_ns() - wall_time[])
    current_date = start_date + Dates.Second(round(Int, time(sim)))

    @info @sprintf("Iter %d, t=%s (%s), wall=%s | u=(%.2e,%.2e) | T=(%.1f,%.1f)",
                   iteration(sim), prettytime(sim), current_date, prettytime(elapsed),
                   maximum(abs, u), maximum(abs, v),
                   minimum(T), maximum(T))
    wall_time[] = time_ns()
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# Get interface for atmosphere wind
interface = coupled_model.interfaces.atmosphere_ocean_interface
exchanger = coupled_model.interfaces.exchanger
atm_state = exchanger.exchange_atmosphere_state

# Callback to save diagnostic snapshots
snapshot_interval = 6hour  # Every 6 hours = 28 frames for 7 days
frame_count = Ref(0)

ocean_lons = range(longitude[1], longitude[2], length=Nx)
ocean_lats = range(latitude[1], latitude[2], length=Ny)

function save_snapshot(sim)
    frame_count[] += 1
    t = time(sim)
    current_date = start_date + Dates.Second(round(Int, t))
    
    # Get ocean data
    T_data = Array(interior(ocean.model.tracers.T, :, :, grid.Nz))
    u_raw = Array(interior(ocean.model.velocities.u, :, :, grid.Nz))
    v_raw = Array(interior(ocean.model.velocities.v, :, :, grid.Nz))
    e_data = Array(interior(ocean.model.tracers.e, :, :, grid.Nz))
    
    # Ocean speed
    u_center = 0.5 * (u_raw[1:end-1, :] .+ u_raw[2:end, :])
    v_center = 0.5 * (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
    speed_data = sqrt.(u_center.^2 .+ v_center.^2)
    
    # Clamp TKE to non-negative
    tke_data = max.(e_data, 0.0f0)
    
    # Get atmosphere wind directly from atmosphere field time series
    # The FTS times are absolute (seconds since JRA55 epoch 1958-01-01)
    # We need to add the offset to convert simulation time to FTS time
    jra55_epoch = DateTime(1958, 1, 1)
    time_offset = (start_date - jra55_epoch).value / 1000  # milliseconds to seconds
    atm_time = Oceananigans.OutputReaders.Time(time_offset + t)
    
    # Get atmosphere data at current time (on atmosphere grid)
    ua_fts = atmosphere.velocities.u
    va_fts = atmosphere.velocities.v
    
    ua_snapshot = ua_fts[atm_time]
    va_snapshot = va_fts[atm_time]
    ua_atm = Array(interior(ua_snapshot, :, :, 1))
    va_atm = Array(interior(va_snapshot, :, :, 1))
    
    # Compute wind speed on atmosphere grid
    wind_atm = sqrt.(ua_atm.^2 .+ va_atm.^2)
    
    # Interpolate to ocean grid (nearest neighbor)
    atm_nx, atm_ny = size(wind_atm)
    wind_data = zeros(Float32, Nx, Ny)
    
    # Get atmosphere grid info
    atm_grid = ua_fts.grid
    atm_lons = Array(atm_grid.λᶜᵃᵃ[1:atm_grid.Nx])
    atm_lats = Array(atm_grid.φᵃᶜᵃ[1:atm_grid.Ny])
    
    # Interpolate: find nearest atmosphere grid point for each ocean grid point
    for j in 1:Ny
        ocean_lat = latitude[1] + (j - 0.5) * (latitude[2] - latitude[1]) / Ny
        aj = argmin(abs.(atm_lats .- ocean_lat))
        for i in 1:Nx
            ocean_lon = longitude[1] + (i - 0.5) * (longitude[2] - longitude[1]) / Nx
            # Convert to 0-360 if atmosphere uses that
            ocean_lon_360 = ocean_lon < 0 ? ocean_lon + 360 : ocean_lon
            ai = argmin(abs.(atm_lons .- ocean_lon_360))
            wind_data[i, j] = wind_atm[ai, aj]
        end
    end
    
    # Compute statistics for TKE auto-scaling
    tke_valid = filter(!isnan, vec(tke_data))
    tke_max = length(tke_valid) > 0 ? maximum(tke_valid) : 0.01
    tke_mean = length(tke_valid) > 0 ? sum(tke_valid)/length(tke_valid) : 0.0
    
    wind_valid = filter(!isnan, vec(wind_data))
    wind_max = length(wind_valid) > 0 ? maximum(wind_valid) : 1.0
    wind_mean = length(wind_valid) > 0 ? sum(wind_valid)/length(wind_valid) : 0.0
    
    @info "Frame $(frame_count[]): wind mean=$(round(wind_mean, digits=2)) max=$(round(wind_max, digits=2)), TKE mean=$(round(tke_mean, sigdigits=2)) max=$(round(tke_max, sigdigits=2))"
    
    # Create figure
    fig = Figure(size=(1600, 1200), fontsize=14)
    
    day_num = round(t / 86400, digits=2)
    title_str = "Day $day_num - $(Dates.format(current_date, "yyyy-mm-dd HH:MM"))"
    Label(fig[0, :], title_str; fontsize=20, tellwidth=false)
    
    # Panel 1: SST
    ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm1 = heatmap!(ax1, collect(ocean_lons), collect(ocean_lats), T_data;
                   colormap=:thermal, colorrange=(15, 32), nan_color=:gray)
    Colorbar(fig[1, 2], hm1)
    
    # Panel 2: Sea Surface Speed
    ax2 = Axis(fig[1, 3]; title="Sea Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm2 = heatmap!(ax2, collect(ocean_lons)[1:size(speed_data,1)], collect(ocean_lats)[1:size(speed_data,2)], speed_data;
                   colormap=:viridis, colorrange=(0, 1.5), nan_color=:gray)
    Colorbar(fig[1, 4], hm2)
    
    # Panel 3: Wind Speed - with statistics in title
    ax3 = Axis(fig[2, 1]; title="Atm. Wind Speed (m/s) [mean=$(round(wind_mean,digits=1)), max=$(round(wind_max,digits=1))]", 
               xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm3 = heatmap!(ax3, collect(ocean_lons), collect(ocean_lats), wind_data;
                   colormap=:plasma, colorrange=(0, max(25, wind_max)), nan_color=:gray)
    Colorbar(fig[2, 2], hm3)
    
    # Panel 4: Surface TKE - auto-scaled
    tke_colormax = max(0.001, tke_max * 1.2)  # Auto-scale with 20% headroom
    ax4 = Axis(fig[2, 3]; title="Surface TKE (m²/s²) [max=$(round(tke_max, sigdigits=2))]", 
               xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm4 = heatmap!(ax4, collect(ocean_lons), collect(ocean_lats), tke_data;
                   colormap=:inferno, colorrange=(0, tke_colormax), nan_color=:gray)
    Colorbar(fig[2, 4], hm4)
    
    # Save
    filename = joinpath(output_dir, @sprintf("frame_%04d_day%.2f.png", frame_count[], day_num))
    save(filename, fig)
    @info "Saved: $filename"
    
    return nothing
end

simulation.callbacks[:snapshot] = Callback(save_snapshot, TimeInterval(snapshot_interval))

# Run
@info "Running 7-day diagnostic simulation..."
@info "Saving snapshots every 6 hours to $output_dir"

run!(simulation)

@info "Simulation complete! Saved $(frame_count[]) snapshots to $output_dir"

# List saved files
@info "Saved files:"
for f in sort(readdir(output_dir))
    if endswith(f, ".png")
        println("  $f")
    end
end

