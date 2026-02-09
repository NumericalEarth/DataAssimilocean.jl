# Hurricane Irma test - verify hurricane shows up in atmosphere wind
# Run a short simulation during peak Irma (September 5-7, 2017)

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
output_dir = "hurricane_test_output"
mkpath(output_dir)

# Bathymetry
bathymetry_file = "atlantic_prototype_output/bathymetry_$(Nx)x$(Ny).jld2"
bottom_height_data = jldopen(bathymetry_file)["bottom_height"]
bottom_height = Field{Center, Center, Nothing}(grid)
set!(bottom_height, bottom_height_data)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Hurricane Irma period:
# - September 5: Cat 5, peak intensity
# - September 6-7: Moving through Caribbean
# Run for 3 days during peak
start_date = DateTime(2017, 9, 5)
run_duration = 3  # days
end_date = start_date + Day(run_duration)

@info "Hurricane Irma test: $start_date to $end_date"

# Ocean model
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

# Simulation - use smaller timestep for hurricane conditions
Δt = 5minutes  # Smaller timestep for extreme winds
stop_time = run_duration * days
simulation = Simulation(coupled_model; Δt, stop_time)

# Progress
wall_time = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_time[])
    current_date = start_date + Dates.Second(round(Int, time(sim)))
    @info @sprintf("Iter %d, t=%s (%s), wall=%s", iteration(sim), prettytime(sim), current_date, prettytime(elapsed))
    wall_time[] = time_ns()
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# Snapshot callback - save every 3 hours
frame_count = Ref(0)
ocean_lons = range(longitude[1], longitude[2], length=Nx)
ocean_lats = range(latitude[1], latitude[2], length=Ny)

# Time offset for atmosphere FTS
jra55_epoch = DateTime(1958, 1, 1)
time_offset = (start_date - jra55_epoch).value / 1000

function save_snapshot(sim)
    frame_count[] += 1
    t = time(sim)
    current_date = start_date + Dates.Second(round(Int, t))
    
    # Ocean data
    T_data = Array(interior(ocean.model.tracers.T, :, :, grid.Nz))
    u_raw = Array(interior(ocean.model.velocities.u, :, :, grid.Nz))
    v_raw = Array(interior(ocean.model.velocities.v, :, :, grid.Nz))
    e_data = Array(interior(ocean.model.tracers.e, :, :, grid.Nz))
    
    u_center = 0.5 * (u_raw[1:end-1, :] .+ u_raw[2:end, :])
    v_center = 0.5 * (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
    speed_data = sqrt.(u_center.^2 .+ v_center.^2)
    tke_data = max.(e_data, 0.0f0)
    
    # Atmosphere wind with correct time offset
    atm_time = Oceananigans.OutputReaders.Time(time_offset + t)
    ua_fts = atmosphere.velocities.u
    va_fts = atmosphere.velocities.v
    
    ua_snapshot = ua_fts[atm_time]
    va_snapshot = va_fts[atm_time]
    ua_atm = Array(interior(ua_snapshot, :, :, 1))
    va_atm = Array(interior(va_snapshot, :, :, 1))
    wind_atm = sqrt.(ua_atm.^2 .+ va_atm.^2)
    
    # Interpolate to ocean grid
    atm_grid = ua_fts.grid
    atm_lons = Array(atm_grid.λᶜᵃᵃ[1:atm_grid.Nx])
    atm_lats = Array(atm_grid.φᵃᶜᵃ[1:atm_grid.Ny])
    
    wind_data = zeros(Float32, Nx, Ny)
    for j in 1:Ny
        ocean_lat = latitude[1] + (j - 0.5) * (latitude[2] - latitude[1]) / Ny
        aj = argmin(abs.(atm_lats .- ocean_lat))
        for i in 1:Nx
            ocean_lon = longitude[1] + (i - 0.5) * (longitude[2] - longitude[1]) / Nx
            ocean_lon_360 = ocean_lon < 0 ? ocean_lon + 360 : ocean_lon
            ai = argmin(abs.(atm_lons .- ocean_lon_360))
            wind_data[i, j] = wind_atm[ai, aj]
        end
    end
    
    wind_valid = filter(!isnan, vec(wind_data))
    wind_max = maximum(wind_valid)
    
    @info "Frame $(frame_count[]): wind max=$(round(wind_max, digits=1)) m/s, date=$current_date"
    
    # Create figure - highlight hurricane with extended wind colorbar
    fig = Figure(size=(1600, 1200), fontsize=14)
    
    day_num = round(t / 86400, digits=2)
    title_str = "Hurricane Irma - $(Dates.format(current_date, "yyyy-mm-dd HH:MM")) (Day $day_num)"
    Label(fig[0, :], title_str; fontsize=20, tellwidth=false)
    
    # SST
    ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm1 = heatmap!(ax1, collect(ocean_lons), collect(ocean_lats), T_data;
                   colormap=:thermal, colorrange=(20, 32), nan_color=:gray)
    Colorbar(fig[1, 2], hm1)
    
    # Sea Surface Speed
    ax2 = Axis(fig[1, 3]; title="Sea Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm2 = heatmap!(ax2, collect(ocean_lons)[1:size(speed_data,1)], collect(ocean_lats)[1:size(speed_data,2)], speed_data;
                   colormap=:viridis, colorrange=(0, 2.0), nan_color=:gray)
    Colorbar(fig[1, 4], hm2)
    
    # Wind Speed - use extended range to capture hurricane winds
    ax3 = Axis(fig[2, 1]; title="Wind Speed (m/s) [max=$(round(wind_max,digits=1))]", 
               xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm3 = heatmap!(ax3, collect(ocean_lons), collect(ocean_lats), wind_data;
                   colormap=:plasma, colorrange=(0, 50), nan_color=:gray)  # Extended for hurricane
    Colorbar(fig[2, 2], hm3)
    
    # TKE - auto-scaled
    tke_max = maximum(filter(!isnan, vec(tke_data)))
    ax4 = Axis(fig[2, 3]; title="Surface TKE (m²/s²) [max=$(round(tke_max, sigdigits=2))]", 
               xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
    hm4 = heatmap!(ax4, collect(ocean_lons), collect(ocean_lats), tke_data;
                   colormap=:inferno, colorrange=(0, max(0.01, tke_max * 1.2)), nan_color=:gray)
    Colorbar(fig[2, 4], hm4)
    
    filename = joinpath(output_dir, @sprintf("irma_frame_%04d.png", frame_count[]))
    save(filename, fig)
end

simulation.callbacks[:snapshot] = Callback(save_snapshot, TimeInterval(3hour))

# Run
@info "Running Hurricane Irma test simulation..."
run!(simulation)
@info "Complete! Saved $(frame_count[]) frames"

# Create animation
@info "Creating animation..."
frames = sort([joinpath(output_dir, f) for f in readdir(output_dir) if startswith(f, "irma_frame_")])

fig = Figure(size=(1600, 1200))
ax = Axis(fig[1, 1])
hidedecorations!(ax)
hidespines!(ax)

animation_file = joinpath(output_dir, "hurricane_irma_animation.mp4")
record(fig, animation_file, frames; framerate=4) do frame_file
    img = load(frame_file)
    empty!(ax)
    image!(ax, rotr90(img))
end
@info "Animation saved to: $animation_file"






