# Test script to create a single frame of the 4-panel animation
# This validates the plotting before creating the full animation

using Oceananigans
using Oceananigans.Units: minutes, days, meters
using Oceananigans.OutputReaders: Linear
using ClimaOcean
using ClimaOcean.JRA55: MultiYearJRA55
using CairoMakie
using Dates
using JLD2

const hour = Oceananigans.Units.hour

output_dir = "atlantic_prototype_output"
surface_file = joinpath(output_dir, "surface_fields.jld2")

# Simulation parameters (must match the simulation)
start_date = DateTime(2017, 6, 26)
longitude = (-100, 0)
latitude = (-10, 60)

# Load JLD2 file
@info "Loading data from $surface_file"
file = jldopen(surface_file)

# Get time keys and actual time values
time_keys = sort([parse(Float64, k) for k in keys(file["timeseries/t"])])

# Build a mapping from key to actual time value
key_to_time = Dict{Int, Float64}()
for k in time_keys
    key_to_time[Int(k)] = file["timeseries/t/$(Int(k))"]
end

# Get actual times in seconds
actual_times = [key_to_time[Int(k)] for k in time_keys]

@info "Time span: $(actual_times[1]/86400) to $(actual_times[end]/86400) days"
@info "Number of snapshots: $(length(time_keys))"

# Pick a frame near the end (during Hurricane Irma)
# Day 70 = about 6048000 seconds
target_day = 70
target_time = target_day * 86400.0
frame_idx = argmin(abs.(actual_times .- target_time))
key = Int(time_keys[frame_idx])
actual_time = key_to_time[key]

@info "Selected frame: key=$key, actual_time=$(actual_time/86400) days"
current_date = start_date + Dates.Second(round(Int, actual_time))
@info "Date: $current_date"

# Load ocean data
T_data = file["timeseries/T/$key"][:, :, 1]
u_raw = file["timeseries/u/$key"][:, :, 1]
v_raw = file["timeseries/v/$key"][:, :, 1]
e_data = file["timeseries/e/$key"][:, :, 1]

close(file)

# Compute ocean speed (interpolate staggered velocities to cell centers)
Nx_T, Ny_T = size(T_data)
@info "T data size: $Nx_T x $Ny_T"

# u is (Nx+1, Ny), v is (Nx, Ny+1) on staggered grid
# Interpolate to cell centers
u_center = 0.5 * (u_raw[1:end-1, :] .+ u_raw[2:end, :])
v_center = 0.5 * (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
speed_data = sqrt.(u_center.^2 .+ v_center.^2)

# Create coordinate vectors for ocean grid
ocean_lons = range(longitude[1], longitude[2], length=Nx_T)
ocean_lats = range(latitude[1], latitude[2], length=Ny_T)

# Load atmosphere for this time
arch = CPU()  # Use CPU for this test
@info "Loading atmosphere..."
atmosphere = JRA55PrescribedAtmosphere(arch;
                                       dataset = MultiYearJRA55(),
                                       start_date = start_date,
                                       end_date = start_date + Day(74),
                                       time_indexing = Linear(),
                                       backend = JRA55NetCDFBackend(4),  # Small for testing
                                       include_rivers_and_icebergs = false)

# Get atmosphere wind at this time
atm_time = Oceananigans.OutputReaders.Time(actual_time)
ua_snapshot = atmosphere.velocities.u[atm_time]
va_snapshot = atmosphere.velocities.v[atm_time]

ua_full = Array(interior(ua_snapshot, :, :, 1))
va_full = Array(interior(va_snapshot, :, :, 1))

# Get atmosphere grid coordinates
atm_grid = atmosphere.velocities.u.grid
atm_lons = Array(atm_grid.λᶜᵃᵃ[1:atm_grid.Nx])
atm_lats = Array(atm_grid.φᵃᶜᵃ[1:atm_grid.Ny])

@info "Atmosphere lon range: $(minimum(atm_lons)) to $(maximum(atm_lons))"
@info "Atmosphere lat range: $(minimum(atm_lats)) to $(maximum(atm_lats))"

# Find indices that correspond to our domain
# Domain: longitude (-100, 0), latitude (-10, 60)
lon_min, lon_max = longitude
lat_min, lat_max = latitude

# Atmosphere uses 0-360 convention, so convert -100 to 260
lon_min_360 = lon_min < 0 ? lon_min + 360 : lon_min
lon_max_360 = lon_max < 0 ? lon_max + 360 : lon_max

# For Atlantic domain (-100 to 0) -> (260 to 360)
# Find indices
lat_indices = findall(lat_min .<= atm_lats .<= lat_max)

# Handle longitude wrap-around
if lon_min_360 > lon_max_360
    # We need indices from lon_min_360 to 360 and 0 to lon_max_360
    lon_indices_high = findall(atm_lons .>= lon_min_360)
    lon_indices_low = findall(atm_lons .<= lon_max_360)
    lon_indices = vcat(lon_indices_high, lon_indices_low)
else
    lon_indices = findall(lon_min_360 .<= atm_lons .<= lon_max_360)
end

@info "Atmosphere domain indices: lon=$(length(lon_indices)), lat=$(length(lat_indices))"

# Extract atmosphere data over domain
ua_domain = ua_full[lon_indices, lat_indices]
va_domain = va_full[lon_indices, lat_indices]
wind_speed = sqrt.(ua_domain.^2 .+ va_domain.^2)

# Get atmosphere coordinates for plotting (convert back to -180 to 180)
atm_lons_domain = atm_lons[lon_indices]
atm_lons_plot = [l > 180 ? l - 360 : l for l in atm_lons_domain]
atm_lats_domain = atm_lats[lat_indices]

@info "Wind speed over domain: size=$(size(wind_speed)), range=$(minimum(wind_speed)) to $(maximum(wind_speed))"
@info "Atmosphere plot lons: $(minimum(atm_lons_plot)) to $(maximum(atm_lons_plot))"

# Clamp TKE to non-negative (numerical artifacts can cause small negatives)
e_data_clamped = max.(e_data, 0.0f0)

# Replace NaN with a value for plotting
T_plot = copy(T_data)
speed_plot = copy(speed_data)
e_plot = copy(e_data_clamped)
wind_plot = copy(wind_speed)

@info "TKE range (clamped): $(minimum(filter(!isnan, e_plot))) to $(maximum(filter(!isnan, e_plot)))"

# Create 4-panel figure
fig = Figure(size=(1600, 1200), fontsize=14)

# Title
title_str = "Hurricane Irma Case - $(Dates.format(current_date, "yyyy-mm-dd HH:MM")) (Day $(round(actual_time/86400, digits=1)))"
Label(fig[0, :], title_str; fontsize=20, tellwidth=false)

# Panel 1: SST
ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude",
           aspect=DataAspect())
hm1 = heatmap!(ax1, collect(ocean_lons), collect(ocean_lats), T_plot; 
               colormap=:thermal, colorrange=(15, 32), nan_color=:gray)
Colorbar(fig[1, 2], hm1)

# Panel 2: Sea Surface Speed
ax2 = Axis(fig[1, 3]; title="Sea Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude",
           aspect=DataAspect())
hm2 = heatmap!(ax2, collect(ocean_lons)[1:size(speed_plot,1)], collect(ocean_lats)[1:size(speed_plot,2)], speed_plot; 
               colormap=:viridis, colorrange=(0, 1.5), nan_color=:gray)
Colorbar(fig[1, 4], hm2)

# Panel 3: Atmospheric Wind Speed
ax3 = Axis(fig[2, 1]; title="Atm. Wind Speed (m/s)", xlabel="Longitude", ylabel="Latitude",
           aspect=DataAspect())
hm3 = heatmap!(ax3, atm_lons_plot, collect(atm_lats_domain), wind_plot; 
               colormap=:plasma, colorrange=(0, 25))
Colorbar(fig[2, 2], hm3)

# Panel 4: Surface TKE
ax4 = Axis(fig[2, 3]; title="Surface TKE (m²/s²)", xlabel="Longitude", ylabel="Latitude",
           aspect=DataAspect())
tke_max = maximum(filter(!isnan, e_plot))
hm4 = heatmap!(ax4, collect(ocean_lons), collect(ocean_lats), e_plot; 
               colormap=:inferno, colorrange=(0, max(0.01, tke_max)), nan_color=:gray)
Colorbar(fig[2, 4], hm4)

# Save
output_file = joinpath(output_dir, "test_single_frame.png")
save(output_file, fig)
@info "Saved test frame to: $output_file"
