# Atlantic Basin Prototype Simulation (1/4°, GPU)
# EN4-initialized, JRA-55 forced, high-frequency output for data assimilation

using Oceananigans
using Oceananigans.Units
using ClimaOcean
using ClimaOcean.EN4: EN4Monthly
using CairoMakie
using Printf
using Dates: DateTime

arch = CPU()  # GPU() has type inference issues; using CPU for now

# Domain: Atlantic 100°W–0°E, 0°N–70°N
longitude = (-100, 0)
latitude  = (0, 70)

# 1/4° resolution
Δφ = 1/4
Nx = Int(100 / Δφ)
Ny = Int(70 / Δφ)
Nz = 50
depth = 5000meters

z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude, longitude, z,
                             halo = (7, 7, 7))

# Bathymetry
bottom_height = regrid_bathymetry(grid;
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Ocean model
free_surface = SplitExplicitFreeSurface(grid; substeps=30)
ocean = ocean_simulation(grid;
                         momentum_advection = WENOVectorInvariant(order=5),
                         tracer_advection = WENO(order=5),
                         free_surface)

# Initial conditions from EN4
start_date = DateTime(2017, 8, 1)
set!(ocean.model,
     T = Metadatum(:temperature; date=start_date, dataset=EN4Monthly()),
     S = Metadatum(:salinity;    date=start_date, dataset=EN4Monthly()))

# Atmospheric forcing
radiation = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch;
                                       backend = JRA55NetCDFBackend(41),
                                       include_rivers_and_icebergs = false)

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# Simulation (TimeStepWizard not supported with OceanSeaIceModel)
Δt = 10minutes
stop_time = 1days
simulation = Simulation(coupled_model; Δt, stop_time)

# Progress callback
wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T, S = ocean.model.tracers.T, ocean.model.tracers.S
    elapsed = 1e-9 * (time_ns() - wall_time[])

    @info @sprintf("Iter %d, t=%s, Δt=%s, wall=%s | u=(%.2e,%.2e,%.2e) | T=(%.1f,%.1f) S=(%.1f,%.1f)",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), minimum(S), maximum(S))
    wall_time[] = time_ns()
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Output
output_dir = "atlantic_prototype_output"
mkpath(output_dir)

surface_fields = merge(ocean.model.tracers, ocean.model.velocities)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, surface_fields;
                                            schedule = TimeInterval(10minutes),
                                            filename = joinpath(output_dir, "surface_fields"),
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η=ocean.model.free_surface.η);
                                                 schedule = TimeInterval(10minutes),
                                                 filename = joinpath(output_dir, "sea_surface_height"),
                                                 overwrite_existing = true)

ocean.output_writers[:snapshots] = JLD2Writer(ocean.model, surface_fields;
                                              schedule = TimeInterval(1hour),
                                              filename = joinpath(output_dir, "snapshots_3d"),
                                              overwrite_existing = true)

# Run
@info "Running Atlantic prototype simulation on GPU..."
run!(simulation)
@info "Complete! Iterations: $(iteration(simulation))"

# Plot final state
T = ocean.model.tracers.T
S = ocean.model.tracers.S
η = ocean.model.free_surface.η

fig = Figure(size=(1400, 500), fontsize=14)

ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude")
hm1 = heatmap!(ax1, view(T, :, :, grid.Nz); colormap=:thermal)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3]; title="SSS (psu)", xlabel="Longitude", ylabel="Latitude")
hm2 = heatmap!(ax2, view(S, :, :, grid.Nz); colormap=:haline)
Colorbar(fig[1, 4], hm2)

ax3 = Axis(fig[1, 5]; title="SSH (m)", xlabel="Longitude", ylabel="Latitude")
hm3 = heatmap!(ax3, η; colormap=:balance)
Colorbar(fig[1, 6], hm3)

output_file = joinpath(output_dir, "final_state.png")
save(output_file, fig)
@info "Saved plot to $output_file"
