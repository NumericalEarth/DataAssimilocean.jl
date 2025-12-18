# # Atlantic Basin Prototype Simulation
#
# This script sets up a coarse-resolution (~1/4°) prototype simulation of the Atlantic basin,
# designed for CPU testing and development of data assimilation workflows.
#
# The simulation covers the Atlantic Ocean from approximately 100°W to 0°E and 0°N to 70°N,
# following the domain strategy outlined in project/README.md. This domain is designed to
# capture Gulf Stream dynamics and be suitable for hurricane-ocean interaction studies.
#
# ## Features
# - EN4-initialized temperature and salinity (no credentials required)
# - JRA-55 atmospheric forcing
# - High-frequency output (10 minutes) for data assimilation experiments
# - Configurable architecture (CPU/GPU) and resolution
#
# ## Initial condition data sources
#
# This script uses EN4 for initial conditions because it doesn't require authentication.
# EN4 is a quality-controlled ocean temperature and salinity dataset from the UK Met Office.
# See: https://www.metoffice.gov.uk/hadobs/en4/
#
# For higher-resolution initial conditions, you can switch to ECCO4, which requires
# setting up credentials. See the ClimaOcean documentation:
# https://github.com/CliMA/ClimaOcean.jl/blob/main/src/DataWrangling/ECCO/README.md
#
# ## Usage
# ```julia
# julia --project=. experiments/atlantic_prototype.jl
# ```

using Oceananigans
using Oceananigans.Units
using ClimaOcean
using ClimaOcean.EN4: EN4Monthly
using Printf
using Dates: DateTime

# ## Configuration
#
# Set the architecture. For development and testing, use CPU.
# For production runs, switch to GPU().

arch = CPU()

# ### Domain specification
#
# Atlantic basin domain following project/README.md:
# - Longitude: 100°W to 0°E (i.e., 260°E to 360°E in [0, 360) convention)
# - Latitude: 0°N to 70°N

longitude_west  = -100  # 100°W
longitude_east  =    0  # 0°E (Greenwich)
latitude_south  =    0  # Equator
latitude_north  =   70  # Extended north for Gulf Stream

longitude = (longitude_west, longitude_east)
latitude  = (latitude_south, latitude_north)

# ### Grid resolution
#
# For the CPU prototype, we use a very coarse resolution (~1/4°).
# This corresponds to roughly 28 km at the equator.

Δφ = 1/4  # degrees per grid cell (latitude)
Nφ = Int((latitude_north - latitude_south) / Δφ)  # number of latitude points
Nλ = Int((longitude_east - longitude_west) / Δφ)  # number of longitude points

# Aspect ratio: at mid-latitudes (~35°N), dx ≈ dy * cos(35°) ≈ 0.82 * dy
# For simplicity, we use square cells in degrees.

Nx = Nλ
Ny = Nφ

# ### Vertical grid
#
# We use a stretched vertical grid with enhanced resolution near the surface.
# For the coarse prototype, we reduce the number of vertical levels.

Nz = 50  # Reduced from production 200 for faster testing
depth = 5000meters

# Exponential discretization gives high resolution near surface
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth / 4)

@info "Grid configuration:" Nx Ny Nz

# ## Build the grid

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude,
                             longitude,
                             z,
                             halo = (7, 7, 7))

# ### Bathymetry
#
# We interpolate ETOPO bathymetry onto our grid and create an immersed boundary.

@info "Regridding bathymetry..."

bottom_height = regrid_bathymetry(grid;
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)  # Keep only Atlantic

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

@info "Grid with bathymetry:" grid

# ## Ocean simulation
#
# We build the ocean simulation using ClimaOcean's `ocean_simulation` constructor,
# which sets up a HydrostaticFreeSurfaceModel with appropriate closures and numerics.

free_surface = SplitExplicitFreeSurface(grid; substeps = 30)
momentum_advection = WENOVectorInvariant(order = 5)
tracer_advection = WENO(order = 5)

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface)

@info "Ocean model:" ocean.model

# ## Initial conditions from EN4
#
# We initialize temperature and salinity from the EN4 monthly dataset.
# EN4 is freely available without credentials (unlike ECCO which requires NASA Earthdata login).
# For hurricane studies, a good starting date is about 2 weeks before storm formation.
# Here we use a default date that can be adjusted for specific cases.
#
# Note: To use ECCO instead (higher resolution), uncomment the ECCO lines below and
# set ECCO_USERNAME and ECCO_WEBDAV_PASSWORD environment variables.
# See: https://github.com/CliMA/ClimaOcean.jl/blob/main/src/DataWrangling/ECCO/README.md

start_date = DateTime(2017, 8, 1)  # ~2 weeks before Hurricane Irma formation

@info "Setting initial conditions from EN4 for $start_date..."

# Using EN4 (no credentials required)
set!(ocean.model,
     T = Metadatum(:temperature; date = start_date, dataset = EN4Monthly()),
     S = Metadatum(:salinity;    date = start_date, dataset = EN4Monthly()))

# To use ECCO instead (requires credentials):
# using ClimaOcean.ECCO: ECCO4Monthly
# set!(ocean.model,
#      T = Metadatum(:temperature; date = start_date, dataset = ECCO4Monthly()),
#      S = Metadatum(:salinity;    date = start_date, dataset = ECCO4Monthly()))

# ## Atmospheric forcing
#
# We prescribe atmospheric conditions from JRA-55 reanalysis.
# This provides surface winds, heat fluxes, and freshwater fluxes.

@info "Setting up JRA-55 atmospheric forcing..."

radiation = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch;
                                       backend = JRA55NetCDFBackend(41),
                                       include_rivers_and_icebergs = false)

# ## Coupled model
#
# We assemble the ocean with atmospheric forcing into a coupled model.

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# ## Simulation setup
#
# For the coarse 1/4° grid, we can use a relatively large time step.
# We start conservatively and let the TimeStepWizard adjust.

Δt = 30minutes  # Initial time step (conservative for spin-up)

# For testing, we run a short simulation. For production, increase stop_time.
stop_time = 1days  # Short for testing; increase for science runs

simulation = Simulation(coupled_model; Δt, stop_time)

# ### Time step wizard
#
# Automatically adjust time step based on CFL condition.

wizard = TimeStepWizard(; cfl = 0.2, max_Δt = 1hour, max_change = 1.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ### Progress callback
#
# Monitor simulation progress with helpful diagnostics.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    S = ocean.model.tracers.S

    Tmax = maximum(T)
    Tmin = minimum(T)
    Smax = maximum(S)
    Smin = minimum(S)

    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    msg = @sprintf("Iter %d, time: %s, Δt: %s, wall: %s\n",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed))
    msg *= @sprintf("    max|u|: (%.2e, %.2e, %.2e) m/s\n", umax, vmax, wmax)
    msg *= @sprintf("    T: (%.2f, %.2f) °C, S: (%.2f, %.2f) psu", Tmin, Tmax, Smin, Smax)

    @info msg

    wall_time[] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ## Output writers
#
# For data assimilation, we need high-frequency output. We save:
# 1. Surface fields every 10 minutes (for SST, SSH assimilation)
# 2. 3D fields less frequently (for profile assimilation)

output_dir = "atlantic_prototype_output"
mkpath(output_dir)

# ### High-frequency surface output (every 10 minutes)
#
# This is the primary output for data assimilation experiments.

surface_fields = merge(ocean.model.tracers, ocean.model.velocities)
surface_output_interval = 10minutes

ocean.output_writers[:surface] = JLD2Writer(ocean.model, surface_fields;
                                            schedule = TimeInterval(surface_output_interval),
                                            filename = joinpath(output_dir, "surface_fields"),
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

# Also save sea surface height (η)
ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η = ocean.model.free_surface.η);
                                                 schedule = TimeInterval(surface_output_interval),
                                                 filename = joinpath(output_dir, "sea_surface_height"),
                                                 overwrite_existing = true)

# ### 3D snapshots (less frequent)
#
# Full 3D fields for detailed analysis, saved less frequently.

snapshot_interval = 1hour

ocean.output_writers[:snapshots] = JLD2Writer(ocean.model, surface_fields;
                                              schedule = TimeInterval(snapshot_interval),
                                              filename = joinpath(output_dir, "snapshots_3d"),
                                              overwrite_existing = true)

@info "Output configuration:"
@info "  Surface fields: every $(prettytime(surface_output_interval))"
@info "  3D snapshots: every $(prettytime(snapshot_interval))"
@info "  Output directory: $output_dir"

# ## Run the simulation
#
# For quick testing, you can limit the number of iterations:
# ```julia
# simulation.stop_iteration = 100
# ```

@info "Starting simulation..."
@info "  Stop time: $(prettytime(stop_time))"
@info "  Initial Δt: $(prettytime(Δt))"

# Uncomment the line below for a quick test run:
# simulation.stop_iteration = 100

run!(simulation)

@info "Simulation complete!"
@info "  Final time: $(prettytime(simulation))"
@info "  Total iterations: $(iteration(simulation))"

# ## Quick visualization (optional)
#
# Uncomment the following to visualize final state:
#=
using CairoMakie

T = ocean.model.tracers.T
S = ocean.model.tracers.S
η = ocean.model.free_surface.η

fig = Figure(size = (1200, 800))

ax1 = Axis(fig[1, 1], title = "Sea Surface Temperature (°C)")
hm1 = heatmap!(ax1, view(T, :, :, grid.Nz), colormap = :thermal)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], title = "Sea Surface Salinity (psu)")
hm2 = heatmap!(ax2, view(S, :, :, grid.Nz), colormap = :haline)
Colorbar(fig[1, 4], hm2)

ax3 = Axis(fig[2, 1], title = "Sea Surface Height (m)")
hm3 = heatmap!(ax3, η, colormap = :balance)
Colorbar(fig[2, 2], hm3)

save(joinpath(output_dir, "final_state.png"), fig)
=#

