# Shared utilities for Hurricane Irma two-stage simulation
# Used by stage1_spinup.jl and stage2_highres.jl
#
# Functions for grid creation, model setup, I/O, and interpolation

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Units: minutes, days, meters
using Oceananigans.OutputReaders: Linear
using Oceananigans.Fields: interpolate!
using ClimaOcean
using ClimaOcean.EN4: EN4Monthly
using ClimaOcean.JRA55: MultiYearJRA55
using Dates
using JLD2
using Printf

# Define hour explicitly to avoid conflict with Dates.hour
const hour = Oceananigans.Units.hour

# ============================================================================
#                           DOMAIN CONFIGURATION
# ============================================================================

const DOMAIN_LONGITUDE = (-100, 0)   # Atlantic 100°W to 0°E
const DOMAIN_LATITUDE  = (-10, 60)   # 10°S to 60°N
const DOMAIN_DEPTH     = 5000meters
const Nz_DEFAULT       = 50

# Lateral restoring configuration
const SPONGE_WIDTH     = 5           # degrees
const RESTORING_RATE   = 1 / (7days) # 7-day timescale

# ============================================================================
#                           GRID CREATION
# ============================================================================

"""
    create_grid(arch, resolution; Nz=50, depth=5000meters)

Create a LatitudeLongitudeGrid for the Atlantic domain at given resolution.

# Arguments
- `arch`: Architecture (CPU() or GPU())
- `resolution`: Grid spacing in degrees (e.g., 1/4 or 1/24)
- `Nz`: Number of vertical levels (default: 50)
- `depth`: Maximum depth (default: 5000m)

# Returns
- `LatitudeLongitudeGrid`
"""
function create_grid(arch, resolution; Nz=Nz_DEFAULT, depth=DOMAIN_DEPTH)
    longitude = DOMAIN_LONGITUDE
    latitude = DOMAIN_LATITUDE
    
    Lλ = longitude[2] - longitude[1]
    Lφ = latitude[2] - latitude[1]
    
    Nx = Int(Lλ / resolution)
    Ny = Int(Lφ / resolution)
    
    # Stretched vertical grid with high resolution near surface
    z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4)
    
    grid = LatitudeLongitudeGrid(arch;
                                 latitude, longitude, z,
                                 size = (Nx, Ny, Nz),
                                 halo = (7, 7, 7))
    
    @info "Created grid: $(Nx) × $(Ny) × $(Nz) at $(resolution)° resolution"
    return grid
end

# ============================================================================
#                           BATHYMETRY
# ============================================================================

"""
    load_or_compute_bathymetry(grid, output_dir)

Load cached bathymetry or compute and cache it.

# Returns
- `ImmersedBoundaryGrid` with `GridFittedBottom`
"""
function load_or_compute_bathymetry(grid, output_dir)
    Nx, Ny, _ = size(grid)
    bathymetry_file = joinpath(output_dir, "bathymetry_$(Nx)x$(Ny).jld2")
    
    if isfile(bathymetry_file)
        @info "Loading cached bathymetry from $bathymetry_file"
        bottom_height_data = jldopen(bathymetry_file)["bottom_height"]
        bottom_height = Field{Center, Center, Nothing}(grid)
        set!(bottom_height, bottom_height_data)
    else
        @info "Computing bathymetry (this may take several minutes)..."
        bottom_height = regrid_bathymetry(grid;
                                          height_above_water = 1,
                                          minimum_depth = 10,
                                          interpolation_passes = 5,
                                          major_basins = 1)
        @info "Saving bathymetry to $bathymetry_file"
        jldsave(bathymetry_file; bottom_height=Array(interior(bottom_height)))
    end
    
    return ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
end

# ============================================================================
#                           OCEAN SIMULATION
# ============================================================================

"""
    create_ocean_simulation(grid, start_date, end_date; with_restoring=true)

Create an ocean simulation with WENO advection and optional lateral restoring.

# Arguments
- `grid`: ImmersedBoundaryGrid
- `start_date`, `end_date`: DateTime for simulation period
- `with_restoring`: Enable lateral restoring at N/S boundaries (default: true)

# Returns
- `ocean` simulation object from `ocean_simulation`
"""
function create_ocean_simulation(grid, start_date, end_date; with_restoring=true, substeps=30)
    arch = architecture(grid)
    
    # Free surface - high-res grids may need more substeps
    free_surface = SplitExplicitFreeSurface(grid; substeps)
    
    # Forcing (restoring at boundaries)
    if with_restoring
        restoring_mask = LinearlyTaperedPolarMask(;
            northern = (DOMAIN_LATITUDE[2] - SPONGE_WIDTH, DOMAIN_LATITUDE[2]),
            southern = (DOMAIN_LATITUDE[1], DOMAIN_LATITUDE[1] + SPONGE_WIDTH),
            z = (-DOMAIN_DEPTH, 0))
        
        T_restoring_metadata = Metadata(:temperature;
                                         dates = start_date:Month(1):end_date,
                                         dataset = EN4Monthly())
        
        S_restoring_metadata = Metadata(:salinity;
                                         dates = start_date:Month(1):end_date,
                                         dataset = EN4Monthly())
        
        T_restoring = DatasetRestoring(T_restoring_metadata, arch;
                                       rate = RESTORING_RATE,
                                       mask = restoring_mask,
                                       time_indexing = Linear())
        
        S_restoring = DatasetRestoring(S_restoring_metadata, arch;
                                       rate = RESTORING_RATE,
                                       mask = restoring_mask,
                                       time_indexing = Linear())
        
        forcing = (T = T_restoring, S = S_restoring)
    else
        forcing = NamedTuple()
    end
    
    ocean = ocean_simulation(grid;
                             momentum_advection = WENOVectorInvariant(order=5),
                             tracer_advection = WENO(order=5),
                             free_surface,
                             forcing)
    
    return ocean
end

# ============================================================================
#                           ATMOSPHERE
# ============================================================================

"""
    create_atmosphere(arch, start_date, end_date)

Create JRA55 prescribed atmosphere and radiation.

# Returns
- `(atmosphere, radiation)` tuple
"""
function create_atmosphere(arch, start_date, end_date)
    radiation = Radiation(arch)
    
    atmosphere = JRA55PrescribedAtmosphere(arch;
                                           dataset = MultiYearJRA55(),
                                           start_date = start_date,
                                           end_date = end_date,
                                           time_indexing = Linear(),
                                           backend = JRA55NetCDFBackend(41),
                                           include_rivers_and_icebergs = false)
    
    return atmosphere, radiation
end

# ============================================================================
#                           OUTPUT WRITERS
# ============================================================================

"""
    setup_output_writers!(ocean, coupled_model, output_dir; 
                          surface_interval=1hour, flux_interval=1hour)

Add output writers for surface fields, SSH, fluxes, and derived quantities.
"""
function setup_output_writers!(ocean, coupled_model, output_dir;
                               surface_interval = hour,
                               flux_interval = hour)
    grid = ocean.model.grid
    Nz = size(grid, 3)
    
    # Surface tracers and velocities
    surface_fields = merge(ocean.model.tracers, ocean.model.velocities)
    
    ocean.output_writers[:surface] = JLD2Writer(ocean.model, surface_fields;
                                                schedule = TimeInterval(surface_interval),
                                                filename = joinpath(output_dir, "surface_fields"),
                                                indices = (:, :, Nz),
                                                overwrite_existing = true)
    
    # Sea surface height
    ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η=ocean.model.free_surface.η);
                                                     schedule = TimeInterval(surface_interval),
                                                     filename = joinpath(output_dir, "sea_surface_height"),
                                                     overwrite_existing = true)
    
    # Atmosphere and flux fields
    interface = coupled_model.interfaces.atmosphere_ocean_interface
    exchanger = coupled_model.interfaces.exchanger
    atm_state = exchanger.exchange_atmosphere_state
    fluxes = interface.fluxes
    
    flux_outputs = (;
        atm_u = atm_state.u,
        atm_v = atm_state.v,
        tau_x = fluxes.x_momentum,
        tau_y = fluxes.y_momentum,
        Qh_sensible = fluxes.sensible_heat,
        Qh_latent = fluxes.latent_heat,
        Qs = fluxes.downwelling_shortwave
    )
    
    ocean.output_writers[:fluxes] = JLD2Writer(ocean.model, flux_outputs;
                                               schedule = TimeInterval(flux_interval),
                                               filename = joinpath(output_dir, "atmosphere_fluxes"),
                                               overwrite_existing = true)
    
    # Surface vorticity (∂v/∂x - ∂u/∂y approximation)
    u, v, _ = ocean.model.velocities
    ζ = Field(∂x(v) - ∂y(u))
    
    ocean.output_writers[:vorticity] = JLD2Writer(ocean.model, (; ζ=ζ);
                                                  schedule = TimeInterval(surface_interval),
                                                  filename = joinpath(output_dir, "surface_vorticity"),
                                                  indices = (:, :, Nz),
                                                  overwrite_existing = true)
    
    @info "Output writers configured in $output_dir"
    return nothing
end

# ============================================================================
#                           PROGRESS CALLBACK
# ============================================================================

"""
    add_progress_callback!(simulation, start_date; interval=100)

Add a progress callback that prints simulation status.
"""
function add_progress_callback!(simulation, start_date; interval=100)
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
    
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(interval))
    return nothing
end

# ============================================================================
#                           STATE SAVE/LOAD
# ============================================================================

"""
    save_model_state(ocean, filename)

Save ocean model state (T, S, u, v, w, η) to JLD2 file.
"""
function save_model_state(ocean, filename)
    model = ocean.model
    grid = model.grid
    
    # Extract fields to CPU
    T = Array(interior(model.tracers.T))
    S = Array(interior(model.tracers.S))
    e = Array(interior(model.tracers.e))
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    w = Array(interior(model.velocities.w))
    η = Array(interior(model.free_surface.η))
    
    # Grid info for verification
    Nx, Ny, Nz = size(grid)
    
    jldsave(filename;
            T, S, e, u, v, w, η,
            grid_size = (Nx, Ny, Nz),
            resolution = DOMAIN_LONGITUDE[2] - DOMAIN_LONGITUDE[1] / Nx,
            longitude = DOMAIN_LONGITUDE,
            latitude = DOMAIN_LATITUDE)
    
    @info "Saved model state to $filename"
    @info "  Grid size: ($Nx, $Ny, $Nz)"
    @info "  T range: ($(minimum(T)), $(maximum(T)))"
    @info "  S range: ($(minimum(S)), $(maximum(S)))"
    return nothing
end

"""
    load_model_state(filename)

Load ocean model state from JLD2 file.

# Returns
- NamedTuple with (T, S, e, u, v, w, η, grid_size)
"""
function load_model_state(filename)
    @info "Loading model state from $filename"
    data = jldopen(filename) do file
        (T = file["T"],
         S = file["S"],
         e = file["e"],
         u = file["u"],
         v = file["v"],
         w = file["w"],
         η = file["η"],
         grid_size = file["grid_size"])
    end
    
    @info "  Source grid size: $(data.grid_size)"
    @info "  T range: ($(minimum(data.T)), $(maximum(data.T)))"
    @info "  S range: ($(minimum(data.S)), $(maximum(data.S)))"
    return data
end

# ============================================================================
#                           INTERPOLATION (COARSE → FINE)
# ============================================================================

"""
    interpolate_state!(target_ocean, source_state, source_resolution, target_resolution)

Interpolate saved state from coarse grid to fine grid using Oceananigans interpolation.

# Arguments
- `target_ocean`: Ocean simulation on the fine grid
- `source_state`: NamedTuple from `load_model_state`
- `source_resolution`: Resolution of source grid (degrees)
- `target_resolution`: Resolution of target grid (degrees)
"""
function interpolate_state!(target_ocean, source_state, source_resolution, target_resolution)
    target_model = target_ocean.model
    target_grid = target_model.grid.underlying_grid  # Get the underlying grid, not ImmersedBoundaryGrid
    arch = architecture(target_grid)
    
    source_Nx, source_Ny, source_Nz = source_state.grid_size
    target_Nx, target_Ny, target_Nz = size(target_grid)
    
    @info "Interpolating state from $(source_Nx)×$(source_Ny)×$(source_Nz) to $(target_Nx)×$(target_Ny)×$(target_Nz)"
    
    # Create source grid on CPU for interpolation
    source_z = ExponentialDiscretization(source_Nz, -DOMAIN_DEPTH, 0; scale=DOMAIN_DEPTH/4)
    source_grid = LatitudeLongitudeGrid(CPU();
                                        latitude = DOMAIN_LATITUDE,
                                        longitude = DOMAIN_LONGITUDE,
                                        z = source_z,
                                        size = (source_Nx, source_Ny, source_Nz),
                                        halo = (7, 7, 7))
    
    # Create CPU target grid for interpolation
    target_z = ExponentialDiscretization(target_Nz, -DOMAIN_DEPTH, 0; scale=DOMAIN_DEPTH/4)
    target_cpu_grid = LatitudeLongitudeGrid(CPU();
                                            latitude = DOMAIN_LATITUDE,
                                            longitude = DOMAIN_LONGITUDE,
                                            z = target_z,
                                            size = (target_Nx, target_Ny, target_Nz),
                                            halo = (7, 7, 7))
    
    # Only interpolate tracers - velocities will be set to zero and spun up
    # This avoids bathymetry mismatch issues between different resolution grids
    for (name, data, loc) in [(:T, source_state.T, (Center, Center, Center)),
                               (:S, source_state.S, (Center, Center, Center)),
                               (:e, source_state.e, (Center, Center, Center))]
        
        @info "  Interpolating $name..."
        
        # Create source field and set data
        source_field = Field{loc[1], loc[2], loc[3]}(source_grid)
        set!(source_field, data)
        
        # Create temporary target field on CPU for interpolation
        target_field_cpu = Field{loc[1], loc[2], loc[3]}(target_cpu_grid)
        
        # Interpolate on CPU
        interpolate!(target_field_cpu, source_field)
        
        # Get interpolated data and copy to GPU model
        # Use Array() to ensure we have a plain CPU array, then set! handles CPU→GPU transfer
        interpolated_data = Array(interior(target_field_cpu))
        
        # Copy to target model
        if name == :T
            copyto!(interior(target_model.tracers.T), interpolated_data)
        elseif name == :S
            copyto!(interior(target_model.tracers.S), interpolated_data)
        elseif name == :e
            # Clamp TKE to positive values for stability
            copyto!(interior(target_model.tracers.e), max.(interpolated_data, 1e-10))
        end
    end
    
    # Velocities and η are left at zero - they will spin up naturally
    # This avoids bathymetry mismatch issues that cause instabilities
    @info "  Velocities and η initialized to zero (will spin up)"
    
    @info "Interpolation complete!"
    return nothing
end

# ============================================================================
#                           EXPORTS
# ============================================================================

# No explicit exports - include this file and use qualified names or
# import specific functions as needed


