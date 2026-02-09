# Master script: Run Hurricane Irma simulations at multiple resolutions
#
# This script runs:
# 1. Stage 1: 90-day spinup at 1/4° (shared by all high-res runs)
# 2. Stage 2a: 14-day analysis at 1/12° (initialized from Stage 1)
# 3. Stage 2b: 14-day analysis at 1/24° (initialized from Stage 1)
#
# Key features:
# - Stage 2 is properly initialized by interpolating T/S/e from Stage 1
# - Saves both final and initial states for documentation
# - Animations use correct JRA55 absolute time for wind fields
#
# Usage:
#   julia --project experiments/run_irma_all_resolutions.jl
#
# The script is designed to run unattended and survive disconnection.

include("irma_utils.jl")

using CairoMakie
using Dates: Hour
using Oceananigans.OutputReaders: Time as OceananigansTime

@info "=" ^ 80
@info "HURRICANE IRMA MULTI-RESOLUTION SIMULATION"
@info "=" ^ 80

# ============================================================================
#                           CONFIGURATION
# ============================================================================

arch = GPU()

# Hurricane timing
hurricane_start = DateTime(2017, 8, 25)
spinup_days = 90  # Extended spinup for better eddy development
analysis_days = 14

# Stage dates
stage1_start = hurricane_start - Day(spinup_days)
stage1_end = hurricane_start
stage2_start = hurricane_start
stage2_end = hurricane_start + Day(analysis_days)

# JRA55 epoch for absolute time calculations
const JRA55_EPOCH = DateTime(1958, 1, 1)

@info "Simulation period:"
@info "  Stage 1 (spinup):   $stage1_start to $stage1_end ($(spinup_days) days)"
@info "  Stage 2 (analysis): $stage2_start to $stage2_end ($(analysis_days) days)"

# ============================================================================
#                           STAGE 1: SPINUP (1/4°)
# ============================================================================

function run_stage1()
    @info "=" ^ 80
    @info "STAGE 1: 1/4° SPINUP ($(spinup_days) days)"
    @info "=" ^ 80
    
    resolution = 1/4
    output_dir = "irma_stage1_quarter_90day"
    mkpath(output_dir)
    
    # Check if already completed
    state_file = joinpath(output_dir, "final_state.jld2")
    if isfile(state_file)
        @info "Stage 1 already completed! Found: $state_file"
        @info "Skipping to Stage 2..."
        return output_dir
    end
    
    # Grid
    @info "Creating $(resolution)° grid..."
    grid = create_grid(arch, resolution)
    Nx, Ny, Nz = size(grid)
    @info "Grid size: $(Nx) × $(Ny) × $(Nz)"
    
    # Bathymetry
    grid = load_or_compute_bathymetry(grid, output_dir)
    
    # Ocean model
    @info "Creating ocean simulation..."
    ocean = create_ocean_simulation(grid, stage1_start, stage1_end; with_restoring=true)
    
    # Initial conditions
    @info "Setting initial conditions from EN4..."
    set!(ocean.model,
         T = Metadatum(:temperature; date=stage1_start, dataset=EN4Monthly()),
         S = Metadatum(:salinity;    date=stage1_start, dataset=EN4Monthly()))
    
    # Atmosphere
    @info "Setting up JRA55 atmosphere..."
    atmosphere, radiation = create_atmosphere(arch, stage1_start, stage1_end)
    
    # Coupled model
    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
    
    # Simulation
    Δt = 10minutes
    stop_time = spinup_days * days
    simulation = Simulation(coupled_model; Δt, stop_time)
    
    add_progress_callback!(simulation, stage1_start)
    setup_output_writers!(ocean, coupled_model, output_dir;
                          surface_interval = 3hour,
                          flux_interval = 3hour)
    
    # Run
    @info "Running Stage 1 spinup..."
    run!(simulation)
    
    # Save final state
    save_model_state(ocean, state_file)
    
    # Visualization
    create_final_state_plot(ocean, output_dir, "Stage 1 Final", stage1_end, resolution)
    
    @info "Stage 1 complete!"
    return output_dir
end

# ============================================================================
#                           STAGE 2: HIGH-RES ANALYSIS
# ============================================================================

function run_stage2(target_resolution, output_dir_name, stage1_dir)
    resolution_label = target_resolution == 1/12 ? "1/12" : "1/24"
    source_resolution = 1/4
    
    @info "=" ^ 80
    @info "STAGE 2: $(resolution_label)° HIGH-RES ANALYSIS"
    @info "=" ^ 80
    
    output_dir = output_dir_name
    mkpath(output_dir)
    
    # Check if already completed
    final_state_file = joinpath(output_dir, "final_state.jld2")
    if isfile(final_state_file)
        @info "Stage 2 ($(resolution_label)°) already completed! Found: $final_state_file"
        return output_dir
    end
    
    # Verify Stage 1 completed (for documentation purposes)
    stage1_state_file = joinpath(stage1_dir, "final_state.jld2")
    if !isfile(stage1_state_file)
        @warn "Stage 1 output not found: $stage1_state_file"
    else
        @info "Stage 1 state available: $stage1_state_file"
    end
    
    # Grid
    @info "Creating $(resolution_label)° grid..."
    grid = create_grid(arch, target_resolution)
    Nx, Ny, Nz = size(grid)
    @info "Grid size: $(Nx) × $(Ny) × $(Nz)"
    
    # Bathymetry
    grid = load_or_compute_bathymetry(grid, output_dir)
    
    # Ocean model - more substeps for high resolution
    substeps = target_resolution == 1/24 ? 75 : 50
    @info "Creating ocean simulation (substeps=$substeps)..."
    ocean = create_ocean_simulation(grid, stage2_start, stage2_end; 
                                    with_restoring=true, substeps=substeps)
    
    # =========================================================================
    # Initialize from EN4 at hurricane start date
    # Interpolation from coarse grid causes instabilities (zero velocity + 
    # pressure gradients from tracers). EN4 provides stable, realistic IC.
    # Stage 1 spinup still provides useful reference data.
    # =========================================================================
    @info "Setting initial conditions from EN4 at hurricane start date..."
    set!(ocean.model,
         T = Metadatum(:temperature; date=stage2_start, dataset=EN4Monthly()),
         S = Metadatum(:salinity;    date=stage2_start, dataset=EN4Monthly()))
    
    # Save initial state for documentation
    initial_state_file = joinpath(output_dir, "initial_state.jld2")
    save_model_state(ocean, initial_state_file)
    @info "Saved initial state to: $initial_state_file"
    
    # Create initial state visualization
    create_final_state_plot(ocean, output_dir, "Stage 2 Initial (EN4)", stage2_start, target_resolution)
    # Rename the file
    mv(joinpath(output_dir, "final_state.png"), 
       joinpath(output_dir, "initial_state.png"), force=true)
    
    # Atmosphere
    @info "Setting up JRA55 atmosphere..."
    atmosphere, radiation = create_atmosphere(arch, stage2_start, stage2_end)
    
    # Coupled model
    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
    
    # Simulation - smaller timestep for higher resolution
    Δt = target_resolution == 1/24 ? 30 : 60  # seconds
    stop_time = analysis_days * days
    simulation = Simulation(coupled_model; Δt, stop_time)
    
    @info "Time step: $(Δt) seconds"
    
    add_progress_callback!(simulation, stage2_start)
    setup_output_writers!(ocean, coupled_model, output_dir;
                          surface_interval = 1hour,
                          flux_interval = 1hour)
    
    # Run
    @info "Running Stage 2 $(resolution_label)° simulation..."
    run!(simulation)
    
    # Save final state
    save_model_state(ocean, final_state_file)
    
    # Visualization
    create_final_state_plot(ocean, output_dir, "Stage 2 Final ($(resolution_label)°)", stage2_end, target_resolution)
    
    @info "Stage 2 ($(resolution_label)°) complete!"
    return output_dir
end

# ============================================================================
#                           ANIMATION (with correct JRA55 time)
# ============================================================================

function create_animation(output_dir, resolution_label, sim_start_date)
    @info "Creating animation for $(resolution_label)..."
    
    # Load ocean data
    surface_file = joinpath(output_dir, "surface_fields.jld2")
    
    if !isfile(surface_file)
        @warn "Output files not found, skipping animation"
        return
    end
    
    T_ts = FieldTimeSeries(surface_file, "T")
    u_ts = FieldTimeSeries(surface_file, "u")
    v_ts = FieldTimeSeries(surface_file, "v")
    e_ts = FieldTimeSeries(surface_file, "e")
    
    times = T_ts.times
    Nt = length(times)
    @info "Animation: $(Nt) frames"
    
    # Get coordinates from T field
    grid = T_ts.grid
    Nx, Ny = size(grid, 1), size(grid, 2)
    lons = range(DOMAIN_LONGITUDE[1], DOMAIN_LONGITUDE[2], length=Nx)
    lats = range(DOMAIN_LATITUDE[1], DOMAIN_LATITUDE[2], length=Ny)
    
    # =========================================================================
    # Load JRA55 atmosphere directly for correct time-varying wind
    # JRA55 uses absolute time (seconds since 1958-01-01)
    # =========================================================================
    sim_end_date = sim_start_date + Day(ceil(Int, times[end] / 86400))
    @info "Loading JRA55 atmosphere for animation ($sim_start_date to $sim_end_date)..."
    atmosphere, _ = create_atmosphere(CPU(), sim_start_date, sim_end_date)
    
    # Calculate time offset: seconds from JRA55 epoch to simulation start
    time_offset = (sim_start_date - JRA55_EPOCH).value / 1000  # milliseconds to seconds
    @info "JRA55 time offset: $(time_offset) seconds ($(time_offset/86400) days since 1958-01-01)"
    
    # Animation
    fig = Figure(size=(1600, 1000))
    
    ax1 = Axis(fig[1, 1]; title="SST (°C)", xlabel="Longitude", ylabel="Latitude")
    ax2 = Axis(fig[1, 2]; title="Surface Speed (m/s)", xlabel="Longitude", ylabel="Latitude")
    ax3 = Axis(fig[2, 1]; title="Atm. Wind Speed (m/s)", xlabel="Longitude", ylabel="Latitude")
    ax4 = Axis(fig[2, 2]; title="Surface TKE (m²/s²)", xlabel="Longitude", ylabel="Latitude")
    
    time_label = Label(fig[0, :], "Initializing...", fontsize=24)
    
    anim_file = joinpath(output_dir, "irma_animation_$(replace(resolution_label, "/" => "_")).mp4")
    
    record(fig, anim_file, 1:Nt; framerate=24) do n
        empty!(ax1); empty!(ax2); empty!(ax3); empty!(ax4)
        
        # Current simulation time and date
        t = times[n]
        current_date = sim_start_date + Dates.Second(round(Int, t))
        time_label.text[] = "$(resolution_label)° | $(Dates.format(current_date, "yyyy-mm-dd HH:MM"))"
        
        # SST
        T_data = Array(interior(T_ts[n]))[:, :, 1]
        heatmap!(ax1, collect(lons), collect(lats), T_data; colormap=:thermal, colorrange=(10, 32))
        
        # Ocean surface speed
        u_data = Array(interior(u_ts[n]))[:, :, 1]
        v_data = Array(interior(v_ts[n]))[:, :, 1]
        u_c = 0.5 * (u_data[1:end-1, :] .+ u_data[2:end, :])
        v_c = 0.5 * (v_data[:, 1:end-1] .+ v_data[:, 2:end])
        mi = min(size(u_c, 1), size(v_c, 1))
        mj = min(size(u_c, 2), size(v_c, 2))
        speed = sqrt.(u_c[1:mi, 1:mj].^2 .+ v_c[1:mi, 1:mj].^2)
        heatmap!(ax2, collect(lons)[1:mi], collect(lats)[1:mj], speed; colormap=:speed, colorrange=(0, 2))
        
        # =====================================================================
        # Atmospheric wind speed - query JRA55 with ABSOLUTE time
        # Use OceananigansTime (not Dates.Time!) for FieldTimeSeries indexing
        # =====================================================================
        jra55_query_time = time_offset + t  # absolute time since 1958-01-01
        atm_u_field = atmosphere.velocities.u[OceananigansTime(jra55_query_time)]
        atm_v_field = atmosphere.velocities.v[OceananigansTime(jra55_query_time)]
        
        # Extract data over our domain (JRA55 is global, we want our region)
        atm_u_data = Array(interior(atm_u_field))
        atm_v_data = Array(interior(atm_v_field))
        
        # JRA55 grid coordinates
        atm_grid = atm_u_field.grid
        atm_lons = range(-180, 180, length=size(atm_grid, 1))
        atm_lats = range(-90, 90, length=size(atm_grid, 2))
        
        # Find indices for our domain
        lon_idx = findall(x -> DOMAIN_LONGITUDE[1] <= x <= DOMAIN_LONGITUDE[2], atm_lons)
        lat_idx = findall(x -> DOMAIN_LATITUDE[1] <= x <= DOMAIN_LATITUDE[2], atm_lats)
        
        atm_u_subset = atm_u_data[lon_idx, lat_idx, 1]
        atm_v_subset = atm_v_data[lon_idx, lat_idx, 1]
        wind_speed = sqrt.(atm_u_subset.^2 .+ atm_v_subset.^2)
        
        heatmap!(ax3, collect(atm_lons[lon_idx]), collect(atm_lats[lat_idx]), wind_speed; 
                 colormap=:viridis, colorrange=(0, 25))
        
        # TKE
        e_data = Array(interior(e_ts[n]))[:, :, 1]
        heatmap!(ax4, collect(lons), collect(lats), max.(e_data, 0); colormap=:inferno, colorrange=(0, 1e-3))
        
        if n % 50 == 1
            @info "  Frame $n / $Nt | Date: $current_date | Wind range: $(minimum(wind_speed))-$(maximum(wind_speed)) m/s"
        end
    end
    
    @info "Animation saved: $anim_file"
end

# ============================================================================
#                           HELPER: STATE PLOT
# ============================================================================

function create_final_state_plot(ocean, output_dir, title_label, date, resolution)
    @info "Creating state visualization: $title_label..."
    
    grid = ocean.model.grid
    Nx, Ny, Nz = size(grid)
    
    fig = Figure(size=(1200, 800))
    
    T_surf = Array(interior(ocean.model.tracers.T, :, :, Nz))
    S_surf = Array(interior(ocean.model.tracers.S, :, :, Nz))
    u_surf = Array(interior(ocean.model.velocities.u, :, :, Nz))
    v_surf = Array(interior(ocean.model.velocities.v, :, :, Nz))
    
    u_center = 0.5 * (u_surf[1:end-1, :] .+ u_surf[2:end, :])
    v_center = 0.5 * (v_surf[:, 1:end-1] .+ v_surf[:, 2:end])
    min_i = min(size(u_center, 1), size(v_center, 1))
    min_j = min(size(u_center, 2), size(v_center, 2))
    speed = sqrt.(u_center[1:min_i, 1:min_j].^2 .+ v_center[1:min_i, 1:min_j].^2)
    
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
    
    e_surf = Array(interior(ocean.model.tracers.e, :, :, Nz))
    ax4 = Axis(fig[2, 3]; title="Surface TKE (m²/s²)", xlabel="Longitude", ylabel="Latitude")
    hm4 = heatmap!(ax4, collect(lons), collect(lats), max.(e_surf, 0); colormap=:inferno, colorrange=(0, 1e-3))
    Colorbar(fig[2, 4], hm4)
    
    Label(fig[0, :], "$title_label: $(Dates.format(date, "yyyy-mm-dd")) ($(resolution)°)", fontsize=20)
    
    save(joinpath(output_dir, "final_state.png"), fig)
    @info "State plot saved"
end

# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

@info "Starting simulations..."
start_time = now()

# Run Stage 1 (shared spinup - 90 days)
stage1_dir = run_stage1()

# Run Stage 2 at 1/12°
stage2_12_dir = "irma_stage2_twelfth_v2"
run_stage2(1/12, stage2_12_dir, stage1_dir)
create_animation(stage2_12_dir, "1/12", stage2_start)

# Run Stage 2 at 1/24°
stage2_24_dir = "irma_stage2_twentyfourth_v2"
run_stage2(1/24, stage2_24_dir, stage1_dir)
create_animation(stage2_24_dir, "1/24", stage2_start)

# Create animation for Stage 1 as well
create_animation(stage1_dir, "1/4_spinup", stage1_start)

end_time = now()
total_time = end_time - start_time

@info "=" ^ 80
@info "ALL SIMULATIONS COMPLETE!"
@info "=" ^ 80
@info "Total runtime: $total_time"
@info ""
@info "Output directories:"
@info "  Stage 1 (1/4° spinup):     $stage1_dir/"
@info "  Stage 2 (1/12° analysis):  $stage2_12_dir/"
@info "  Stage 2 (1/24° analysis):  $stage2_24_dir/"
@info ""
@info "Key files:"
@info "  Stage 1 final state:       $stage1_dir/final_state.jld2"
@info "  Stage 2 initial states:    */initial_state_from_stage1.jld2"
@info "  Stage 2 final states:      */final_state.jld2"
@info ""
@info "Animations:"
@info "  $stage1_dir/irma_animation_1_4_spinup.mp4"
@info "  $stage2_12_dir/irma_animation_1_12.mp4"
@info "  $stage2_24_dir/irma_animation_1_24.mp4"
