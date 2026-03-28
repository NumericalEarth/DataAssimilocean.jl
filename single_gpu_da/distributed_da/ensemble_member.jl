# Ensemble member forward run: forecast a batch of members for one DA cycle
#
# Usage:
#   julia --project single_gpu_da/distributed_da/ensemble_member.jl \
#       SPINUP_DIR ENSEMBLE_DIR OUTPUT_DIR CYCLE MEMBER_START MEMBER_END [N_CYCLES]
#
# SPINUP_DIR:   directory with spinup final_state.jld2 (for bathymetry cache)
# ENSEMBLE_DIR: directory containing cycle_XX/ with member state files
# OUTPUT_DIR:   directory where forecast_XX/ will be created
# CYCLE:        1-based cycle number
# MEMBER_START, MEMBER_END: range of members to process (inclusive)
# N_CYCLES:     optional total number of DA cycles (default 14)
#
# For each member: load state -> set! -> reset clocks -> run! -> extract -> save

include(joinpath(@__DIR__, "..", "..", "experiments", "irma_utils.jl"))
include(joinpath(@__DIR__, "da_utils.jl"))

using Oceananigans.Units: minutes, days

spinup_dir   = ARGS[1]
ensemble_dir = ARGS[2]
output_dir   = ARGS[3]
cycle        = parse(Int, ARGS[4])
member_start = parse(Int, ARGS[5])
member_end   = parse(Int, ARGS[6])

# Override N_cycles if provided (for short test runs)
N_cycles = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : DA_N_CYCLES

@info "Ensemble member forecast"
@info "  Cycle: $cycle"
@info "  Members: $member_start to $member_end"
@info "  Ensemble dir: $ensemble_dir"
@info "  Output dir: $output_dir"

# --- Time windows ---
analysis_window = DA_ANALYSIS_WINDOW_SECONDS
Dt = DA_DT_MINUTES * 60.0  # seconds
t_start = (cycle - 1) * analysis_window
t_end = cycle * analysis_window
start_date = DA_START_DATE
end_date = start_date + Day(N_cycles)

@info "  Time window: $(t_start/86400) to $(t_end/86400) days"

# --- Setup grid and model ---
arch = GPU()
resolution = DA_RESOLUTION

@info "Creating grid..."
grid = create_grid(arch, resolution)

# Add bathymetry
grid = load_or_compute_bathymetry(grid, spinup_dir)

@info "Creating ocean model..."
ocean = create_ocean_simulation(grid, start_date, end_date; with_restoring=true)

@info "Setting up atmosphere..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=Dt, stop_time=t_end)

# Remove output writers
empty!(ocean.output_writers)

# --- Load previous cycle's member states ---
prev_cycle_dir = cycle_dirname(ensemble_dir, cycle - 1)

# --- Create forecast output directory ---
fc_dir = forecast_dirname(output_dir, cycle)
mkpath(fc_dir)

# --- Warm-up compilation on first member ---
@info "Loading member $member_start for warm-up compilation..."
state = load_member_state(member_filename(prev_cycle_dir, member_start))

set!(ocean.model, T=state.T, S=state.S, e=state.e, u=state.u, v=state.v)
copyto!(interior(ocean.model.velocities.w), state.w)
copyto!(interior(ocean.model.free_surface.η), state.eta)

# Reset both clocks (critical: OceanSeaIceModel has separate clocks)
simulation.model.clock.time = t_start
simulation.model.clock.iteration = 0
ocean.model.clock.time = t_start
ocean.model.clock.iteration = 0
simulation.stop_time = t_start + Dt  # single timestep

@info "Compiling (warm-up timestep)..."
run!(simulation)
@info "Compilation complete."

# --- Process each member ---
for m in member_start:member_end
    @info "=== Member $m (cycle $cycle) ==="

    # Load member state
    state = load_member_state(member_filename(prev_cycle_dir, m))

    # Set state
    set!(ocean.model, T=state.T, S=state.S, e=state.e, u=state.u, v=state.v)
    copyto!(interior(ocean.model.velocities.w), state.w)
    copyto!(interior(ocean.model.free_surface.η), state.eta)

    # Reset both clocks
    simulation.model.clock.time = t_start
    simulation.model.clock.iteration = 0
    ocean.model.clock.time = t_start
    ocean.model.clock.iteration = 0
    simulation.stop_time = t_end

    # Run forecast
    run!(simulation)

    # Extract forecast state to CPU
    T_fc = Array(interior(ocean.model.tracers.T))
    S_fc = Array(interior(ocean.model.tracers.S))
    e_fc = Array(interior(ocean.model.tracers.e))
    u_fc = Array(interior(ocean.model.velocities.u))
    v_fc = Array(interior(ocean.model.velocities.v))
    w_fc = Array(interior(ocean.model.velocities.w))
    eta_fc = Array(interior(ocean.model.free_surface.η))

    # Save forecast state
    save_member_state(member_filename(fc_dir, m);
        T=T_fc, S=S_fc, e=e_fc, u=u_fc, v=v_fc, w=w_fc, eta=eta_fc)

    @info "  Member $m done. T range: ($(minimum(T_fc)), $(maximum(T_fc)))"
end

@info "Member forecast complete. Members $member_start:$member_end saved to $fc_dir"
