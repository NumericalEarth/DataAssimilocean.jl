# Free run: 14-day simulation from noised/biased initial condition (no DA)
#
# This is the baseline that the DA must beat.
# Uses the same model setup as the nature run but with perturbed ICs.
#
# Perturbation applied to the upper 20 levels:
#   T += N(0, 0.5) + 0.3  (noise + warm bias)
#   S += N(0, 0.05) + 0.02 (noise + fresh bias)
#
# Usage:
#   julia --project single_gpu_da/free_run.jl SPINUP_DIR OUTPUT_DIR

include(joinpath(@__DIR__, "..", "experiments", "irma_utils.jl"))

using Random

spinup_dir = ARGS[1]
output_dir = ARGS[2]
mkpath(output_dir)

# Configuration
arch = GPU()
resolution = 0.25
run_days = 14
Δt = 5minutes

# Perturbation parameters
T_noise_std = 0.5    # degC
S_noise_std = 0.05   # psu
T_bias = 0.3         # degC (warm bias)
S_bias = 0.02        # psu
noise_depth_levels = 20

# Dates
start_date = DateTime(2017, 8, 25)
end_date = start_date + Day(run_days)

@info "Free run: $start_date to $end_date ($run_days days)"
@info "Perturbation: T noise=$(T_noise_std)°C, bias=$(T_bias)°C; S noise=$(S_noise_std) psu, bias=$(S_bias) psu"

# Grid (reuse cached bathymetry)
grid = create_grid(arch, resolution)
grid = load_or_compute_bathymetry(grid, spinup_dir)
Nx, Ny, Nz = size(grid)

# Ocean model
ocean = create_ocean_simulation(grid, start_date, end_date; with_restoring=true)

# Load spinup state and add perturbation
@info "Loading spinup state and adding perturbation..."
state = load_model_state(joinpath(spinup_dir, "final_state.jld2"))

# Perturb T and S in the upper ocean
Random.seed!(12345)  # Reproducible
T_perturbed = copy(state.T)
S_perturbed = copy(state.S)

Nz_state = size(T_perturbed, 3)
k_start = max(1, Nz_state - noise_depth_levels + 1)

for k in k_start:Nz_state
    T_perturbed[:, :, k] .+= T_noise_std .* randn(size(T_perturbed, 1), size(T_perturbed, 2)) .+ T_bias
    S_perturbed[:, :, k] .+= S_noise_std .* randn(size(S_perturbed, 1), size(S_perturbed, 2)) .+ S_bias
end

@info "  Perturbed T range: ($(minimum(T_perturbed)), $(maximum(T_perturbed)))"
@info "  Perturbed S range: ($(minimum(S_perturbed)), $(maximum(S_perturbed)))"

# Set perturbed state
set!(ocean.model, T=T_perturbed, S=S_perturbed, e=state.e, u=state.u, v=state.v)
copyto!(interior(ocean.model.velocities.w), state.w)
copyto!(interior(ocean.model.free_surface.η), state.η)

# Save perturbed initial condition for reference
jldsave(joinpath(output_dir, "initial_perturbation.jld2");
        T_perturbed, S_perturbed,
        T_noise_std, S_noise_std, T_bias, S_bias,
        noise_depth_levels, seed=12345)

# Atmosphere
@info "Setting up JRA55 atmosphere..."
atmosphere, radiation = create_atmosphere(arch, start_date, end_date)

# Coupled model
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time=run_days * days)

# Progress
add_progress_callback!(simulation, start_date)

# Output
setup_output_writers!(ocean, coupled_model, output_dir;
                      surface_interval=6hour, flux_interval=6hour)

# Run
@info "Running free run (no DA)..."
run!(simulation)

# Save final state
save_model_state(ocean, joinpath(output_dir, "final_state.jld2"))

# === Animation ===
@info "Generating free run animation..."

using CairoMakie
using Oceananigans.OutputReaders: FieldTimeSeries

try
    surface_file = joinpath(output_dir, "surface_fields.jld2")
    Tt = FieldTimeSeries(surface_file, "T")
    times = Tt.times
    Nt = length(times)

    n = Observable(1)
    Tn = @lift interior(Tt[$n], :, :, 1)

    title = @lift begin
        day_num = times[$n] / (24 * 3600)
        current_date = start_date + Day(round(Int, day_num))
        "Free Run SST — Day $(round(day_num, digits=1)) ($current_date)"
    end

    fig = Figure(size=(900, 500))
    Label(fig[0, :], title, fontsize=18, tellwidth=false)
    ax = Axis(fig[1, 1], xlabel="i", ylabel="j")
    hm = heatmap!(ax, Tn, colormap=:thermal, colorrange=(10, 32))
    Colorbar(fig[1, 2], hm, label="SST (°C)")

    CairoMakie.record(fig, joinpath(output_dir, "free_run_animation.mp4"),
                       1:Nt, framerate=8) do nn
        n[] = nn
    end

    n[] = Nt
    save(joinpath(output_dir, "free_run_final.png"), fig)
    @info "Animation saved to $(joinpath(output_dir, "free_run_animation.mp4"))"
catch e
    @warn "Animation failed (non-fatal)" exception=(e, catch_backtrace())
end

@info "Free run complete!"
