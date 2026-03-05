# Initialize ensemble: create perturbed member states from spinup
#
# Usage:
#   julia --project single_gpu_da/distributed_da/initialize_ensemble.jl \
#       SPINUP_DIR OUTPUT_DIR [N_ENS]
#
# Creates OUTPUT_DIR/cycle_00/member_01.jld2 ... member_NN.jld2

include(joinpath(@__DIR__, "..", "..", "experiments", "irma_utils.jl"))
include(joinpath(@__DIR__, "da_utils.jl"))

using Oceananigans.Grids: znodes
using Printf
using Dates

spinup_dir = ARGS[1]
output_dir = ARGS[2]
N_ens = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : DA_N_ENS

@info "Initializing $N_ens ensemble members"
@info "  Spinup dir: $spinup_dir"
@info "  Output dir: $output_dir"

# --- Setup grid (needed for vertical levels) ---
arch = GPU()
grid = create_grid(arch, DA_RESOLUTION)
Nx, Ny, Nz = size(grid)

z_faces = Array(znodes(grid, Face()))
z_centers = [0.5 * (z_faces[k] + z_faces[k+1]) for k in 1:Nz]

# --- Load spinup state ---
@info "Loading spinup state..."
spinup_state = load_model_state(joinpath(spinup_dir, "final_state.jld2"))
Nx_state, Ny_state, Nz_state = size(spinup_state.T)

# --- Create cycle_00 directory ---
cycle_dir = cycle_dirname(output_dir, 0)
mkpath(cycle_dir)

# --- Generate perturbed members ---
for i in 1:N_ens
    rng = MersenneTwister(42 + i)

    T_i = copy(spinup_state.T)
    S_i = copy(spinup_state.S)

    for k in 1:Nz_state
        alpha = perturbation_profile(z_centers[k]; z_peak=DA_Z_PEAK)
        T_i[:, :, k] .+= alpha .* (DA_T_NOISE_STD .* randn(rng, Nx_state, Ny_state) .+ DA_T_BIAS_MAX)
        S_i[:, :, k] .+= alpha .* (DA_S_NOISE_STD .* randn(rng, Nx_state, Ny_state) .+ DA_S_BIAS_MAX)
    end

    fname = member_filename(cycle_dir, i)
    save_member_state(fname;
        T   = T_i,
        S   = S_i,
        e   = copy(spinup_state.e),
        u   = copy(spinup_state.u),
        v   = copy(spinup_state.v),
        w   = copy(spinup_state.w),
        eta = copy(spinup_state.η))

    @info "  Member $i saved: T range ($(minimum(T_i)), $(maximum(T_i)))"
end

@info "Ensemble initialization complete. $N_ens members in $cycle_dir"
