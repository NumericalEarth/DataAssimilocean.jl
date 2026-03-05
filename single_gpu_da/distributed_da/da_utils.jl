# Shared utilities for distributed ensemble DA
#
# Contains: DA configuration, ETKF algorithm, observation extraction,
# perturbation profile, and member state I/O.

using Random
using Statistics
using LinearAlgebra
using JLD2
using Dates
using Printf

# ============================================================================
#                           CONFIGURATION
# ============================================================================

const DA_N_ENS = 10                   # Ensemble members
const DA_N_CYCLES = 14                # Number of analysis cycles
const DA_ANALYSIS_WINDOW = 1          # days (used as Float64 seconds below)
const DA_ANALYSIS_WINDOW_SECONDS = DA_ANALYSIS_WINDOW * 86400.0
const DA_DT_MINUTES = 5               # Model time step in minutes
const DA_INFLATION = 1.05             # Multiplicative inflation (post-analysis, perturbations only)

# Observation errors (instrument + representativeness for 1/4 deg grid)
const DA_SIGMA_T = 0.5                # T observation error (C)
const DA_SIGMA_S = 0.05               # S observation error (psu)

# Argo-like observation network
const DA_N_PROFILES_PER_CYCLE = 76
const DA_OBS_DEPTH_LIMIT = -1000.0    # meters: observe upper 1000m only
const DA_OBS_VERTICAL_SKIP = 4        # observe every Nth level (reduces overdetermination)
const DA_OBS_SEED = 12345             # base seed for reproducible profile locations

# Perturbation parameters: depth-dependent, subsurface-focused
const DA_T_NOISE_STD = 0.5            # C, ensemble spread
const DA_S_NOISE_STD = 0.05           # PSU, ensemble spread
const DA_T_BIAS_MAX = 1.5             # C, bias at thermocline
const DA_S_BIAS_MAX = 0.1             # PSU, bias at thermocline
const DA_Z_PEAK = -150.0              # depth of maximum perturbation (m)

# Localization
const DA_LOCALIZATION_RADIUS = 1000.0 # km, Gaspari-Cohn cutoff radius
const DA_EARTH_RADIUS = 6371.0        # km

# Dates
const DA_RESOLUTION = 0.25
const DA_START_DATE = DateTime(2017, 8, 25)

# ============================================================================
#                           ETKF ALGORITHM
# ============================================================================

"""
    compute_etkf_weights(Y_b, y_obs, R_inv_diag)

Compute ETKF weight matrix following Hunt et al. (2007).
Y_b: (n_obs, N_ens), y_obs: (n_obs,), R_inv_diag: (n_obs,)
Returns W: (N_ens, N_ens)
"""
function compute_etkf_weights(Y_b, y_obs, R_inv_diag)
    N = size(Y_b, 2)

    # Ensemble mean and perturbations in observation space
    y_mean = vec(mean(Y_b, dims=2))
    S = Y_b .- y_mean  # (n_obs, N)

    # Innovation
    d = y_obs .- y_mean  # (n_obs,)

    # C = S' R^-1 S  (N x N)
    RinvS = R_inv_diag .* S  # (n_obs, N)
    C = S' * RinvS           # (N, N)

    # (N-1)I + C
    invTTt = (N - 1) * I + C

    # Eigendecomposition
    F = eigen(Symmetric(invTTt))
    inv_sigma = 1.0 ./ F.values
    U = F.vectors

    # Mean weight: w_mean = U diag(inv_sigma) U' S' R^-1 d
    RinvD = R_inv_diag .* d
    w_mean = U * (inv_sigma .* (U' * (S' * RinvD)))

    # Perturbation weight: W_pert = U diag(sqrt((N-1) inv_sigma)) U'
    W_pert = U * Diagonal(sqrt.((N - 1) .* inv_sigma)) * U'

    # Full weight: W = w_mean * 1' + W_pert
    W = w_mean * ones(N)' + W_pert

    return W
end

"""
    apply_etkf_update!(ensemble_field, W, inflation)

Apply ETKF weight matrix to update a 3D field for all ensemble members.
Inflation is applied to perturbations around the analysis mean (post-analysis),
NOT to the full weight matrix (which would overshoot the analysis mean).
"""
function apply_etkf_update!(ensemble_field, W, inflation)
    N = length(ensemble_field)

    # Save forecast (we overwrite in place)
    forecast = [copy(f) for f in ensemble_field]
    field_mean = mean(forecast)

    # Step 1: Apply ETKF weights (no inflation) to get analysis
    for m in 1:N
        ensemble_field[m] .= field_mean
        for n in 1:N
            ensemble_field[m] .+= W[n, m] .* (forecast[n] .- field_mean)
        end
    end

    # Step 2: Inflate perturbations around analysis mean
    if inflation != 1.0
        analysis_mean = mean(ensemble_field)
        for m in 1:N
            ensemble_field[m] .= analysis_mean .+ inflation .* (ensemble_field[m] .- analysis_mean)
        end
    end

    return nothing
end

# ============================================================================
#                           OBSERVATIONS
# ============================================================================

"""
    extract_profile_observations!(obs_vector, field_3d, i_global, j_global, k_obs)

Extract observations at Argo-like profile locations from a 3D field.
Layout: obs_vector[(p-1)*n_levels + l] = field_3d[i, j, k_obs[l]]
"""
function extract_profile_observations!(obs_vector, field_3d, i_global, j_global, k_obs)
    n_profiles = length(i_global)
    n_levels = length(k_obs)
    fill!(obs_vector, 0.0)

    for p in 1:n_profiles
        i = i_global[p]
        j = j_global[p]
        for (li, k) in enumerate(k_obs)
            obs_vector[(p-1)*n_levels + li] = field_3d[i, j, k]
        end
    end
end

"""
Depth-dependent perturbation profile. Maximum at z_peak, zero at surface,
decaying below. Returns value in [0, 1].
"""
function perturbation_profile(z; z_peak=DA_Z_PEAK)
    z >= 0 && return 0.0
    d = -z / (-z_peak)
    return d * exp(1 - d)
end

# ============================================================================
#                           LOCALIZATION (LETKF)
# ============================================================================

"""
    gaspari_cohn(r)

Gaspari-Cohn correlation function. Input r = distance/cutoff_radius (0 to 1).
Returns 0 for r >= 1, smooth taper for r < 1.
"""
function gaspari_cohn(r)
    r = abs(r)
    r >= 1.0 && return 0.0
    # Gaspari-Cohn 5th order piecewise polynomial (compact support at 2c,
    # but we normalize so cutoff is at r=1 by using r_scaled = 2r)
    z = 2.0 * r
    if z <= 1.0
        return 1.0 - 5.0/3.0*z^2 + 5.0/8.0*z^3 + z^4/2.0 - z^5/4.0
    else
        return 4.0 - 5.0*z + 5.0/3.0*z^2 + 5.0/8.0*z^3 - z^4/2.0 + z^5/12.0 - 2.0/(3.0*z)
    end
end

"""
    haversine_km(lon1, lat1, lon2, lat2)

Great-circle distance in km between two points given in degrees.
"""
function haversine_km(lon1, lat1, lon2, lat2)
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2
    return 2.0 * DA_EARTH_RADIUS * asin(sqrt(a))
end

"""
    apply_letkf_update!(ensemble_T, ensemble_S, Y_b, y_obs, R_inv, valid_mask,
                         i_profiles, j_profiles, k_obs, n_obs_levels,
                         lon_centers, lat_centers, lon_profiles, lat_profiles,
                         inflation, loc_radius_km)

Local ETKF (LETKF): for each grid column, find nearby observations,
apply Gaspari-Cohn localization, and compute local ETKF weights.
Updates ensemble_T and ensemble_S in place.
"""
function apply_letkf_update!(ensemble_T, ensemble_S, Y_b, y_obs, R_inv, valid_mask,
                              i_profiles, j_profiles, k_obs, n_obs_levels,
                              lon_centers, lat_centers, lon_profiles, lat_profiles,
                              inflation, loc_radius_km)
    N_ens = length(ensemble_T)
    Nx, Ny, Nz = size(ensemble_T[1])
    n_obs_per_field = length(i_profiles) * n_obs_levels
    n_obs_total = 2 * n_obs_per_field

    # Pre-compute profile distances won't change per column
    n_profiles = length(i_profiles)

    # Save forecast copies
    forecast_T = [copy(f) for f in ensemble_T]
    forecast_S = [copy(f) for f in ensemble_S]
    T_mean = mean(forecast_T)
    S_mean = mean(forecast_S)

    n_updated = 0
    n_skipped = 0

    for j in 1:Ny
        for i in 1:Nx
            # Grid column location
            col_lon = lon_centers[i]
            col_lat = lat_centers[j]

            # Find nearby profiles and compute localization weights
            local_obs_indices = Int[]
            local_loc_weights = Float64[]

            for p in 1:n_profiles
                dist = haversine_km(col_lon, col_lat, lon_profiles[p], lat_profiles[p])
                r = dist / loc_radius_km
                gc = gaspari_cohn(r)
                if gc > 0.0
                    # Add all levels of this profile (T and S)
                    for l in 1:n_obs_levels
                        t_idx = (p-1)*n_obs_levels + l        # T obs index
                        s_idx = n_obs_per_field + t_idx        # S obs index
                        if valid_mask[t_idx]  # T and S share same land mask
                            push!(local_obs_indices, t_idx)
                            push!(local_loc_weights, gc)
                            push!(local_obs_indices, s_idx)
                            push!(local_loc_weights, gc)
                        end
                    end
                end
            end

            n_local = length(local_obs_indices)
            if n_local == 0
                n_skipped += 1
                continue
            end

            # Extract local obs subset
            Y_b_local = Y_b[local_obs_indices, :]
            y_obs_local = y_obs[local_obs_indices]
            R_inv_local = R_inv[local_obs_indices] .* local_loc_weights

            # Compute local ETKF weights
            W = compute_etkf_weights(Y_b_local, y_obs_local, R_inv_local)

            # Apply weights to this column for all k levels
            for k in 1:Nz
                # T update: apply weights (no inflation), then inflate around analysis mean
                t_vals = [forecast_T[m][i, j, k] for m in 1:N_ens]
                t_mean = T_mean[i, j, k]
                for m in 1:N_ens
                    ensemble_T[m][i, j, k] = t_mean
                    for n in 1:N_ens
                        ensemble_T[m][i, j, k] += W[n, m] * (t_vals[n] - t_mean)
                    end
                end
                # Inflate perturbations around analysis mean
                t_a_mean = sum(ensemble_T[m][i, j, k] for m in 1:N_ens) / N_ens
                for m in 1:N_ens
                    ensemble_T[m][i, j, k] = t_a_mean + inflation * (ensemble_T[m][i, j, k] - t_a_mean)
                end

                # S update: same pattern
                s_vals = [forecast_S[m][i, j, k] for m in 1:N_ens]
                s_mean = S_mean[i, j, k]
                for m in 1:N_ens
                    ensemble_S[m][i, j, k] = s_mean
                    for n in 1:N_ens
                        ensemble_S[m][i, j, k] += W[n, m] * (s_vals[n] - s_mean)
                    end
                end
                s_a_mean = sum(ensemble_S[m][i, j, k] for m in 1:N_ens) / N_ens
                for m in 1:N_ens
                    ensemble_S[m][i, j, k] = s_a_mean + inflation * (ensemble_S[m][i, j, k] - s_a_mean)
                end
            end

            n_updated += 1
        end
    end

    @info "  LETKF: updated $n_updated columns, skipped $n_skipped (no nearby obs)"
    return nothing
end

# ============================================================================
#                           MEMBER STATE I/O
# ============================================================================

"""
    save_member_state(filename; T, S, e, u, v, w, eta)

Save a single ensemble member's state to a JLD2 file.
All arrays should be CPU arrays (use Array(interior(field))).
"""
function save_member_state(filename; T, S, e, u, v, w, eta)
    jldsave(filename; T, S, e, u, v, w, eta)
end

"""
    load_member_state(filename)

Load a single ensemble member's state from a JLD2 file.
Returns NamedTuple (T, S, e, u, v, w, eta).
"""
function load_member_state(filename)
    jldopen(filename) do file
        (T   = file["T"],
         S   = file["S"],
         e   = file["e"],
         u   = file["u"],
         v   = file["v"],
         w   = file["w"],
         eta = file["eta"])
    end
end

"""
    member_filename(dir, member_id)

Standard filename for a member state file: member_01.jld2, member_02.jld2, ...
"""
member_filename(dir, member_id) = joinpath(dir, @sprintf("member_%02d.jld2", member_id))

"""
    forecast_observations_filename(dir, member_id)

Standard filename for forecast observations: obs_member_01.jld2, ...
"""
forecast_observations_filename(dir, member_id) = joinpath(dir, @sprintf("obs_member_%02d.jld2", member_id))

"""
    cycle_dirname(base_dir, cycle)

Standard directory name for a cycle: cycle_00, cycle_01, ...
"""
cycle_dirname(base_dir, cycle) = joinpath(base_dir, @sprintf("cycle_%02d", cycle))

"""
    forecast_dirname(base_dir, cycle)

Standard directory name for forecast output: forecast_01, forecast_02, ...
"""
forecast_dirname(base_dir, cycle) = joinpath(base_dir, @sprintf("forecast_%02d", cycle))
