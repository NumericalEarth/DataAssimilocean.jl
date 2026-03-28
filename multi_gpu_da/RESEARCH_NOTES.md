# Ocean DA Problem Design: Research Notes

## The Problem with Our Current Setup

Our current ETKF experiment applies surface-biased perturbations (0.3 C bias + 0.5 C noise in the upper 20 levels) and observes surface fields. The RMSE decreases from 0.355 to 0.241 C over 14 days, but **this is mostly the atmosphere doing the work, not the DA**. SST anomalies have a persistence timescale of ~11-14 days under prescribed atmospheric forcing — the perturbations decay naturally regardless of DA.

This makes the problem unrealistically easy and the DA results uncompelling.

## What Makes Ocean DA Hard (and Interesting)

### The subsurface is the real challenge

| Variable | Observability | Persistence | Hurricane Impact |
|----------|--------------|-------------|------------------|
| SST | Excellent (daily satellite) | Days-weeks | Moderate (boundary condition) |
| Mixed layer depth | Poor (Argo only) | Days-weeks | High (controls SST cooling) |
| Thermocline depth (50-300m) | Poor (Argo only) | Months-years | Critical (controls OHC) |
| Ocean heat content (OHC) | Inferred from SSH + profiles | Months-years | Critical (fuel for hurricanes) |
| Subsurface T at 200m | Poor (Argo every 10 days) | Months-years | High |
| Eddy positions | Moderate (SSH altimetry) | Weeks-months | High (warm core rings) |

**Key insight**: Two locations with identical SST but different subsurface structure produce dramatically different hurricane responses. A warm SST over a shallow warm layer (thin mixed layer) cools rapidly under hurricane winds, weakening the storm. A warm SST over a deep warm layer (high OHC) sustains intensity — this is how Hurricane Opal intensified from Cat 1 to Cat 4 in 14 hours over a warm-core ring.

### Typical error magnitudes in ocean reanalyses

| Depth | Temperature RMSE | Notes |
|-------|-----------------|-------|
| Surface (SST) | 0.2-0.5 C | Well-constrained by satellites |
| 0-100 m | 0.3-0.5 C | Constrained by SST + Argo |
| 100-300 m (thermocline) | 0.5-1.0 C | **Largest errors** — hardest to constrain |
| 300-1000 m | 0.3-0.5 C | Less variability |
| Salinity | 0.1 PSU global avg | Larger in halocline regions |
| SSH | 2-5 cm RMSE | After assimilating altimetry |
| Mixed layer depth | 10-30 m | Highly variable |

### Persistence of perturbations

| Perturbation | Persistence Without DA | Recovery With DA |
|-------------|----------------------|------------------|
| SST (0.5-2 C) | Days-weeks (atm damping) | Days (if atm forcing correct) |
| Mixed layer T (1 C) | Weeks | ~1 week with SST obs |
| Thermocline depth (20 m) | **Months-years** | Weeks-months (needs profiles or SSH) |
| Subsurface T at 200 m (1 C) | **Months-years** | Months (needs Argo profiles) |
| Eddy position (50 km shift) | Weeks-months | Weeks (needs SSH) |

## Recommended Problem Redesign

### 1. Perturbation strategy: subsurface-focused

Instead of surface bias + noise, perturb the **subsurface thermal structure**:
- **Zero perturbation at surface** (SST is well-observed, not the challenge)
- **Maximum perturbation at 100-300 m** (thermocline region): 1-2 C
- **Spatially correlated** with decorrelation length ~200 km (mesoscale)
- **Depth profile**: ramp from 0 at surface to max at ~150m, decay below 500m
- This mimics realistic errors in thermocline depth / OHC

Example perturbation profile (temperature):
```
z = 0 m:     0.0 C  (surface — well observed)
z = -50 m:   0.3 C
z = -100 m:  0.8 C
z = -150 m:  1.5 C  (maximum — thermocline region)
z = -200 m:  1.2 C
z = -300 m:  0.7 C
z = -500 m:  0.2 C
z = -1000 m: 0.0 C
```

### 2. Observation network: multi-platform

Mimic the real operational observation network:

| Platform | Frequency | Spacing | Obs Error | Variables |
|----------|-----------|---------|-----------|-----------|
| Satellite SST | Daily | Full grid (thinned 4x) | 0.3 C | SST only |
| Argo profiles | Every 10 days | ~300 km (~76/day in our domain) | 0.3 C (T), 0.01 PSU (S) | T/S to 1000m |
| Satellite SSH | Every 10 days | Along-track (~7 km spacing) | 2 cm | Sea surface height |

SSH observations are particularly powerful because they project subsurface information — a warm-core eddy has a positive SSH anomaly of ~10-30 cm that maps to the entire vertical temperature structure through geostrophic balance.

### 3. Analysis window: 3-7 days

Operational ocean 4DVAR systems use:
- NCEP RTOFS: 24-hour windows (3DVAR)
- Mercator GLORYS12: 7-day windows (reduced-order Kalman filter)
- ROMS 4DVAR research: 3-7 day windows
- ECMWF ORAS5: 5-day windows

For our variational approach, **3-7 day windows** are standard. Longer windows allow more dynamical information propagation but strain the tangent-linear assumption.

### 4. Success metrics

Instead of just SST RMSE, measure:
1. **Temperature RMSE by depth** (surface, 50m, 100m, 200m, 500m)
2. **Ocean Heat Content error** (integral of T above 26 C from surface to 26 C isotherm)
   - TCHP > 60 kJ/cm2 supports TC intensification
   - TCHP > 90 kJ/cm2 associated with rapid intensification
3. **Mixed layer depth error**
4. **Thermocline depth error** (depth of max dT/dz)

### 5. Hurricane Irma context

Hurricane Irma (Sept 2017) context for our domain (Atlantic, 100W-0E, 10S-60N):
- Irma tracked through our domain Sept 4-10, 2017
- Our spinup starts Aug 25, nature run covers Sept 8-22
- **Pre-storm OHC** in Irma's path was critical: the storm weakened over cold wakes from prior hurricanes (Jose) and intensified over the Loop Current / warm core eddies
- The scientifically compelling question: **can we recover the pre-storm subsurface ocean state from sparse observations, well enough to predict the ocean's response to hurricane forcing?**

## Implementation Plan

### Phase 1: Better perturbations (current priority)
- [ ] Implement depth-dependent perturbation profile (zero at surface, max at thermocline)
- [ ] Add spatially correlated perturbations (use Gaussian random field with ~200 km correlation)
- [ ] Verify perturbations persist over 14-day window (unlike current surface perturbations)

### Phase 2: Multi-platform observations
- [ ] Keep Argo-like profiles (already implemented)
- [ ] Add satellite SST observations (daily, thinned surface grid)
- [ ] Add SSH observations (requires computing model SSH from free surface eta)

### Phase 3: Better DA algorithm
- [ ] Integrate localization (via Manteia.jl) — critical for preventing ensemble collapse
- [ ] Scale to 100 ensemble members
- [ ] Or: switch to variational (4DVAR) method using Reactant.jl / Enzyme

### Phase 4: Hurricane-relevant metrics
- [ ] Compute and track OHC, mixed layer depth, thermocline depth
- [ ] Compare DA improvement in terms of hurricane-relevant quantities
- [ ] Frame results in terms of intensity forecast improvement

## ETKF vs EnKF and Operational Ocean DA Methods

### Ensemble Kalman Filter Variants

| Method | Type | Key Feature | Pros | Cons |
|--------|------|-------------|------|------|
| **EnKF** (Evensen 1994) | Stochastic | Perturbed observations | Simple to implement | Sampling noise from perturbed obs |
| **ETKF** (Bishop et al. 2001) | Deterministic square root | Transform in ensemble space | No perturbed obs noise; works in N_ens space | Can cluster in nonlinear regimes |
| **EnSRF** (Whitaker & Hamill 2002) | Deterministic square root | Serial obs processing | Exact posterior covariance | Serial processing limits parallelism |
| **LETKF** (Hunt et al. 2007) | Localized ETKF | Local analysis at each grid point | Embarrassingly parallel; natural localization | Must choose localization radius |
| **DEnKF** (Sakov & Oke 2008) | Deterministic | Half the anomaly update | Simple; avoids square root | Approximate |
| **EAKF** (Anderson 2001) | Deterministic | Adjustment Kalman filter | Exact marginal posteriors | |
| **SEEK/SEIK** (Pham et al. 1998) | Reduced-rank | Static or slowly-evolving ensemble | Works with few modes (7-30) | Less flow-dependent than full EnKF |

**ETKF** (what we use): Computes analysis entirely in ensemble space. The weight matrix is N_ens × N_ens, so computational cost scales with ensemble size, not state or observation dimension. No perturbed observations means no sampling noise, but the deterministic transform can cause ensemble members to cluster in highly nonlinear regimes.

**EnKF** (original): Each member gets different perturbed observations (y + ε, ε ~ N(0,R)), introducing sampling noise that grows as observation error decreases or observation count increases. With many observations relative to ensemble size, this noise can dominate.

**LETKF**: The most popular variant for large-scale geophysical DA. Performs ETKF independently at each grid point using only nearby observations (localization). Embarrassingly parallel — each grid point is independent. This is what we should target if staying with ensemble methods.

### What Operational Ocean DA Systems Use

| System | Organization | Method | Ensemble Size | Window | Resolution |
|--------|-------------|--------|---------------|--------|------------|
| **RTOFS** | NCEP/USA | 3DVAR (NCODA) | None | 24 hr | 1/12 deg |
| **GOFS 3.1** | US Navy | 3DVAR (NCODA) | None | 24 hr | 1/12 deg (HYCOM) |
| **FOAM** | Met Office/UK | 3DVAR FGAT (NEMOVAR) | None | 1 day | 1/12 deg (NEMO) |
| **ORAS5** | ECMWF | 3DVAR FGAT + bias correction | 5 (for uncertainty) | 5 days | 1/4 deg (NEMO) |
| **GLORYS12** | Mercator/France | **SEEK** + 3DVAR bias correction | 7 modes (reduced-rank) | 7 days | 1/12 deg (NEMO) |
| **TOPAZ** | NERSC/Norway | **DEnKF** | **100 members** | 7 days | ~12 km (HYCOM) |
| **BRAN** | BOM/Australia | EnOI → hybrid EnKF | Static ensemble | 3 days | 1/10 deg |
| **ROMS 4DVAR** | Research | 4DVAR (adjoint) | None | 3-7 days | Variable |

### Key Observations

1. **Most operational systems use variational methods (3DVAR/4DVAR)**, not ensemble methods. This is because:
   - Variational methods don't suffer from ensemble collapse
   - They can handle millions of observations without rank deficiency
   - The adjoint model provides exact gradient information
   - They've been operational longer and are more mature

2. **GLORYS12** (the most widely-used ocean reanalysis) uses **SEEK** — a reduced-rank Kalman filter with only 7 ensemble modes (not full ensemble members). These modes are computed from a long free-run simulation and represent the dominant ocean variability patterns. Combined with 3DVAR for bias correction on a 7-day window.

3. **TOPAZ is the exception**: Uses a full dynamical DEnKF with 100 ensemble members. This is the closest to what we're doing but at operational scale. They use:
   - 100 members (our target)
   - Localization (critical — we need this)
   - 7-day analysis windows
   - DEnKF instead of ETKF (deterministic, no square root)

4. **The trend** in operational ocean DA is toward hybrid ensemble-variational methods:
   - Use ensemble covariances for flow-dependent background error
   - Embed in variational framework for observation handling
   - Avoids pure ensemble limitations while capturing flow dependence

5. **4DVAR is powerful but expensive**: Requires developing and maintaining the adjoint (tangent-linear) model. This is exactly what Reactant.jl / Enzyme could enable — automatic differentiation eliminates the need for hand-coded adjoints, which is the main barrier to 4DVAR in ocean models.

### Implications for Our Work

- **Short term (ETKF)**: We need localization (LETKF via Manteia.jl) and more ensemble members (100, like TOPAZ) to make the ensemble approach viable
- **Long term (4DVAR)**: The differentiable workflow with Reactant.jl/Enzyme is scientifically the most compelling direction — it would be the first automatic-differentiation-based 4DVAR for a full-complexity ocean model, leapfrogging the hand-coded adjoints that limit current operational systems
- **The ETKF experiments serve as a baseline** to demonstrate what ensemble methods can achieve (and where they struggle), motivating the variational approach

## References

- Halliwell et al. (2017): OSSE for evaluating ocean observing systems — satellite altimetry has greatest impact, then Argo, then SST
- Mainelli et al.: OHC improves SHIPS intensity forecasts by 5-20%
- Oke et al. (2019): Non-identical twin experiments show subsurface profiles add 10-20% beyond SSH+SST
- Jean-Michel et al. (2021): GLORYS12 reanalysis — subsurface T errors 0.45-0.75 C
- Deser et al. (2003): SST persistence timescale ~11-14 days
- Storto et al. (2019): Synthesis of operational ocean DA systems
- Evensen (1994): Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods
- Bishop et al. (2001): Adaptive sampling with the ensemble transform Kalman filter
- Hunt et al. (2007): Efficient data assimilation for spatiotemporal chaos: a local ensemble transform Kalman filter
- Pham et al. (1998): Singular evolutive extended Kalman filter (SEEK)
- Sakov & Oke (2008): A deterministic formulation of the ensemble Kalman filter (DEnKF)
- Sakov et al. (2012): TOPAZ4 — an ocean-sea ice data assimilation system for the North Atlantic and Arctic
