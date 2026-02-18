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

## References

- Halliwell et al. (2017): OSSE for evaluating ocean observing systems — satellite altimetry has greatest impact, then Argo, then SST
- Mainelli et al.: OHC improves SHIPS intensity forecasts by 5-20%
- Oke et al. (2019): Non-identical twin experiments show subsurface profiles add 10-20% beyond SSH+SST
- Jean-Michel et al. (2021): GLORYS12 reanalysis — subsurface T errors 0.45-0.75 C
- Deser et al. (2003): SST persistence timescale ~11-14 days
- Storto et al. (2019): Synthesis of operational ocean DA systems
