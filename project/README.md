# Strategy: Atlantic-basin regional ocean state estimation and forward simulation

This document summarizes the simulation strategy discussed in-chat for an **Atlantic-basin** regional ocean model experiment aimed at emulating **ocean state estimation / initialization prior to a hurricane**, and then **forward-simulating ocean evolution** through storm passage.

---

## 1) Background and purpose

### Objective
- Emulate the *ocean-side* of a tropical-cyclone modeling workflow: estimate (or approximate) the **pre-storm ocean initial condition**, then run a **regional forward ocean simulation** through the storm period.
- Motivation: tropical-cyclone intensity is strongly modulated by **upper-ocean stratification**, mixed-layer depth, and mesoscale structures (e.g., Loop Current / Gulf Stream / rings). This aligns with the modeling philosophy behind NOAA’s HAFS coupled system (atmosphere + ocean).

### Why HAFS matters here
- HAFS operationally uses an ocean model (HYCOM) coupled to the atmosphere; HAFS documentation and peer-reviewed studies describing the **HAFS ocean component and domains** provide the reference “target domain” and the realism requirements.
- In the HAFS ocean-component evaluation paper (Kim et al.), the **NHC** HYCOM regional domain uses a **1/12° Mercator grid** and uses a **1/12° Mercator** regional grid (with hybrid vertical coordinates); the exact horizontal dimensions are specified in the paper. This is the starting point for our domain reasoning and for ensuring the Atlantic-basin subset is faithful to the HAFS basin geometry.

**Key reference (HAFS ocean domain and configuration):**
- Kim et al., *Ocean component of the first operational version of HAFS: Evaluation of HYCOM…* (Frontiers in Earth Science, 2024). (Use the uploaded PDF.)

**Key reference (GPU ocean dycore performance and dt scaling):**
- Silvestri et al., *A GPU-Based Ocean Dynamical Core for Routine Mesoscale-Resolving Climate…* (JAMES, 2025). (Use the uploaded PDF.)

---

## 2) Target forward simulation (64 GPUs): Atlantic-only “HAFS-like” regional domain

### Domain goal
- **Regional Atlantic-basin domain** based on the HAFS “NHC” grid footprint, **excluding the eastern Pacific**.
- Push the **northern boundary farther north than the HAFS NHC domain** used for hurricane operations to better capture **full Gulf Stream development** (and downstream eddy field), while staying regional.
- Practical working bounds (adjustable once the agent maps to an exact mesh):
  - **Longitude:** ~100°W to 0°E  (Atlantic basin)
  - **Latitude:** ~0°N to 70°N  (north boundary extended for Gulf Stream)

These bounds are intended as a *first-pass* domain definition; the agentic utility should implement the final extents consistently with the chosen grid/mesh conventions (e.g., unwrapped longitude in [0, 360), Mercator spacing, etc.).

### Forcing and initialization pipeline (data dependencies)
- **Atmospheric forcing:** JRA-55 reanalysis (surface winds, heat fluxes, precipitation, etc.)
- **Ocean initial conditions:** ECCO (state estimate / reanalysis)
- **Boundary treatment:**
  - **Top boundary:** air–sea fluxes from JRA-55
  - **Lateral boundaries:** nudging/relaxation (“restoring”) to ECCO or another large-scale product
  - **Bottom boundary:** topography + bottom drag; optional deep restoring if needed for stability

**Implementation note:** use **ClimaOcean** to download/prepare JRA-55 and ECCO, and to manage forcing + initial condition ingestion.

### Vertical grid
- **Nz = 200** vertical levels
- Stretched grid with higher resolution near the surface:
  - O(1–5 m) resolution in the upper mixed layer and thermocline region
  - increasing thickness with depth to O(100 m) at the bottom

### Candidate horizontal resolutions (production)
Two practical choices:

#### Option A: ~1.6 km class (maximal resolution under conservative memory assumptions)
- Target points-per-degree (meridional): ~68 ⇒ **Δy ≈ 1.64 km**
- Example grid (domain spans 100° × 70°):
  - **Nx = 6800**
  - **Ny = 4760**
  - **Nz = 200**
  - Total 3D cells: **6,473,600,000**
  - Cells/GPU (64 GPUs): **101,150,000**

**Time step estimate (baroclinic):**
- Silvestri reports a baroclinic dt reaching **270 s** at 1/12° after spin-up in their quasi-realistic test.
- Assuming dt ∝ Δx, a ~1.6 km grid suggests **Δt ≈ 50–60 s**.

**Throughput estimate (order-of-magnitude):**
- Scaling from Silvestri’s near-global benchmark (10 simulated years/day at 1/12°, 100 levels, 64 GPUs):
  - **≈ 0.246 simulated years/day**
  - **≈ 89.9 simulated days/day**
- Estimated wall time for **14 simulated days**:
  - **≈ 3.7 hours** (compute baseline; add overhead for I/O/diagnostics)

#### Option B: ~2 km class (more comfortable operational candidate)
- Target points-per-degree (meridional): ~56 ⇒ **Δy ≈ 2.0 km**
- Example grid:
  - **Nx = 5600**
  - **Ny = 3920**
  - **Nz = 200**
  - Total 3D cells: **4,390,400,000**
  - Cells/GPU (64 GPUs): **68,600,000**

**Time step estimate (baroclinic):** **Δt ≈ 60–70 s**.

**Throughput estimate:**
- **≈ 0.443 simulated years/day**
- **≈ 161.6 simulated days/day**
- Estimated wall time for **14 simulated days**:
  - **≈ 2.1 hours** (compute baseline; add overhead)

### Hardware targeting: Perlmutter
- Perlmutter GPU nodes provide **4× NVIDIA A100 GPUs with 40 GB HBM2 each** (so **40 GB per GPU**).
- Production runs will use **64 GPUs** (e.g., 16 Perlmutter GPU nodes).

Because GPU memory is 40 GB on Perlmutter, we should **not** assume the “233M cells/GPU” upper bound holds unless explicitly validated with an Oceananigans memory model; the configs above intentionally allow a 2 km fallback.

---

## 3) Prototype simulation setup (single GPU, same physical domain)

### Goal
Prototype the *exact same physical domain* (Atlantic-only) on a **single GPU** with **80 GB** memory, at reduced horizontal resolution, while preserving:
- boundary/forcing machinery (ClimaOcean + JRA-55 + ECCO)
- vertical grid approach (Nz=200, surface refinement)
- numerics as close as possible to production

### Example “1 GPU / 80 GB” prototype grid
Choose a coarse grid that remains faithful to the domain extents:

- **Longitude span:** 100°
- **Latitude span:** 70°
- Example:
  - **Nx = 2048**
  - **Ny = 1024**
  - **Nz = 200**
  - Total cells: 2048×1024×200 = **419,430,400**

This corresponds to:
- Δφ ≈ 70° / 1024 ≈ 0.068° ⇒ **~7.6 km** in y
- Δλ ≈ 100° / 2048 ≈ 0.049° ⇒ **~3–5 km** in x depending on latitude

**Time step (baroclinic):**
- Relative to 1/12° (~8 km), this is comparable/coarser in at least one direction, so **Δtₗ ≈ 300–600 s** is a plausible starting guess; final dt should be tuned to CFL constraints.

### Additional prototype: "single A100-40GB" portability
For portability to Perlmutter single-GPU tests, a safer prototype is:

- Nx = 1536
- Ny = 768
- Nz = 200
- Total cells: **235,929,600** (closer to the previously assumed "~233M cells/GPU" capacity)

### Ultra-coarse "CPU development" prototype (~1/4°)
For rapid iteration, unit testing, and data assimilation algorithm development on a CPU:

- **Resolution:** 1/4° (~28 km at equator)
- **Nx = 400** (100° longitude span)
- **Ny = 280** (70° latitude span)
- **Nz = 50** (reduced vertical levels)
- Total cells: **5,600,000**

This configuration is implemented in `experiments/atlantic_prototype.jl` and:
- Runs on a laptop CPU in reasonable time
- Uses the same physical domain as production runs
- Preserves the full forcing/initialization pipeline (JRA-55 + ECCO)
- Saves output every 10 minutes for data assimilation experiments

**Time step (baroclinic):** ~30–60 minutes (CFL-limited by TimeStepWizard)

---

## 4) Candidate simulation start dates (pre-major-hurricane cases)

The simulation should start **before cyclogenesis / rapid intensification** so that:
- the model ocean is adjusted to forcing and boundary restoring
- the Gulf Stream / mesoscale field is spun-up within the chosen domain
- the run covers the **pre-storm** and **storm** window

### High-priority candidates
#### Hurricane Irma (2017)
- Formation: **30 Aug 2017** (NHC record; widely reported)  
- Suggested start window: **~15–25 Aug 2017** (gives ~5–15 days lead-in before formation)

**Why Irma:** 2017 had extensive targeted ocean observing, including gliders sampling during Irma/Jose/Maria and analysis studies using those observations.

#### Hurricane Matthew (2016)
- Formation: **28 Sep 2016** (NHC record; widely reported)  
- Suggested start window: **~15–22 Sep 2016**

**Why Matthew:** extensive aircraft recon (AOML/HRD), plus multiple ocean-response studies; good for Gulf Stream/coastal interaction.

### Additional candidate hurricanes to evaluate (Atlantic)
The best additional candidates are those with strong **ocean observing campaigns**, e.g.:
- Underwater glider deployments (NOAA Hurricane Glider Project)
- Air-deployed ocean profilers (AXBT, ALAMO floats)
- Uncrewed surface vehicles (e.g., Saildrone hurricane missions)
- Intensively observed hurricanes in the Gulf Stream / western Atlantic

Candidate hurricanes to evaluate:
- **Hurricane Maria (2017)**
- **Hurricane Dorian (2019)**
- **Hurricane Ida (2021)**
- **Hurricane Sandy (2012)**

**Action item for agent:** perform a targeted literature search for “{hurricane name} + glider”, “AXBT”, “ALAMO float”, “Saildrone”, “ocean observing campaign”, and compile a short ranked list with DOIs and observational dataset pointers.

---

## 5) Simulation configuration checklist (for an agentic coding utility)

### Core configuration
- **Model core:** Oceananigans (GPU)
- **Grid:** lat–lon or planar Mercator approximation (choose one and be consistent with boundary restoring)
- **Domain:** Atlantic-only (bounds above), north boundary pushed to ~70°N
- **Resolution:** 1.6–2.0 km (production), coarse (prototype)
- **Vertical:** Nz=200, stretched
- **Forcing:** JRA-55 via ClimaOcean
- **Init/BC:** ECCO via ClimaOcean
- **Lateral BC:** restoring/nudging (configurable timescale τ_restore and sponge thickness)
- **Bottom drag:** quadratic
- **Mixing:** Ri-based vertical mixing + implicit vertical diffusion (as in Silvestri)

### Time stepping
- **Baroclinic dt (production):**
  - ~55 s at 1.6 km
  - ~65 s at 2 km
- **Barotropic subcycling:** ~50 substeps (Δt_barotropic ≈ 1–2 s)

### Throughput expectations (compute baseline, excluding heavy I/O)
- Production 1.6 km class: ~90 simulated days/day ⇒ ~3.7 wall-hours for 14 days
- Production 2 km class: ~162 simulated days/day ⇒ ~2.1 wall-hours for 14 days

**Note:** add margin for:
- diagnostics and checkpointing
- additional tracers
- halo sizes and work arrays
- IO bandwidth variability across machines

### Output for data assimilation

For data assimilation experiments, high-frequency output is essential:

- **Surface fields:** Every **10 minutes**
  - Sea surface temperature (SST)
  - Sea surface salinity (SSS)
  - Sea surface height (SSH / η)
  - Surface velocities (u, v)
  
- **3D snapshots:** Every **1 hour** (or as needed)
  - Full temperature and salinity profiles
  - Velocity fields
  
- **Output format:** JLD2 via `JLD2Writer` (fast, Julia-native)

This output frequency enables:
- Comparison with satellite SST/SSH products
- Assimilation of in-situ profiles (gliders, AXBT, Argo)
- Ensemble perturbation studies
- Adjoint sensitivity calculations

### Perlmutter-specific notes
- Per GPU memory: **40 GB** (A100 40GB)
- Production uses 64 GPUs (16 nodes × 4 GPUs/node)
- Prototype targets a single GPU with 80 GB memory (development workstation or equivalent); also provide a 40 GB single-GPU variant for Perlmutter debugging.

---

## 6) References and data sources (curated starter list)

### HAFS / HYCOM ocean component
- Kim, H.-S. et al. (2024): *Ocean component of the first operational version of Hurricane Analysis and Forecast System: Evaluation of HYCOM and hurricane feedback forecasts.* Frontiers in Earth Science. DOI: 10.3389/feart.2024.1399409. (Use uploaded PDF.)

### Oceananigans / GPU dycore performance
- Silvestri, S. et al. (2025): *A GPU-Based Ocean Dynamical Core for Routine Mesoscale-Resolving Climate…* JAMES. DOI: 10.1029/2024MS004465. (Use uploaded PDF.)

### Candidate-storm timing (authoritative “formed” dates)
- NHC Tropical Cyclone Report: Hurricane Irma (AL112017), “30 August–12 September 2017”. PDF: https://www.nhc.noaa.gov/data/tcr/AL112017_Irma.pdf
- NHC Tropical Cyclone Report: Hurricane Matthew (AL142016), “28 September–9 October 2016”. PDF: https://www.nhc.noaa.gov/data/tcr/AL142016_Matthew.pdf

### Ocean observing campaigns / upper-ocean observations
- NOAA AOML Hurricane Glider Project (overview + deployment map): https://www.aoml.noaa.gov/hurricane-glider-project/
- NOAA AOML article (2017 season): *Underwater Gliders Contribute to Atlantic Hurricane Season Operational Forecasts* (2018-02-01): https://www.aoml.noaa.gov/gliders-hurricane-forecasts/
- NOAA AOML article (early program): *Gliders Patrol Ocean Waters with a Goal of Improved Prediction of Hurricane Intensity*: https://www.aoml.noaa.gov/glider-patrol-hurricanes/
- Luettich, R. et al. (2018): *Transient Response of the Gulf Stream to Multiple Hurricanes in 2017.* Geophysical Research Letters. (NOAA repository landing page): https://repository.library.noaa.gov/view/noaa/54884

(Agent should expand this into a ranked storm list with observation types: gliders, AXBT/ALAMO floats, Saildrone, etc., and include DOIs + dataset access points where possible.)
