# DataAssimilocean.jl rules for agent-assisted development

## Project Overview

DataAssimilocean.jl is a Julia project for experiments in ensemble-based and variational data assimilation with ocean models.
The project uses Oceananigans.jl for ocean simulation and ClimaOcean.jl for realistic forcing, initial conditions, and boundary conditions.
The primary use case is regional ocean state estimation and forward simulation, with an emphasis on hurricane-ocean interaction studies.

## Language & Environment

- **Language**: Julia 1.10+
- **Architectures**: CPU and GPU (CUDA)
- **Key Packages**: Oceananigans.jl, ClimaOcean.jl, Enzyme.jl (for AD-based data assimilation)
- **Data Sources**: ECCO (state estimates), JRA-55 (atmospheric reanalysis)

## Code Style & Conventions

### Julia Best Practices

1. **Explicit Imports in source code**: Use `ExplicitImports.jl` style for source files in `src/`
   - Import from modules explicitly
   
2. **User scripts and experiments**: Follow the ClimaOcean example style:
   - Use `using Oceananigans`, `using ClimaOcean` for the user interface
   - Only explicitly import names that are not exported
   - Minimize explicit imports; rely on the exported user interface
   - Use `using Oceananigans.Units` for time/length units
   
3. **Type Stability**: Prioritize type-stable code for performance
   - All structs must be concretely typed
   
4. **Kernel Functions**: For GPU compatibility:
   - Use KernelAbstractions.jl syntax for kernels (`@kernel`, `@index`)
   - Keep kernels type-stable and allocation-free
   - Use `ifelse` instead of short-circuiting `if`/`else` or ternary operators `?` ... `:`
   - No error messages inside kernels
   - Mark functions inside kernels with `@inline`
   - Models _never_ go inside kernels

5. **Memory efficiency**
   - Favor inline computations over temporary memory allocation
   - Design solutions that work within the Oceananigans framework
   - Minimize memory allocation in hot paths

6. **Debugging tips**
   - Sometimes "julia version compatibility" issues are resolved by deleting `Manifest.toml` and running `Pkg.instantiate()`
   - GPU tests may fail with "dynamic invocation error" - test on CPU first to isolate type-inference issues

### General Coding Style

- Variables may use "symbolic form" (unicode, useful in equations) or "English form" (descriptive). Use consistently within expressions.
- Keyword argument formatting:
  * Inline: `f(x=1, y=2)`
  * Multiline:
    ```julia
    long_function(a = 1,
                  b = 2)
    ```
- Use `const` only when necessary, not by default
- Number variables (`Nx`, `Ny`, `Nz`) start with capital `N`. Use `Nt` for time steps.
- Spatial indices are `i, j, k`; time index is `n`

### Naming Conventions

- **Files**: snake_case (e.g., `atlantic_prototype.jl`)
- **Types**: PascalCase (e.g., `HydrostaticFreeSurfaceModel`)
- **Functions**: snake_case (e.g., `ocean_simulation`, `regrid_bathymetry`)
- **Kernels**: May be prefixed with underscore (e.g., `_compute_tendency_kernel`)
- **Variables**: English long names or mathematical notation with unicode
- **Avoid abbreviations**: Use `latitude` not `lat`, `temperature` not `temp`

### Project Structure

```
DataAssimilocean.jl/
├── AGENTS.md              # This file
├── Project.toml           # Package dependencies
├── README.md              # Top-level project overview
├── project/
│   └── README.md          # Detailed simulation strategy document
├── experiments/           # Simulation scripts
│   └── atlantic_prototype.jl  # Atlantic-basin prototype simulation
├── src/
│   └── DataAssimilocean.jl    # Main module
└── test/                  # Tests (when added)
```

## Simulation Guidelines

### Grid Configuration

- Use `LatitudeLongitudeGrid` for regional Atlantic simulations
- Use `ImmersedBoundaryGrid` with `GridFittedBottom` for bathymetry
- Vertical grid: stretched with high resolution near surface (use `ExponentialDiscretization`)
- Always specify `halo = (7, 7, 7)` for WENO schemes

### Forcing and Initialization (via ClimaOcean)

- **Atmospheric forcing**: `JRA55PrescribedAtmosphere`
- **Ocean initial conditions**: 
  - EN4 via `Metadatum` with `EN4Monthly()` (no credentials required)
  - ECCO via `Metadatum` with `ECCO4Monthly()` (requires NASA Earthdata credentials)
- **Lateral restoring**: `ECCO_restoring_forcing` for boundary nudging
- **Bathymetry**: `regrid_bathymetry` from ClimaOcean
- **Radiation**: `Radiation(arch)` for shortwave/longwave fluxes

### Output for Data Assimilation

- Save fields at high temporal frequency (e.g., every 10 minutes) for assimilation experiments
- Use `JLD2Writer` with `TimeInterval` scheduling
- Save both surface fields (2D slices) and 3D snapshots as needed
- For surface fields, use `indices = (:, :, grid.Nz)`

### Example Script Patterns

Following ClimaOcean examples:

```julia
using Oceananigans
using Oceananigans.Units
using ClimaOcean
using Dates
using Printf

arch = CPU()  # or GPU() for production

# Grid setup
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4)
grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), latitude, longitude, z, halo=(7, 7, 7))

# Bathymetry
bottom_height = regrid_bathymetry(grid; minimum_depth=10, interpolation_passes=5)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Ocean simulation
ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface)

# Initial conditions from ECCO
set!(ocean.model, T = Metadatum(:temperature; date, dataset=ECCO4Monthly()),
                  S = Metadatum(:salinity;    date, dataset=ECCO4Monthly()))

# Atmospheric forcing
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(41))
radiation = Radiation(arch)

# Coupled model and simulation
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time)

# TimeStepWizard for adaptive time stepping
wizard = TimeStepWizard(; cfl=0.2, max_Δt=1hour, max_change=1.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Progress callback
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Output writers
ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                            schedule = TimeInterval(10minutes),
                                            filename = "surface_fields",
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

run!(simulation)
```

### Script Writing Guidelines

- Explain at the top of the file what the simulation does
- Let code "speak for itself" - keep explanations concise (Literate style)
- Call `set!` ideally once to set the entire model state
- Instances of ocean simulations are usually called `ocean`
- Instances of `Simulation` are called `simulation`
- Use integers for integer values; don't eagerly convert to Float64 with `.0`
- Use exported names from `Oceananigans` and `ClimaOcean` whenever possible

## Data Assimilation Focus

### Output Requirements

- High-frequency output (10-minute intervals) for assimilation experiments
- Save all prognostic variables (T, S, u, v, w, η)
- Consider saving at observation locations for comparison with in-situ data

### Future Directions

- Enzyme.jl integration for adjoint-based 4D-Var
- Ensemble methods (EnKF, ensemble smoother)
- Observation operators for glider/AXBT/Saildrone data

## Testing Guidelines

### Running Experiments

```julia
# For quick tests, reduce resolution and iterations:
# - Use coarse grid (1/4 degree or coarser)
# - Use CPU() architecture
# - Set simulation.stop_iteration = 100

# Example test run:
simulation.stop_iteration = 100
run!(simulation)
```

### Debugging

- GPU tests may fail with "dynamic invocation error" - test on CPU first
- Check type stability for GPU-bound code
- If a test fails after a change, revisit whether the change was correct

### Debugging terminal output from Julia

Julia error messages can be overwhelmed by extremely long type signatures, making errors unreadable. Use these techniques:

1. **Filter for actual errors:**
   ```bash
   julia script.jl 2>&1 | grep -E "^ERROR|error:|Exception"
   ```

2. **Use `--color=no` to reduce ANSI clutter:**
   ```bash
   julia --color=no script.jl 2>&1 | tail -50
   ```

3. **Wrap simulation runs in try/catch for cleaner errors:**
```julia
   try
       run!(simulation)
   catch e
       @error "Simulation failed" exception=(e, catch_backtrace())
       rethrow()
   end
   ```

4. **For ClimaOcean simulations, common runtime errors include:**
   - `NaN` or `Inf` values from numerical instability (reduce time step)
   - Missing JRA55 data files (check download completed)
   - ECCO credential issues (use EN4 instead, no credentials needed)
   - Memory issues with large grids on CPU

5. **Check specific error types:**
   ```bash
   julia script.jl 2>&1 | grep -E "MethodError|InexactError|DomainError|TypeError"
   ```

6. **Truncate long type signatures in error output:**
   Julia error messages often contain massive type signatures. Use this pattern to get readable errors:
   ```julia
   try
       run!(simulation)
   catch e
       println("=== ERROR TYPE ===")
       println(typeof(e))
       println("=== ERROR MESSAGE ===")
       msg = sprint(showerror, e)
       for line in split(msg, "\n")[1:min(5, end)]
           println(length(line) > 200 ? line[1:200] * "..." : line)
       end
   end
   ```

## Common Pitfalls

1. **Type Instability**: Especially in kernel functions - ruins GPU performance
2. **Overconstraining types**: Julia compiler can infer types. Use type annotations for _multiple dispatch_, not documentation.
3. **Forgetting Explicit Imports**: In source code, explicitly import all used functions
4. **Using `interior()` for plotting**: Use `Field` objects directly with Makie; use `view(field, :, :, k)` for slices
5. **ECCO credentials**: ECCO data requires `ECCO_USERNAME` and `ECCO_WEBDAV_PASSWORD` environment variables. Use EN4 instead if you don't have credentials (see https://github.com/CliMA/ClimaOcean.jl/blob/main/src/DataWrangling/ECCO/README.md)
6. **TimeStepWizard with OceanSeaIceModel**: `TimeStepWizard` doesn't work with `OceanSeaIceModel` (missing `cell_advection_timescale` method). Use a fixed time step for coupled models.

## References

### Key Papers

- Kim et al. (2024): HAFS ocean component (domain reference)
- Silvestri et al. (2025): GPU ocean dycore performance

### Datasets

- ECCO: https://www.ecco-group.org/
- JRA-55: https://jra.kishou.go.jp/JRA-55/

### Documentation

- Oceananigans: https://clima.github.io/OceananigansDocumentation/stable/
- ClimaOcean: https://clima.github.io/ClimaOceanDocumentation/dev/
- KernelAbstractions.jl: https://github.com/JuliaGPU/KernelAbstractions.jl
- Enzyme.jl: https://enzyme.mit.edu/julia/dev

## AI Assistant Behavior

- Follow ClimaOcean example patterns for experiment scripts
- Prioritize type stability and GPU compatibility
- Keep simulations configurable (architecture, resolution, time step)
- Include progress callbacks for monitoring
- Add output writers for data assimilation needs
- Test scripts on CPU with reduced resolution before GPU runs
- Avoid over-engineering; make minimal changes to achieve the goal
- When modifying code, prefer editing existing files over creating new ones
- Check existing examples in ClimaOcean for usage patterns
- Reference physics equations in comments when implementing dynamics

## When Unsure

1. Check existing examples in ClimaOcean's `examples/` directory
2. Look at similar implementations in Oceananigans.jl
3. Review the project strategy in `project/README.md`
4. Check documentation links above
5. Ask in GitHub discussions for Oceananigans or ClimaOcean
