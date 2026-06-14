# API Reference

Complete API documentation for `soliton_solver`. This section documents the public interfaces for creating and running simulations.

## Quick start

### Loading a theory and running a simulation

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

# Load a theory
theory = load_theory("Chiral magnet")

# Create parameters
params = theory.params.default_params(
    xlen=256, ylen=256, xsize=10.0, ysize=10.0
)

# Create simulation
sim = Simulation(params, theory)
sim.initialize({"mode": "ground"})

# Run minimization step
energy, err = sim.step(prev_energy=0.0)
print(f"Energy: {energy}, Gradient norm: {err}")
```

### Listing available theories

```python
from soliton_solver import theories

# List all theories
theories.print_table()

# Get canonical names
available = theories.list()
print(available)

# Get theory description
theories.print_description("Baby Skyrme")
```

## soliton_solver.theories

Theory registry and loading interface.

### Functions

**`load_theory(name: str) → module`**

Load a theory by name.

```python
theory = load_theory("Ginzburg-Landau superconductor")
```

- **Parameters:**
  - `name` (str) — Canonical theory name or alias
- **Returns:** Theory module with submodules: `params`, `kernels`, `initial_config`, `observables`, `io`, `render_gl`
- **Raises:** KeyError if theory name not registered

**`list_theories() → tuple[str, ...]`**

Return canonical names of all registered theories.

```python
for theory_name in theories.list():
    print(theory_name)
```

**`list() → tuple[str, ...]`**

Convenience alias for `list_theories()`.

**`print_table() → None`**

Print a formatted table of registered theories with descriptions.

**`print_description(name: str) → None`**

Print detailed description of a theory.

```python
theories.print_description("Chiral magnet")
```

**`get_theory_spec(name: str) → TheorySpec`**

Get metadata for a theory.

```python
spec = theories.registry.get_theory_spec("Baby Skyrme")
print(spec.name, spec.version, spec.description)
```

### Classes

**`TheorySpec(name, import_path, description, version, aliases, required_submodules)`**

Immutable metadata for a theory.

- **Attributes:**
  - `name` (str) — Canonical theory name
  - `import_path` (str) — Python import path
  - `description` (str) — Short description
  - `version` (str) — Version identifier
  - `aliases` (tuple[str]) — Alternative names
  - `required_submodules` (tuple[str]) — Required submodules

## soliton_solver.core.simulation

Main simulation interface.

### Simulation class

**`Simulation(params, theory)`**

High-level simulation wrapper binding a theory to the CUDA solver.

```python
sim = Simulation(params, theory)
```

- **Parameters:**
  - `params` (Params) — Parameter set
  - `theory` (module) — Loaded theory module

#### Methods

**`initialize(init_config: dict | None = None) → None`**

Set up the grid and apply initial field configuration.

```python
sim.initialize({"mode": "ground"})  # Ground state
sim.initialize({"mode": "vortex"})  # Vortex configuration
```

- **Parameters:**
  - `init_config` (dict, optional) — Initialization options (theory-specific)

**`step(prev_energy: float) → tuple[float, float]`**

Advance simulation by one arrested Newton flow step.

```python
energy, err = sim.step(prev_energy=energy)
```

- **Parameters:**
  - `prev_energy` (float) — Energy from previous step (used for arrest logic)
- **Returns:** Tuple (new_energy, gradient_norm_max)

**`observables() → dict`**

Compute theory-specific observables.

```python
obs = sim.observables()
energy = obs["energy"]
topological_charge = obs.get("topological_charge")
```

- **Returns:** Dictionary of observable values

**`save_output(output_dir: str | None = None, precision: int = 32) → None`**

Write simulation results to disk.

```python
sim.save_output(output_dir="results", precision=32)
```

- **Parameters:**
  - `output_dir` (str, optional) — Output directory (defaults to theory results dir)
  - `precision` (int) — Number of significant figures for formatting

#### Attributes

- `Field` (CUDA device array) — Current field configuration
- `Velocity` (CUDA device array) — Current velocity field
- `grid` (CUDA device array) — Physical coordinate grid
- `p_i_h`, `p_f_h` (numpy arrays) — Integer and float parameters (host)
- `p_i_d`, `p_f_d` (CUDA device arrays) — Integer and float parameters (device)
- `theory` (module) — Theory module
- `rp` (ResolvedParams) — Resolved parameters

## soliton_solver.core.params

Parameter management.

### Classes

**`Params(xlen, ylen, halo, xsize, ysize, lsx, lsy, courant, time_step, killkinen, newtonflow, unit_magnetization)`**

User-facing parameter set.

```python
params = Params(
    xlen=256, ylen=256,
    xsize=80.0, ysize=80.0,
    time_step=0.01
)
```

- **Attributes:**
  - `xlen` (int) — Grid points along x (default: 256)
  - `ylen` (int) — Grid points along y (default: 256)
  - `halo` (int) — Halo width for stencils (default: 2)
  - `xsize` (float) — Physical domain size x (default: 80.0)
  - `ysize` (float) — Physical domain size y (default: 80.0)
  - `lsx`, `lsy` (float | None) — Lattice spacings (computed if None)
  - `courant` (float) — Courant number for adaptive time step (default: 0.5)
  - `time_step` (float | None) — Explicit time step (if None, computed from Courant)
  - `killkinen` (bool) — Reset velocity on energy increase (default: True)
  - `newtonflow` (bool) — Enable Newton flow vs. gradient descent (default: True)
  - `unit_magnetization` (bool) — Enforce unit magnetization constraint (default: False)

**`Params.with_(**kwargs) → Params`**

Create a modified copy.

```python
params2 = params.with_(xlen=512, ylen=512)
```

**`Params.resolved() → ResolvedParams`**

Derive solver-ready parameters.

```python
rp = params.resolved()
print(rp.lsx, rp.lsy, rp.time_step)
```

**`ResolvedParams`**

Computed parameters ready for the solver.

- **Attributes:**
  - `dim_grid` (int) — Total grid points
  - `dim_fields` (int) — Total field values
  - `lsx`, `lsy` (float) — Lattice spacings
  - `grid_volume` (float) — Cell area
  - `time_step` (float) — Actual time step
  - `killkinen`, `newtonflow`, `unit_magnetization` (int) — Integer flags

## soliton_solver.core.integrator

Time integration and minimization algorithms.

### Functions

**`make_do_gradient_step_kernel(do_gradient_step_point) → cuda.jit`**

Factory for creating a gradient computation kernel.

```python
def my_gradient_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    # Compute gradient at (x, y)
    ...

gradient_kernel = make_do_gradient_step_kernel(my_gradient_point)
```

- **Parameters:**
  - `do_gradient_step_point` (function) — Per-thread gradient computation
- **Returns:** CUDA kernel for computing gradients

**`make_do_rk4_kernel(compute_norm, project_orthogonal) → cuda.jit`**

Factory for creating an RK4 finalization kernel with constraint projection.

- **Parameters:**
  - `compute_norm` (function) — Field normalization (e.g., for magnetization)
  - `project_orthogonal` (function) — Constraint projection (e.g., unit norm)
- **Returns:** CUDA kernel for RK4 updates with constraints

**`do_arrested_newton_flow(...) → tuple[float, float]`**

Perform one arrested Newton flow step.

Returns: (new_energy, convergence_error)

## soliton_solver.core.derivatives

Finite-difference operators.

### Functions

**`compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f) → None`**

Compute fourth-order first spatial derivatives at (x, y).

Device function (for use inside CUDA kernels):

```python
@cuda.jit
def my_kernel(d1fd1x, Field, p_i, p_f):
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        compute_derivative_first(d1fd1x, Field, 0, x, y, p_i, p_f)
```

**`compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f) → None`**

Compute fourth-order second spatial derivatives (Laplacian + mixed) at (x, y).

## soliton_solver.core.utils

GPU utilities and helpers.

### Functions

**`idx_field(a, i, j, p_i) → int`**

Map field component and lattice site to flattened index.

Device function:

```python
idx = idx_field(0, x, y, p_i)
value = Field[idx]
```

**`idx_d1(coord, field, i, j, p_i) → int`**

Map coordinate, field component, and lattice site to first derivative index.

**`idx_d2(coord1, coord2, field, i, j, p_i) → int`**

Map two coordinates, field component, and lattice site to second derivative index.

**`launch_2d(p_i_h, threads=(8, 8)) → tuple`**

Construct a 2D CUDA launch configuration.

```python
grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
my_kernel[grid2d, block2d](...)
```

- **Returns:** Tuple (grid, block) for CUDA kernel launch

**`in_bounds(x, y, p_i) → bool`**

Device function to check if (x, y) is within the interior domain (beyond halo).

```python
@cuda.jit
def my_kernel(Field, p_i):
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        # Compute on interior points only
        ...
```

**`compute_sum(Field, gridsum_partial, dim_grid) → float`**

Reduce a field to a scalar sum.

**`compute_max(Field, max_partial, dim_grid) → float`**

Reduce a field to the maximum absolute value.

**`compute_min(Field, min_partial, dim_grid) → float`**

Reduce a field to the minimum value.

## soliton_solver.core.colormaps

Color mapping for visualization.

### Functions

**`render_jet_density_to_rgba(pbo_rgba, density, min_val, max_val, p_i) → None`**

Map a density field to jet colormap and write to RGBA buffer.

CUDA kernel:

```python
render_jet_density_to_rgba[grid2d, block2d](
    pbo_rgba, density, 0.0, 1.0, p_i_d
)
```

**`render_gray_density_to_rgba(pbo_rgba, density, min_val, max_val, p_i) → None`**

Map a density field to grayscale.

**`render_magnetization_to_rgba(pbo_rgba, magnetization, p_i) → None`**

Map a magnetization vector field (3-component) to HSV colormap.

## soliton_solver.visualization.gl_backend

OpenGL and CUDA-OpenGL interoperability.

### GLBackend class

**`GLBackend(width, height, title="CUDA-OpenGL")`**

GLFW window and OpenGL-CUDA rendering backend.

```python
backend = GLBackend(512, 512, title="My Simulation")
```

#### Methods

**`map_pbo() → cuda_array`**

Map the pixel buffer object for CUDA access.

```python
pbo_rgba = backend.map_pbo()
my_render_kernel[grid2d, block2d](pbo_rgba, ...)
cuda.synchronize()
```

- **Returns:** CUDA array view of the PBO (uint8, shape (height, width, 4))

**`unmap_pbo() → None`**

Release the PBO back to OpenGL.

**`upload_and_draw() → None`**

Copy PBO to texture and display.

**`should_close() → bool`**

Check if the user requested window close.

**`set_title(text: str) → None`**

Update the window title.

## Theory interface

Each theory module must provide:

### Required submodules

**`params`**

- `Params` class (extends core `Params`)
- `default_params(**kwargs)` function
- `pack_device_params(params)` function returning (p_i_h, p_f_h) tuples

**`kernels`**

- `do_gradient_step_kernel` — CUDA kernel for gradient computation
- `do_rk4_kernel` (optional) — CUDA kernel for RK4 with constraints

**`initial_config`**

- `initialize(Velocity, Field, ..., config)` — Set up initial fields

**`observables`**

- `compute_energy(Field, d1fd1x, ...)` — Compute total energy
- Theory-specific observable functions

**`io`**

- `save_output(output_dir, h_Field, ...)` — Write results to disk

**`render_gl`**

- `run_viewer(sim, params, **kwargs)` — Launch interactive viewer

### Optional submodules

**`instructions.py`**

- `print_instructions()` — Print keyboard controls

**`results/plotting.py`**

- Post-processing and analysis scripts

## Common patterns

### Running a minimization loop

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Chiral magnet")
params = theory.params.default_params(xlen=256, ylen=256, xsize=10.0, ysize=10.0)
sim = Simulation(params, theory)
sim.initialize({"mode": "ground"})

# Initial energy
energy = sim.observables_mod.compute_energy(
    sim.Field, sim.d1fd1x, sim.en, sim.entmp, 
    sim.gridsum_partial, sim.p_i_d, sim.p_f_d, 
    sim.p_i_h, sim.p_f_h
)

# Minimize
for step in range(10000):
    new_energy, err = sim.step(prev_energy=energy)
    energy = new_energy
    
    if step % 100 == 0:
        print(f"Step {step:5d}: E={energy:.6e}, ||∇E||={err:.6e}")
    
    if err < 1e-4:
        print(f"Converged in {step} steps")
        break

sim.save_output()
```

### Computing custom observables

```python
from numba import cuda

@cuda.jit
def my_observable_kernel(observable, Field, p_i, p_f):
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        # Theory-specific computation
        observable[idx_field(0, x, y, p_i)] = ...

# Compute
grid2d, block2d = launch_2d(sim.p_i_h, threads=(8, 8))
my_observable_kernel[grid2d, block2d](observable_buffer, sim.Field, sim.p_i_d, sim.p_f_d)
cuda.synchronize()

# Copy to host
h_observable = observable_buffer.copy_to_host()
result = np.sum(h_observable)  # Aggregate
```

### Interactive visualization with custom controls

```python
from soliton_solver.visualization.gl_backend import GLBackend

backend = GLBackend(512, 512, title="Custom Viewer")

while not backend.should_close():
    # Map PBO for rendering
    pbo_rgba = backend.map_pbo()
    
    # Render field density
    render_density_kernel[grid2d, block2d](pbo_rgba, sim.Field, ...)
    cuda.synchronize()
    
    # Unmap and display
    backend.unmap_pbo()
    backend.upload_and_draw()
    
    # Advance solver
    energy, err = sim.step(energy)
    
    # Update title with status
    backend.set_title(f"E={energy:.4e} ||∇E||={err:.4e}")

backend.terminate()
```

## Extending the solver

To add a new theory, see [Extending the Solver](extending.md).

