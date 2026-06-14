# Architecture

`soliton_solver` is built on a modular, object-oriented architecture that separates the theory-agnostic numerical infrastructure from theory-specific physics implementations.

## Design Philosophy

The core principle is **separation of concerns**:

- **Numerical core** — Generic PDE solver, finite-difference operators, time integrators, and CUDA utilities
- **Physics theories** — Theory-specific parameter sets, energy functionals, initial conditions, observables, and visualization

This separation enables:
- Rapid theory development without touching the solver
- Reuse of optimized CUDA kernels across different models
- Easy testing and validation of numerical methods
- Clean, maintainable codebase

## Plug-and-Play Theory Registry

Theories are discovered and loaded dynamically via the **theory registry** (`soliton_solver.theories.registry`).

### Registration mechanism

Each theory package provides a `THEORY_SPEC` descriptor:

```python
from soliton_solver.theories.registry import TheorySpec

THEORY_SPEC = TheorySpec(
    name="Chiral magnet",
    aliases=("Chiral ferromagnet", "DM magnet"),
    import_path="soliton_solver.theories.chiral_magnet",
    description="Chiral ferromagnet with Dzyaloshinskii-Moriya interaction",
    version="1.0"
)
```

The registry maintains a mapping of theory names to their specifications:

```python
from soliton_solver.theories import list_theories, load_theory

# List all registered theories
theories = list_theories()

# Load a theory
theory = load_theory("Chiral magnet")
```

### Dynamic theory discovery

Theories are automatically registered on import. The registry validates that each theory package contains required submodules:
- `params` — Parameter definitions
- `kernels` — CUDA kernels
- `initial_config` — Initial field configurations

## Dependency Injection

`soliton_solver` uses **dependency injection** to decouple the numerical solver from physics-specific code.

### Dependency injection pattern

The `Simulation` class accepts both the parameters and the theory as constructor arguments:

```python
from soliton_solver.core.simulation import Simulation
from soliton_solver.theories import load_theory

theory = load_theory("Chiral magnet")
params = theory.params.default_params(xlen=256, ylen=256, ...)

sim = Simulation(params, theory)
```

The theory provides:

- `theory.kernels.do_gradient_step_kernel` — Energy gradient computation kernel
- `theory.kernels.do_rk4_kernel` — RK4 integrator kernel (optional)
- `theory.initial_config` — Initial field setup
- `theory.observables` — Observable calculations
- `theory.params` — Parameter management
- `theory.io` — Checkpoint/output routines
- `theory.render_gl` — Visualization backend

### Benefits

This design allows:

- **Isolation** — The solver doesn't know or care about specific physics
- **Testability** — Numerical methods can be tested against any theory
- **Extensibility** — New theories plug in without modifying the solver
- **Reusability** — Solver improvements benefit all theories immediately

## Numerical Infrastructure

The core numerical components live in `soliton_solver.core`:

### Modules

**simulation.py** — Main `Simulation` class
- Grid initialization and memory management
- Energy and gradient computation
- Integration time-stepping
- Observable evaluation and checkpoint I/O

**integrator.py** — Time-stepping algorithms
- Arrested Newton flow minimization
- Runge-Kutta 4th order (RK4) integration
- Velocity reset logic for flow arrest

**derivatives.py** — Finite-difference operators
- First and second spatial derivatives
- Laplacian operator
- Curl operator for gauge fields

**params.py** — Parameter management
- `Params` base class for user-facing parameter definition
- `ResolvedParams` for computed grid quantities (lattice spacing, etc.)
- Device parameter packing/unpacking utilities

**io.py** — Input/output utilities
- Checkpoint saving and loading
- Field data serialization
- Observable and energy history I/O

**utils.py** — GPU utilities
- CUDA kernel launch wrappers
- Grid indexing utilities
- Device memory allocation helpers

**colormaps.py** — Color mapping for visualization
- Standard colormaps
- Custom colormap definition

## Theory Model Interface

Each theory must implement these components:

### Required modules

**params.py**
- Extends `CoreParams` with theory-specific parameters
- Defines `pack_device_params()` to prepare data for CUDA kernels
- Provides `default_params()` factory method

Example:

```python
from dataclasses import dataclass
from soliton_solver.core.params import Params as CoreParams

@dataclass(frozen=True)
class Params(CoreParams):
    """Theory-specific parameters."""
    J: float = 1.0         # Exchange coupling
    K: float = 0.1         # Anisotropy
    D: float = 0.01        # DMI interaction
    # ... additional parameters
```

**kernels.py**
- Implements `do_gradient_step_kernel` — computes energy gradient at each grid point
- Optionally implements `do_rk4_kernel` — custom RK4 integrator
- May include auxiliary kernels for energy density, intermediate calculations, etc.

Gradients are typically generated via `make_do_gradient_step_kernel()` factory:

```python
from soliton_solver.core.integrator import make_do_gradient_step_kernel

def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """Compute local energy at (x, y)."""
    # Theory-specific energy functional
    ...

def compute_gradient_point(Field, d1fd1x, x, y, p_i, p_f):
    """Compute local gradient at (x, y)."""
    # Theory-specific energy gradient
    ...

do_gradient_step_kernel = make_do_gradient_step_kernel(
    compute_energy_point, 
    compute_gradient_point
)
```

**initial_config.py**
- Implements `get_initial_config()` — sets up field configurations
- Handles different ansätze (vortex types, skyrmion patterns, etc.)

Example:

```python
def get_initial_config(theory, sim, mode="ground"):
    """Initialize fields based on mode."""
    if mode == "ground":
        # Vacuum configuration
        ...
    elif mode == "vortex":
        # Single vortex
        ...
```

**observables.py**
- Computes quantities of interest: energy, field norms, topological charge, etc.
- Each observable is a function that reads field data and returns a scalar or array

Example:

```python
def compute_energy(Field, p_i, p_f):
    """Compute total energy."""
    ...

def compute_topological_charge(Field, p_i, p_f):
    """Compute winding number."""
    ...
```

**io.py**
- Saves and loads checkpoint files
- Formats output data (HDF5, NetCDF, etc.)

**render_gl.py**
- Visualization and real-time rendering
- `run_viewer()` launches the interactive window
- Colormap and field-to-texture mapping

### Optional modules

**instructions.py** — Terminal help text and theory description

**results/** — Post-processing and analysis scripts

## Module Organization

```
soliton_solver/
├── core/                    # Theory-agnostic numerical infrastructure
│   ├── simulation.py        # Main Simulation driver
│   ├── integrator.py        # Time integrators
│   ├── derivatives.py       # Finite-difference operators
│   ├── params.py            # Parameter management
│   ├── io.py                # I/O utilities
│   ├── utils.py             # GPU utilities
│   └── colormaps.py         # Color mapping
│
├── theories/                # Physics models
│   ├── registry.py          # Theory discovery and registration
│   ├── chiral_magnet/       # Example theory
│   │   ├── __init__.py      # THEORY_SPEC
│   │   ├── params.py        # Theory parameters
│   │   ├── kernels.py       # CUDA kernels
│   │   ├── initial_config.py
│   │   ├── observables.py
│   │   ├── io.py
│   │   ├── render_gl.py
│   │   └── results/         # Analysis and plotting
│   │
│   ├── baby_skyrme/         # Another theory...
│   ├── ginzburg_landau_superconductor/
│   └── ... (other theories)
│
├── visualization/           # Shared visualization backend
│   └── gl_backend.py        # OpenGL/CUDA rendering
│
├── examples/                # Runnable demonstrations
│   ├── chiral_magnet_gl.py
│   └── ... (other examples)
│
└── assets/                  # Logos, images
```

## Extensibility

To add a new theory, implement the required modules above and register it with the registry. See [Extending the Solver](extending.md) for a step-by-step guide.

## Example: Theory Loading and Simulation Setup

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

# Load theory
theory = load_theory("Chiral magnet")

# Create parameters
params = theory.params.default_params(
    xlen=256, ylen=256,
    xsize=10.0, ysize=10.0,
    J=40e-12,
    K=0.8e+6,
    D=4e-3,
)

# Inject theory into solver
sim = Simulation(params, theory)

# Initialize fields
sim.initialize({"mode": "ground"})

# Minimize energy
result = sim.minimize(tol=1e-4, max_iterations=10000)

# Compute observables
energy = sim.compute_observable("energy")
charge = sim.compute_observable("topological_charge")
```

The `Simulation` class handles all GPU memory management, CUDA kernel launches, and integration logic without knowing anything specific about the theory's physics.
