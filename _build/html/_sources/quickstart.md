# Quickstart

Get up and running with `soliton_solver` in minutes.

## Running a built-in example

The fastest way to see `soliton_solver` in action is to run one of the built-in examples:

```bash
python -m soliton_solver.examples.chiral_magnet_gl
```

This launches an interactive visualization of magnetic skyrmions in a chiral ferromagnet, simulated entirely on the GPU with real-time rendering.

Other available examples:

- `abelian_higgs_gl` — Abelian Higgs vortices
- `anisotropic_gl` — Anisotropic superconductor
- `anyon_gl` — Anyons in Chern-Simons theory
- `baby_skyrme_gl` — Baby Skyrme model
- `bose_einstein_condensate_gl` — Rotating BEC
- `chiral_magnet_gl` — Chiral ferromagnet skyrmions
- `liquid_crystal_gl` — Chiral liquid crystal
- `spin_triplet_gl` — Spin-triplet superconductor
- `super_ferro_gl` — Ferromagnetic superconductor

## Basic workflow

Here is a typical workflow:

### 1. Load a theory

```python
from soliton_solver.theories import load_theory

theory = load_theory("Chiral magnet")
```

### 2. Create simulation parameters

```python
params = theory.params.default_params(
    xlen=320, ylen=320,           # Grid points
    xsize=10.0, ysize=10.0,       # Physical domain size
    # Theory-specific parameters follow
    J=40e-12,                     # Exchange coupling
    K=0.8e+6,                     # Anisotropy
    D=4e-3,                       # Dzyaloshinskii-Moriya interaction
    M=580e+3,                     # Saturation magnetization
    B=0e-3,                       # Magnetic field
)
```

### 3. Initialize simulation

```python
from soliton_solver.core.simulation import Simulation

sim = Simulation(params, theory)
sim.initialize({"mode": "ground"})
```

The `initialize` method sets up initial field conditions. The `"ground"` mode initializes fields in a topological configuration suitable for soliton relaxation.

### 4. Run with visualization

```python
theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)
```

This launches an interactive OpenGL window showing the field configuration. Simulations execute entirely on the GPU; field data streams directly from CUDA memory into OpenGL buffers using zero-copy CUDA–OpenGL interop.

## Complete example: Chiral magnet skyrmions

Here is a complete runnable script:

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Chiral magnet")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, 
        xsize=10.0, ysize=10.0,
        J=40e-12, 
        K=0.8e+6, 
        D=4e-3, 
        M=580e+3, 
        B=0e-3,
        mu0=1.25663706127e-6,
        dmi_term="Heusler", 
        ansatz="anti",
        demag=True,
        newtonflow=False,
        unit_magnetization=True
    )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()
```

Run this script:

```bash
python my_simulation.py
```

## Visualizing results

After running a simulation, you can plot results:

```bash
python -m soliton_solver.theories.chiral_magnet.results.plotting
```

This generates plots of field densities, energy, and other observables.

## Next steps

- Explore the [Supported Theories](theories.md) to find your physics model
- Learn the [Numerical Solver](numerical_solver.md) for advanced configuration
- See [GPU Acceleration](gpu_acceleration.md) for performance tuning
- Understand the [Architecture](architecture.md) to customize the solver
- Create your own theory following [Extending the Solver](extending.md)

