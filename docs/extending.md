# Extending the Solver

Guide to implementing new physics theories in `soliton_solver`.

## Overview

The solver separates **theory-agnostic numerics** (core module) from **physics-specific implementations** (theories module). To add a new theory:

1. Create a new subdirectory under `soliton_solver/theories/`
2. Implement required submodules (params, kernels, initial_config, observables, io, render_gl)
3. Register the theory via `TheorySpec` metadata
4. Test with the solver and viewer

The solver handles all discretization, time integration, GPU memory management, and visualization. You only write:
- The energy functional gradients (field evolution)
- Initial field configurations
- Observable computations
- Result I/O

This document walks through a complete example: adding a **"Minimal Model"** theory with a single complex scalar field.

## Minimal skeleton

Start by creating the directory structure and stub files:

```
soliton_solver/theories/minimal_model/
├── __init__.py
├── params.py
├── kernels.py
├── initial_config.py
├── observables.py
├── io.py
├── render_gl.py
└── results/
    └── plotting.py
```

### Creating the __init__.py

Register your theory so it can be discovered:

```python
# soliton_solver/theories/minimal_model/__init__.py

from soliton_solver.theories.registry import TheorySpec, register_theory

# Import submodules (loaded on-demand)
from . import params, kernels, initial_config, observables, io, render_gl

# Define theory metadata
_spec = TheorySpec(
    name="Minimal model",
    import_path="soliton_solver.theories.minimal_model",
    description="Single complex scalar field with U(1) symmetry",
    version="1.0.0",
    aliases=("minimal", "scalar"),
    required_submodules=(
        "params", "kernels", "initial_config", 
        "observables", "io", "render_gl"
    )
)

# Register globally
register_theory(_spec)
```

## Step-by-step walkthrough

### Step 1: Define parameters (params.py)

Theory parameters are derived from the base `Params` class. Define your energy functional parameters:

```python
# soliton_solver/theories/minimal_model/params.py

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from soliton_solver.core.params import Params

@dataclass
class Params(Params):
    """Minimal model parameters.
    
    Energy functional:
        E = ∫ dx dy [ |∇φ|² + V(|φ|) ]
    
    where V(|φ|) = λ(1 - |φ|²)² is the Higgs potential.
    """
    
    # Theory-specific parameters
    lam: float = 1.0  # Coupling strength of potential
    phi_0: float = 1.0  # Vacuum expectation value
    
    def pack_device_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pack parameters for GPU kernels.
        
        Returns:
            (p_i_h, p_f_h) — Integer and float parameter arrays for GPU
        """
        # Integers: [xlen, ylen, dim_grid, halo, killkinen, newtonflow, unit_magnetization]
        p_i_h = np.array([
            self.xlen, self.ylen,
            self.xlen * self.ylen,
            self.halo,
            int(self.killkinen),
            int(self.newtonflow),
            int(self.unit_magnetization)
        ], dtype=np.int32)
        
        # Floats: [lsx, lsy, courant, time_step, lam, phi_0]
        p_f_h = np.array([
            self.lsx or (self.xsize / self.xlen),
            self.lsy or (self.ysize / self.ylen),
            self.courant,
            self.time_step or (self.courant * (self.lsx or self.xsize / self.xlen) ** 2),
            self.lam,
            self.phi_0
        ], dtype=np.float32)
        
        return p_i_h, p_f_h


def default_params(**kwargs) -> Params:
    """Create default parameters for Minimal model.
    
    Example:
        params = default_params(xlen=512, ylen=512)
    """
    defaults = {
        "xlen": 256,
        "ylen": 256,
        "xsize": 10.0,
        "ysize": 10.0,
        "lam": 1.0,
        "phi_0": 1.0,
    }
    defaults.update(kwargs)
    return Params(**defaults)
```

**Key points:**
- Extend base `Params` with theory-specific fields
- Implement `pack_device_params()` to convert Python parameters to GPU arrays
- First 7 integers are mandatory (grid, halo, flags)
- Float array starts with (lsx, lsy, courant, time_step), then your parameters
- Provide `default_params()` factory function

### Step 2: Implement CUDA kernels (kernels.py)

The gradient kernel computes ∂E/∂φ at each lattice point:

```python
# soliton_solver/theories/minimal_model/kernels.py

import numpy as np
from numba import cuda, types
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second
from soliton_solver.core.utils import (
    idx_field, idx_d1, idx_d2, launch_2d, in_bounds
)
from soliton_solver.core.integrator import (
    make_do_gradient_step_kernel, make_do_rk4_kernel
)


@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient,
                           d1fd1x, d2fd2x, x, y, p_i, p_f):
    """Compute ∇E at point (x, y).
    
    Energy: E = ∫ [ |∇φ|² + λ(1 - |φ|²)² ]
    Gradient: ∂E/∂φ* = -∇²φ + 2λ(1 - |φ|²)φ
    """
    halo = p_i[3]
    lsx = p_f[0]
    lsy = p_f[1]
    lam = p_f[4]
    
    # Get field value: φ(x, y)
    idx_phi = idx_field(0, x, y, p_i)
    phi_x, phi_y = Field[idx_phi], Field[idx_phi + 1]  # Real, imaginary parts
    
    # Compute |φ|²
    phi_sq = phi_x * phi_x + phi_y * phi_y
    
    # Compute ∇φ and ∇²φ (Laplacian)
    compute_derivative_first(d1fd1x, Field, 0, x, y, p_i, p_f)
    compute_derivative_second(d2fd2x, Field, 0, x, y, p_i, p_f)
    
    # ∂E/∂φ* = -∇²φ + 2λ(1 - |φ|²)φ
    # Real part: Re(∂E/∂φ*_re) = -∇²φ_re + 2λ(1 - φ_sq)φ_re
    d2phi_re = d2fd2x[idx_d2(0, 0, 0, x, y, p_i)]
    d2phi_im = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    laplacian_re = (d2phi_re + d2phi_im)
    
    grad_re = -laplacian_re + 2.0 * lam * (1.0 - phi_sq) * phi_x
    grad_im = -laplacian_re + 2.0 * lam * (1.0 - phi_sq) * phi_y
    
    # Store gradient
    EnergyGradient[idx_phi] = grad_re
    EnergyGradient[idx_phi + 1] = grad_im


# Create the actual gradient kernel
do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)
```

**Key points:**
- Device function receives field values, derivatives, and parameters
- Compute energy gradients using finite differences already computed
- Return via `EnergyGradient` array (passed by reference)
- Use `idx_field()`, `idx_d1()`, `idx_d2()` for array indexing
- Extract parameters from `p_i` (integers) and `p_f` (floats) arrays

**Optional: Custom RK4 with constraints**

If your theory has constraints (e.g., unit magnetization), implement custom RK4:

```python
@cuda.jit(device=True)
def project_orthogonal(Field, x, y, p_i, p_f):
    """Enforce unit norm constraint: |φ| = φ_0."""
    idx = idx_field(0, x, y, p_i)
    phi_0 = p_f[5]
    
    phi_x, phi_y = Field[idx], Field[idx + 1]
    phi_norm = (phi_x * phi_x + phi_y * phi_y) ** 0.5
    
    if phi_norm > 0.0:
        Field[idx] = phi_x * phi_0 / phi_norm
        Field[idx + 1] = phi_y * phi_0 / phi_norm


@cuda.jit(device=True)
def compute_norm(Field, x, y, p_i, p_f):
    """Return the field norm for normalization."""
    idx = idx_field(0, x, y, p_i)
    phi_x, phi_y = Field[idx], Field[idx + 1]
    return (phi_x * phi_x + phi_y * phi_y) ** 0.5


do_rk4_kernel = make_do_rk4_kernel(compute_norm, project_orthogonal)
```

### Step 3: Initial configurations (initial_config.py)

Define how to initialize fields for different scenarios:

```python
# soliton_solver/theories/minimal_model/initial_config.py

import numpy as np
from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds, launch_2d


@cuda.jit
def initialize_ground_state(Field, grid, p_i, p_f):
    """Uniform ground state: φ = φ₀."""
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        phi_0 = p_f[5]
        idx = idx_field(0, x, y, p_i)
        Field[idx] = phi_0  # Real part
        Field[idx + 1] = 0.0  # Imaginary part


@cuda.jit
def initialize_vortex(Field, grid, p_i, p_f):
    """Vortex: φ = φ₀ exp(i*θ) where θ is angle from center."""
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        phi_0 = p_f[5]
        xlen, ylen = p_i[0], p_i[1]
        xsize, ysize = p_f[0] * xlen, p_f[1] * ylen
        
        # Coordinates relative to center
        xc = grid[idx_field(0, x, y, p_i)] - xsize / 2.0
        yc = grid[idx_field(1, x, y, p_i)] - ysize / 2.0
        
        # Angle
        theta = np.arctan2(yc, xc)
        
        # φ = φ₀ exp(i*θ)
        idx = idx_field(0, x, y, p_i)
        Field[idx] = phi_0 * np.cos(theta)
        Field[idx + 1] = phi_0 * np.sin(theta)


def initialize(Velocity, Field, grid, p_i_h, p_f_h, p_i_d, p_f_d, config):
    """Initialize fields according to config dict.
    
    Parameters:
        config (dict) — Options: mode in ["ground", "vortex", "random"]
    """
    mode = config.get("mode", "ground")
    
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    
    if mode == "ground":
        initialize_ground_state[grid2d, block2d](Field, grid, p_i_d, p_f_d)
    elif mode == "vortex":
        initialize_vortex[grid2d, block2d](Field, grid, p_i_d, p_f_d)
    elif mode == "random":
        # Random initialization from host
        Field[:] = np.random.randn(*Field.shape).astype(np.float32) * 0.1
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Zero initial velocity
    Velocity.fill(0)
```

**Key points:**
- Implement different initialization modes (ground state, defects, random)
- Use `launch_2d()` to get proper CUDA grid/block dims
- Check `in_bounds()` to skip halo regions
- Zero the `Velocity` array initially

### Step 4: Observables (observables.py)

Define quantities to compute during simulation:

```python
# soliton_solver/theories/minimal_model/observables.py

import numpy as np
from numba import cuda
from soliton_solver.core.utils import (
    idx_field, idx_d1, launch_2d, compute_sum, compute_max, in_bounds
)


@cuda.jit
def do_compute_energy_partial(energy_partial, Field, d1fd1x, d2fd2x,
                              p_i, p_f):
    """Compute energy density: E = |∇φ|² + λ(1 - |φ|²)²."""
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        idx = idx_field(0, x, y, p_i)
        phi_x, phi_y = Field[idx], Field[idx + 1]
        
        # Kinetic energy: |∇φ|²
        d1phi_x_re = d1fd1x[idx_d1(0, 0, 0, x, y, p_i)]
        d1phi_x_im = d1fd1x[idx_d1(0, 0, 1, x, y, p_i)]
        d1phi_y_re = d1fd1x[idx_d1(1, 0, 0, x, y, p_i)]
        d1phi_y_im = d1fd1x[idx_d1(1, 0, 1, x, y, p_i)]
        
        kinetic = (d1phi_x_re**2 + d1phi_x_im**2 + 
                   d1phi_y_re**2 + d1phi_y_im**2)
        
        # Potential energy: λ(1 - |φ|²)²
        phi_sq = phi_x**2 + phi_y**2
        lam = p_f[4]
        potential = lam * (1.0 - phi_sq) ** 2
        
        energy_partial[idx_field(0, x, y, p_i)] = kinetic + potential


def compute_energy(Field, d1fd1x, d2fd2x, energy_partial,
                   gridsum_partial, p_i_h, p_f_h, p_i_d, p_f_d):
    """Compute total energy."""
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    do_compute_energy_partial[grid2d, block2d](
        energy_partial, Field, d1fd1x, d2fd2x, p_i_d, p_f_d
    )
    return compute_sum(energy_partial, gridsum_partial, p_i_h[2])


@cuda.jit
def do_compute_field_norm_partial(norm_partial, Field, p_i, p_f):
    """Compute |φ|."""
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        idx = idx_field(0, x, y, p_i)
        phi_x, phi_y = Field[idx], Field[idx + 1]
        norm_partial[idx_field(0, x, y, p_i)] = (phi_x**2 + phi_y**2) ** 0.5


def compute_field_norm(Field, norm_partial, gridsum_partial,
                       p_i_h, p_f_h, p_i_d, p_f_d):
    """Compute mean field norm."""
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    do_compute_field_norm_partial[grid2d, block2d](norm_partial, Field, p_i_d, p_f_d)
    total = compute_sum(norm_partial, gridsum_partial, p_i_h[2])
    return total / p_i_h[2]
```

### Step 5: Output and I/O (io.py)

Save results to disk:

```python
# soliton_solver/theories/minimal_model/io.py

import numpy as np
import os


def save_output(output_dir, h_Field, p_i_h, p_f_h, precision=32):
    """Save field configuration to disk.
    
    Parameters:
        output_dir (str) — Directory for output files
        h_Field (ndarray) — Field on host (CPU)
        p_i_h, p_f_h (ndarray) — Parameters
        precision (int) — Single (32) or double (64) precision
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dtype = np.float32 if precision == 32 else np.float64
    
    # Save field
    field_file = os.path.join(output_dir, "field.npy")
    np.save(field_file, h_Field.astype(dtype))
    
    # Save parameters
    params_file = os.path.join(output_dir, "params.npz")
    np.savez(params_file,
             p_i_h=p_i_h,
             p_f_h=p_f_h)
    
    print(f"Saved field to {field_file}")
    print(f"Saved parameters to {params_file}")


def load_output(output_dir):
    """Load previously saved field and parameters."""
    field_file = os.path.join(output_dir, "field.npy")
    params_file = os.path.join(output_dir, "params.npz")
    
    h_Field = np.load(field_file)
    params_data = np.load(params_file)
    
    return h_Field, params_data["p_i_h"], params_data["p_f_h"]
```

### Step 6: Visualization (render_gl.py)

Interactive viewer and rendering:

```python
# soliton_solver/theories/minimal_model/render_gl.py

import numpy as np
from numba import cuda
from soliton_solver.core.utils import idx_field, launch_2d, in_bounds
from soliton_solver.core.colormaps import render_jet_density_to_rgba
from soliton_solver.visualization.gl_backend import GLBackend


@cuda.jit
def render_field_magnitude(pbo_rgba, Field, p_i, p_f):
    """Render |φ| to screen."""
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        idx = idx_field(0, x, y, p_i)
        phi_x, phi_y = Field[idx], Field[idx + 1]
        magnitude = (phi_x**2 + phi_y**2) ** 0.5
        render_jet_density_to_rgba(pbo_rgba, magnitude, 0.0, 2.0, p_i)


def run_viewer(sim, params, **kwargs):
    """Launch interactive viewer.
    
    Controls:
        n — Toggle Newton flow
        k — Toggle arrest/kill kinetic energy
        Esc — Exit
    """
    backend = GLBackend(params.xlen, params.ylen, 
                       title=f"Minimal model - {params.xlen}×{params.ylen}")
    
    grid2d, block2d = launch_2d(sim.p_i_h, threads=(8, 8))
    
    step_count = 0
    energy = 0.0
    
    try:
        while not backend.should_close():
            # Render
            pbo_rgba = backend.map_pbo()
            render_field_magnitude[grid2d, block2d](
                pbo_rgba, sim.Field, sim.p_i_d, sim.p_f_d
            )
            cuda.synchronize()
            backend.unmap_pbo()
            backend.upload_and_draw()
            
            # Advance
            energy, err = sim.step(prev_energy=energy)
            step_count += 1
            
            # Update title
            backend.set_title(
                f"Minimal model | Step {step_count} | "
                f"E={energy:.4e} | ||∇E||={err:.4e}"
            )
    
    finally:
        backend.terminate()
```

**Key points:**
- Create rendering kernels that write to the PBO (pixel buffer object)
- Use `backend.map_pbo()` / `unmap_pbo()` for zero-copy rendering
- Call `backend.upload_and_draw()` to display the frame
- Implement interactive controls by checking keyboard input
- Always wrap in try/finally to ensure cleanup

### Step 7: Optional — Instructions and Results

Keyboard controls:

```python
# soliton_solver/theories/minimal_model/instructions.py

def print_instructions():
    """Print keyboard and mouse controls."""
    print("""
    ╔══════════════════════════════════════════╗
    ║         Minimal Model Controls           ║
    ╚══════════════════════════════════════════╝
    
    Keyboard:
      n  — Toggle Newton flow (minimize energy)
      k  — Toggle kinetic energy arrest (reset velocity on energy increase)
      Esc — Exit viewer
    
    The viewer renders |φ|, the magnitude of the complex field.
    Color indicates magnitude (blue = 0, red = 2).
    """)
```

Post-processing:

```python
# soliton_solver/theories/minimal_model/results/plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_field(h_Field, p_i_h, output_dir="figures"):
    """Plot field configuration."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    xlen, ylen = p_i_h[0], p_i_h[1]
    
    # Field magnitude
    field_2d = h_Field[:xlen*ylen].reshape(xlen, ylen)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(field_2d, cmap="viridis", origin="lower")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("|φ|")
    plt.colorbar(im, ax=ax)
    
    output_file = os.path.join(output_dir, "field.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved figure to {output_file}")
```

## Best practices

### CUDA kernel performance

1. **Maximize thread occupancy** — Use 256-512 threads per block (8×8 or 16×16 grid)
2. **Minimize global memory traffic** — Reuse computed values, avoid redundant loads
3. **Watch for warp divergence** — Keep conditionals outside innermost loops
4. **Use shared memory for stencil operations** — Load halo regions efficiently (requires careful indexing)

```python
# Good: Minimize memory operations
@cuda.jit
def efficient_kernel(Field, gradient, p_i, p_f):
    x, y = cuda.grid(2)
    if in_bounds(x, y, p_i):
        idx = idx_field(0, x, y, p_i)
        
        # Load once, reuse multiple times
        phi_x = Field[idx]
        phi_y = Field[idx + 1]
        phi_sq = phi_x * phi_x + phi_y * phi_y
        
        # Compute with cached values
        term1 = phi_sq ** 2
        term2 = 2.0 * phi_sq
```

### Memory management

1. **Pre-allocate buffers** — Reuse temporary arrays between steps
2. **Use float32 by default** — Double precision is rarely needed for PDEs
3. **Transfer data minimally** — Keep working arrays on GPU, save at intervals

### Code organization

1. **One theory per subdirectory** — Makes discovery/registration automatic
2. **Keep kernels short** — Device functions should be pure, stateless
3. **Document kernel interfaces** — Comment the meaning of parameter indices (e.g., p_f[4] = coupling)

## Testing your theory

1. **Test energy minimization** — Run ground state search, verify energy decreases monotonically

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Minimal model")
params = theory.params.default_params(xlen=128, ylen=128)
sim = Simulation(params, theory)
sim.initialize({"mode": "ground"})

energy = sim.observables()["energy"]
for step in range(100):
    energy_new, err = sim.step(prev_energy=energy)
    assert energy_new <= energy or energy_new <= energy + 1e-3  # Allow small numerical error
    energy = energy_new
    if err < 1e-4:
        print(f"Converged in {step} steps")
        break
```

2. **Verify observables** — Compare against known analytic values (if available)
3. **Check visualization** — Run viewer, look for reasonable field evolution
4. **Profile performance** — Use `nvidia-smi` or `nsys` to check GPU utilization (should be >80%)

## Registering your theory

Once your theory is complete, add it to the registry. This happens automatically if:

1. Your `__init__.py` calls `register_theory(_spec)` at module load
2. The theory is importable from `soliton_solver.theories`

Test registration:

```python
from soliton_solver import theories

theories.print_table()  # Should list your theory
theory = theories.load_theory("Minimal model")
print(theory)
```

## Contributing back

To contribute your theory to soliton_solver:

1. **Follow the structure** — Subdirectory with all required modules, clear code style
2. **Add comprehensive docstrings** — Document energy functional, parameters, field meanings
3. **Include example usage** — Add a file to `soliton_solver/examples/` showing how to use your theory
4. **Provide references** — Cite papers or resources for the energy functional in docstrings
5. **Test thoroughly** — Verify energy minimization, field stability, visualization correctness
6. **Submit a pull request** — Include: theory subdirectory, example file, addition to README theory table

Example contribution structure:

```
soliton_solver/
├── theories/
│   └── my_new_theory/
│       ├── __init__.py
│       ├── params.py
│       ├── kernels.py
│       ├── initial_config.py
│       ├── observables.py
│       ├── io.py
│       └── render_gl.py
└── examples/
    └── my_new_theory_gl.py  # Run viewer example
```

### Example contribution file

```python
# soliton_solver/examples/my_new_theory_gl.py
"""Interactive viewer for My New Theory.

Usage:
    python examples/my_new_theory_gl.py
"""

from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

if __name__ == "__main__":
    # Load and run with default parameters
    theory = load_theory("My new theory")
    params = theory.params.default_params(xlen=512, ylen=512)
    
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    
    # Launch interactive viewer
    theory.render_gl.run_viewer(sim, params)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `KeyError: 'My theory'` | Check `__init__.py` calls `register_theory()` before function returns |
| Kernel launch errors | Verify CUDA code syntax; use `cuda.synchronize()` then check output |
| Field goes NaN | Check potential energy terms; may need Courant number reduction |
| Slow performance | Profile with `nvidia-smi` or `nsys`; check kernel occupancy; consider shared memory |
| Visualization shows noise | Check energy gradients are computed correctly; verify normalization |