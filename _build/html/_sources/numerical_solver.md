# Numerical Solver

The `soliton_solver` implements a theory-agnostic numerical solver for 2D PDEs describing topological solitons. This document describes the finite-difference method, the arrested Newton flow minimization algorithm, and solver configuration.

## Overview

The numerical core solves the relaxation equation:

$$
\ddot{\phi}(t) = -\nabla_\phi E_h[\phi]
$$

where $E_h[\phi]$ is a discretized energy functional and $\phi$ is the field configuration. The goal is to find local minima of $E_h$.

Key features:
- **Second-order relaxation** — Velocity helps escape shallow regions of the energy landscape
- **Flow arrest** — Velocity resets if energy increases, preventing oscillations
- **Theory-agnostic** — Core solver doesn't know specific physics
- **GPU-accelerated** — All kernels run on NVIDIA GPUs via Numba CUDA

## Finite Difference Method

Fields are discretized on a 2D Cartesian grid with uniform spacing. Spatial derivatives are computed via fourth-order central finite differences.

### Grid setup

The computational domain is discretized into $n_x \times n_y$ lattice points:

- **Domain**: $[0, L_x] \times [0, L_y]$ in physical space
- **Grid points**: $n_x \times n_y$ lattice sites
- **Spacing**: $\Delta x = L_x / n_x$, $\Delta y = L_y / n_y$
- **Halo**: Additional boundary points for stencil computations

Parameters:

```python
params = theory.params.default_params(
    xlen=256,        # Number of lattice points x
    ylen=256,        # Number of lattice points y
    xsize=80.0,      # Physical domain size x
    ysize=80.0,      # Physical domain size y
    halo=2,          # Halo width (default: 2)
)
rp = params.resolved()
print(f"Lattice spacing: dx={rp.lsx}, dy={rp.lsy}")
print(f"Grid volume: {rp.grid_volume}")
```

**Halo width** — Boundary region used by finite-difference stencils. For fourth-order stencils with ±2 offset, `halo=2` is standard.

### Stencils

#### First derivatives

Fourth-order central difference for $\partial_x f$:

$$
\frac{\partial f}{\partial x}\bigg|_{i,j} \approx \frac{1}{12\Delta x}(f_{i-2,j} - 8f_{i-1,j} + 8f_{i+1,j} - f_{i+2,j})
$$

Implementation (from `derivatives.py`):

```python
@cuda.jit(device=True)
def compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f):
    """Compute first derivatives at (x, y) for field component a."""
    xlen = p_i[0]
    ylen = p_i[1]
    halo = p_i[2]
    lsx = p_f[2]  # Lattice spacing
    
    # Check interior (not in halo)
    if x > halo - 1 and x < xlen - halo:
        d1fd1x[idx_d1(0, a, x, y, p_i)] = (
            (1.0/12.0) * Field[idx_field(a, x-2, y, p_i)]
            - (2.0/3.0) * Field[idx_field(a, x-1, y, p_i)]
            + (2.0/3.0) * Field[idx_field(a, x+1, y, p_i)]
            - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]
        ) / lsx
    else:
        d1fd1x[idx_d1(0, a, x, y, p_i)] = 0.0
```

**Accuracy** — Fourth-order: error $\mathcal{O}(\Delta x^4)$

#### Second derivatives

Fourth-order Laplacian $\partial_{xx} f$:

$$
\frac{\partial^2 f}{\partial x^2}\bigg|_{i,j} \approx \frac{1}{(\Delta x)^2}\left(-\frac{1}{12}f_{i-2,j} + \frac{4}{3}f_{i-1,j} - \frac{5}{2}f_{i,j} + \frac{4}{3}f_{i+1,j} - \frac{1}{12}f_{i+2,j}\right)
$$

Similarly for $\partial_{yy}$ and mixed derivatives $\partial_{xy}$ (via chain rule from diagonals).

### Boundary conditions

Fields satisfy **Dirichlet boundary conditions** on the computational domain. The halo region fixes boundary values.

For a field with `halo=2`:
- Interior points: `2 ≤ x < nx - 2`
- Boundary points: `0 ≤ x < 2` or `nx - 2 ≤ x < nx`

### Stability and accuracy

**Stability criterion** — The Courant condition for explicit time-stepping:

$$
\Delta t \leq C \cdot \min(\Delta x, \Delta y)^2
$$

where $C$ is a theory-dependent constant (default Courant number ≈ 0.5).

**Accuracy** — Overall fourth-order in space, second-order in time (RK4).

## Arrested Newton Flow Algorithm

ANF is a second-order relaxation scheme combining fast convergence with stability.

### Algorithm description

The continuous ANF equation is:

$$
\ddot{\phi} = -\nabla_\phi E_h[\phi]
$$

This is a system of ODEs in velocity $v = \dot{\phi}$ and position $\phi$:

$$
\begin{aligned}
\dot{\phi} &= v \\
\dot{v} &= -\nabla_\phi E_h[\phi]
\end{aligned}
$$

**Arrest rule** — If energy increases, velocity is reset:

$$
v^{n+1} \leftarrow 0 \quad \text{if} \quad E_h[\phi^{n+1}] > E_h[\phi^n]
$$

### Convergence properties

- **Second-order dynamics** — Velocity accelerates motion; useful for escaping shallow regions
- **Energy stable** — Arrest prevents oscillations; energy is non-increasing (except at resets)
- **Faster than gradient descent** — Particularly for multi-soliton configurations
- **Convergence criterion** — Stop when $\|\nabla E\|_\infty < \epsilon$ (convergence tolerance)

### GPU implementation: RK4 integration

One full ANF step consists of:

1. **Compute gradient** — $\nabla E$ at all grid points via finite differences
2. **RK4 step** — Advance $(φ, v)$ using fourth-order Runge-Kutta
3. **Arrest check** — Reset velocity if energy increases
4. **Convergence check** — Test $\|\nabla E\|_\infty$ against tolerance

#### RK4 stages

The RK4 method for $\dot{\mathbf{y}} = \mathbf{f}(t, \mathbf{y})$ is:

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t, \mathbf{y}) \\
\mathbf{k}_2 &= \mathbf{f}(t + \Delta t/2, \mathbf{y} + \Delta t \mathbf{k}_1/2) \\
\mathbf{k}_3 &= \mathbf{f}(t + \Delta t/2, \mathbf{y} + \Delta t \mathbf{k}_2/2) \\
\mathbf{k}_4 &= \mathbf{f}(t + \Delta t, \mathbf{y} + \Delta t \mathbf{k}_3) \\
\mathbf{y}^{n+1} &= \mathbf{y}^n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
$$

For the ANF system, $\mathbf{y} = (\phi, v)$ and:

$$
\mathbf{f}(t, \mathbf{y}) = \begin{pmatrix} v \\ -\nabla_\phi E_h[\phi] \end{pmatrix}
$$

The implementation launches four gradient computation kernels (one per RK4 stage) and updates field and velocity buffers.

## Solver Configuration

The solver is configured via the `Params` and `ResolvedParams` classes.

### Core parameters

**Grid and domain:**

```python
Params(
    xlen=256,              # Grid points in x
    ylen=256,              # Grid points in y
    xsize=80.0,            # Physical size in x
    ysize=80.0,            # Physical size in y
    halo=2,                # Halo width for stencils
)
```

**Time stepping:**

```python
Params(
    time_step=0.01,        # Explicit time step (if set)
    courant=0.5,           # Courant number for auto time stepping
)
```

If `time_step` is `None`, it is computed from the Courant number:

$$
\Delta t = \text{courant} \cdot \min(\Delta x, \Delta y)^2
$$

**Arrested Newton flow:**

```python
Params(
    newtonflow=True,       # Enable Newton flow (vs. gradient descent)
    killkinen=True,        # Reset velocity on energy increase
)
```

**Constraints:**

```python
Params(
    unit_magnetization=False,  # Enforce unit norm on magnetization (some theories)
)
```

### Resolved parameters

After calling `.resolved()`, derived parameters are computed:

```python
rp = params.resolved()

print(rp.xlen, rp.ylen)              # Grid dimensions
print(rp.lsx, rp.lsy)                # Lattice spacings
print(rp.grid_volume)                # Cell area
print(rp.time_step)                  # Actual time step
print(rp.dim_grid)                   # Total grid points
print(rp.dim_fields)                 # Total field values
```

### Example configuration

```python
from soliton_solver.theories import load_theory

theory = load_theory("Chiral magnet")

# Create base parameters
params = theory.params.default_params(
    xlen=512, ylen=512,      # Finer grid
    xsize=20.0, ysize=20.0,  # Larger domain
    courant=0.3,             # Conservative time step
    newtonflow=True,
    killkinen=True,
)

# Resolve and create simulation
sim = Simulation(params, theory)
sim.initialize({"mode": "ground"})

# Relax via arrested Newton flow
from soliton_solver.core.simulation import Simulation
energy = sim.observables_mod.compute_energy(
    sim.Field, sim.d1fd1x, sim.en, sim.entmp, 
    sim.gridsum_partial, sim.p_i_d, sim.p_f_d, 
    sim.p_i_h, sim.p_f_h
)

# Advance one step
new_energy, err = sim.step(prev_energy=energy)
print(f"Energy: {new_energy}, Gradient norm: {err}")
```

## Performance considerations

### Time step selection

- **Too small** — Slow convergence, many unnecessary steps
- **Too large** — Instability, divergence, energy oscillations

The Courant condition provides automatic stability; manual override via `time_step` should be conservative.

### Grid resolution

- **Coarse grids** — Fast but inaccurate; may miss fine soliton details
- **Fine grids** — Accurate but slow; scales as $O(n_x n_y)$ per step

Recommended: start coarse, refine around features of interest.

### Convergence tolerance

Typically $10^{-4}$ to $10^{-6}$ depending on application:

```python
# Run until convergence
energy = initial_energy
for step in range(max_steps):
    new_energy, err = sim.step(energy)
    energy = new_energy
    
    if err < 1e-4:
        print(f"Converged in {step} steps")
        break
    
    if step % 100 == 0:
        print(f"Step {step}: E={energy:.6e}, ||∇E||_∞={err:.6e}")
```

### Memory usage

Field arrays are stored in GPU device memory:

$$
\text{Memory per field} = n_x \cdot n_y \cdot 8 \text{ bytes} \quad (\text{float64})
$$

For multiple fields and buffers (Field, Velocity, derivatives, RK4 stages):

$$
\text{Total} \approx 10 \times n_x \cdot n_y \cdot 8 \text{ bytes}
$$

Example: 1024×1024 grid with 4 fields ≈ 80 MB on GPU.

### Typical runtime

A single RK4 step on NVIDIA A100:
- 512×512, 2–4 fields: ~1–2 ms
- 1024×1024, 4–8 fields: ~5–10 ms

Convergence typically requires 1,000–10,000 steps depending on configuration complexity.

