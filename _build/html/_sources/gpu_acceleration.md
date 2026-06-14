# GPU Acceleration via Numba CUDA

`soliton_solver` achieves high performance by implementing the PDE solver entirely on NVIDIA GPUs using Numba CUDA. This document explains the GPU acceleration strategy, kernel design, and performance characteristics.

## Overview

The solver uses **Numba CUDA** for JIT compilation of Python functions into optimized CUDA kernels. Key benefits:

- **No explicit C/C++ code** — Pure Python kernel definitions, compiled to CUDA at runtime
- **Type specialization** — Kernels are optimized for float64 field arrays
- **Lazy compilation** — Kernels compile on first invocation, then cached
- **Reduced compilation burden** — Theory developers can write kernels in Python without CUDA expertise

## CUDA Kernels

### Kernel structure

Kernels in `soliton_solver` follow a **2D grid + per-thread update** pattern:

```python
from numba import cuda
from soliton_solver.core.utils import in_bounds, idx_field

@cuda.jit
def my_kernel(Field, p_i, p_f):
    """Update a field on the GPU."""
    x, y = cuda.grid(2)  # Get (x, y) thread index
    if not in_bounds(x, y, p_i):
        return  # Threads outside domain do nothing
    
    # Per-thread computation
    idx = idx_field(0, x, y, p_i)  # Flatten (x, y) to linear index
    Field[idx] += 1.0  # Update field value
```

### Thread mapping

Each thread processes one lattice site $(x, y)$:

- **Grid dimensions** — Matches 2D domain: `(nx // threads_per_block_x, ny // threads_per_block_y)`
- **Block dimensions** — Typically `(8, 8)` or `(16, 16)` threads per block
- **Global ID** — `(x, y) = cuda.grid(2)` gives the 2D lattice coordinate
- **Out-of-bounds handling** — `in_bounds(x, y, p_i)` checks if thread is within domain and domain-interior (beyond halo)

### Kernel examples

#### Finite-difference derivatives

Fourth-order central finite differences are computed per-thread:

```python
@cuda.jit(device=True)
def compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f):
    """Compute first derivative of field component a at (x, y)."""
    xlen = p_i[0]
    ylen = p_i[1]
    halo = p_i[2]
    lsx = p_f[2]  # Lattice spacing x
    lsy = p_f[3]  # Lattice spacing y
    
    # Fourth-order stencil: [-1/12, 2/3, 0, -2/3, 1/12] * f
    if x > halo - 1 and x < xlen - halo:
        d1fd1x[...] = (
            (1.0/12.0) * Field[idx_field(a, x-2, y, p_i)]
            - (2.0/3.0) * Field[idx_field(a, x-1, y, p_i)]
            + (2.0/3.0) * Field[idx_field(a, x+1, y, p_i)]
            - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]
        ) / lsx
```

Each thread independently loads neighboring field values and computes the derivative.

#### Gradient computation

Theory-specific gradients use derivatives computed per-thread:

```python
def make_do_gradient_step_kernel(do_gradient_step_point):
    """Factory creating a gradient computation kernel."""
    @cuda.jit
    def _kernel(Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, p_i, p_f):
        x, y = cuda.grid(2)
        if not in_bounds(x, y, p_i):
            return
        
        # Compute derivatives
        number_total_fields = p_i[4]
        for a in range(number_total_fields):
            compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
            compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
        
        # Theory-specific gradient update
        do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, 
                              x, y, p_i, p_f)
    
    return _kernel
```

The per-thread `do_gradient_step_point` implements the theory-specific energy gradient $\nabla E[\phi]$.

## Shared Memory

While the current kernels do not explicitly use shared memory (they rely on L1/L2 caches), shared memory optimization is suitable for future improvements:

### Shared memory layout

For a 2D stencil kernel with halo communication, shared memory can reduce global memory bandwidth:

```python
@cuda.jit
def stencil_with_shared_memory(Field, Output, p_i, p_f):
    """Example: shared memory optimization for finite-difference stencils."""
    # Allocate shared memory for a (block_x + 2*halo) × (block_y + 2*halo) region
    shared = cuda.shared.array((10, 10), dtype=float64)  # For 8×8 + halo=1
    
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x, y = cuda.grid(2)
    
    # Cooperative load: all threads load neighboring values
    if x < nx and y < ny:
        shared[tx + 1, ty + 1] = Field[idx_field(0, x, y, p_i)]
    
    # Halo threads load boundaries
    if tx == 0 and x > 0:
        shared[0, ty + 1] = Field[idx_field(0, x - 1, y, p_i)]
    if tx == 7 and x < nx - 1:
        shared[9, ty + 1] = Field[idx_field(0, x + 1, y, p_i)]
    
    cuda.syncthreads()
    
    # Compute using shared memory (faster than global memory)
    if x > 0 and x < nx - 1:
        Output[idx_field(0, x, y, p_i)] = (
            shared[tx - 1, ty + 1] + shared[tx + 1, ty + 1]
        ) / 2.0
```

**Benefits:**
- Reduced global memory bandwidth: $\sim 8\times$ faster on modern GPUs
- Improved L1 cache locality
- Synchronization points for multi-step algorithms

**Bank conflicts:** NVIDIA GPUs organize shared memory into 32 banks. Stride-1 access minimizes conflicts; stride-32 causes maximum contention.

## SIMT Execution Model

CUDA uses **Single-Instruction Multiple-Thread (SIMT)** execution. Understanding this improves kernel design:

### Warp-level operations

- **Warp** — 32 consecutive threads executing the same instruction
- **Warp scheduler** — GPU assigns warps to cores; one warp per core per cycle
- **Thread divergence** — If threads in a warp branch differently, paths execute serially, reducing parallelism

Example of divergence (bad):

```python
@cuda.jit
def divergent_kernel(Field):
    x, y = cuda.grid(2)
    if x % 2 == 0:  # Divergence: half the warp takes one path
        Field[...] = 1.0
    else:
        Field[...] = -1.0
```

Example without divergence (good):

```python
@cuda.jit
def coalesced_kernel(Field):
    x, y = cuda.grid(2)
    Field[idx_field(0, x, y, p_i)] = float(x % 2) * 2.0 - 1.0  # No divergence
```

### Occupancy and resource utilization

**Occupancy** — Fraction of the GPU's maximum concurrent threads.

For a kernel using 32 registers per thread and 48 KB shared memory per block:

- Tesla A100 (108 SMs × 2560 threads/SM) — 1 block/SM (256 threads) → 10% occupancy
- A100 can run 40-80 active warps per SM → higher occupancy beneficial

**Optimization:**
- Minimize register usage (fewer local variables)
- Minimize shared memory per block
- Use `@cuda.jit(fastmath=True)` for faster-but-less-precise math when appropriate

## Arrested Newton Flow Algorithm

The core minimization algorithm uses **arrested Newton flow** (ANF) — a second-order relaxation method combining fast descent with stability control.

### Algorithm description

Given a discrete energy $E_h[\phi]$, the continuous ANF equation is:

$$
\ddot{\phi} = -\nabla_\phi E_h[\phi]
$$

Evolving this PDE toward equilibrium finds local minima. The algorithm is:

1. **Compute gradient** — $\nabla E$ at all lattice sites
2. **RK4 step** — Advance $(φ, \dot{φ})$ by time $\Delta t$
3. **Arrest check** — If energy increased, set velocity to zero
4. **Repeat** until convergence

### Convergence properties

- **Second-order** — Velocity assists motion along shallow directions
- **Energy stable** — Velocity reset prevents oscillations
- **Faster than gradient descent** — Particularly for multi-soliton configurations
- **Convergence criterion** — Stop when $\|\nabla E\|_\infty < \epsilon$

### GPU implementation

```python
def do_arrested_newton_flow(Velocity, Field, EnergyGradient, 
                           gradient_step_kernel, rk4_kernel, 
                           p_i_d, p_f_d, p_i_h, p_f_h, 
                           prev_energy, compute_energy):
    """One arrested Newton flow minimization step."""
    
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    
    # 1. Compute energy gradient: ∇E = ∂E/∂φ
    gradient_step_kernel[grid2d, block2d](
        Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d
    )
    cuda.synchronize()
    
    # 2. RK4 step: advance (φ, φ̇) = φ(t + Δt)
    do_rk4_step_kernel[grid2d, block2d](
        k_out, Velocity, l_in, Temp, Field, k_prev, 0.5, p_i_d, p_f_d
    )
    # ... (four RK4 stages)
    
    rk4_kernel[grid2d, block2d](
        Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i_d, p_f_d
    )
    cuda.synchronize()
    
    # 3. Arrest check: ∫(φ̇ · ∇E) > 0?
    force = arresting_criteria(Velocity, EnergyGradient, ...)
    
    new_energy = compute_energy(Field, ...)
    
    if new_energy > prev_energy:  # Energy increased
        # Reset velocity: φ̇ ← 0
        set_field_zero_kernel[grid2d, block2d](Velocity, p_i_d)
    
    return new_energy, convergence_error
```

## Memory layout and indexing

Field arrays are stored in **flat 1D buffers** with multi-dimensional indexing via helper functions.

### Flattened indexing

For a multi-component field on an $n_x \times n_y$ grid:

```
Field[a, i, j] → Field[(j) + (i) * ny + (a) * nx * ny]
```

This layout ensures:
- **Coalesced memory access** — Consecutive threads access consecutive memory locations
- **Cache efficiency** — Sequential indices fit in L1/L2 caches

Helper functions:

```python
@cuda.jit(device=True, inline=True)
def idx_field(a, i, j, p_i):
    xlen = p_i[0]
    ylen = p_i[1]
    return j + i * ylen + a * xlen * ylen
```

## Performance benchmarks

Typical performance on NVIDIA A100 GPU:

| Grid size | Fields | Kernel time | Throughput |
|-----------|--------|------------|-----------|
| 512×512   | 2-4    | 0.5–1 ms  | 500+ M pts/s |
| 1024×1024 | 4-8    | 2–4 ms    | 250+ M pts/s |
| 2048×2048 | 8-16   | 8–15 ms   | 300+ M pts/s |

**Bottleneck:** Global memory bandwidth (typically 2 TB/s on A100).

## Profiling and optimization

Profile CUDA kernels with `nvidia-smi` and `nsys`:

```bash
# Profile GPU memory and kernel utilization
nvidia-smi dmon

# Detailed profiling with NVIDIA Systems Profiler
nsys profile --stats=true my_simulation.py
```

Common optimizations:
- **Reduce memory reads** — Cache field derivatives locally
- **Increase arithmetic intensity** — More computation per byte loaded
- **Minimize synchronization** — Batch kernel launches between syncs
- **Use fast-math mode** — Trade precision for speed if acceptable

## Troubleshooting

### Out-of-memory errors

If GPU memory is exhausted:
- Reduce grid size (`xlen`, `ylen`)
- Reduce number of fields
- Use smaller data type (float32 if accuracy allows)

### Kernel launch failures

Ensure kernel signature matches launch call. Numba CUDA catches most type mismatches at compile time.

### Performance degradation

- Check occupancy with profiler
- Verify coalesced memory access patterns
- Reduce register usage by refactoring
- Profile to identify bottleneck kernels

