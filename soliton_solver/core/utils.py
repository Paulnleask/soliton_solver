# =====================================================================================
# soliton_solver/core/utils.py
# =====================================================================================
"""
Core CUDA utilities for indexing, launch configuration, and simple reductions.

Purpose:
    Provides device-side index helpers for flattened arrays, bounds checks, common
    initialization kernels, and host-side wrappers for sum/min/max reductions.

Usage:
    - Use idx_field/idx_d1/idx_d2 inside CUDA kernels to map (component, x, y) to flat indices.
    - Use launch_2d(...) to pick a (grid, block) configuration for (x, y) kernels.
    - Use compute_sum/compute_max/compute_min to reduce 1D device arrays to scalars.

Outputs:
    - Defines core device functions and CUDA kernels used throughout the solver.
    - Provides host functions returning scalar reductions computed on the GPU.
"""
# ---------------- Imports ----------------
from __future__ import annotations
from numba import cuda, float64

MAX_TPB = 1024

# ---------------- Indexing by array unrolling----------------
@cuda.jit(device=True, inline=True)
def idx_field(a, i, j, p_i):
    """
    Map (field component, x, y) to a flat index for Field-like arrays.

    Usage:
        k = idx_field(a, x, y, p_i)
        Field[k] = ...

    Parameters:
        a: Field component index.
        i, j: Lattice indices (x=i, y=j).
        p_i: Device int parameter array (uses p_i[0]=xlen, p_i[1]=ylen).

    Outputs:
        - Returns the flat index into a Field buffer of length number_total_fields * xlen * ylen.
    """
    xlen = p_i[0]; ylen = p_i[1]
    return j + i * ylen + a * xlen * ylen

# ---------------- First derivative indexing ----------------
@cuda.jit(device=True, inline=True)
def idx_d1(coord, field, i, j, p_i):
    """
    Map (coord, field component, x, y) to a flat index for first-derivative buffers.

    Usage:
        k = idx_d1(coord, a, x, y, p_i)
        d1fd1x[k] = ...

    Parameters:
        coord: Coordinate index (0..number_coordinates-1).
        field: Field component index.
        i, j: Lattice indices.
        p_i: Device int parameter array (uses p_i[0], p_i[1], p_i[3]=number_coordinates).

    Outputs:
        - Returns the flat index into a first-derivative buffer laid out by (field, coord, x, y).
    """
    xlen = p_i[0]; ylen = p_i[1]
    number_coordinates = p_i[3]
    return i + j * xlen + coord * xlen * ylen + field * xlen * ylen * number_coordinates

# ---------------- Second derivative indexing ----------------
@cuda.jit(device=True, inline=True)
def idx_d2(coord1, coord2, field, i, j, p_i):
    """
    Map (coord1, coord2, field component, x, y) to a flat index for second-derivative buffers.

    Usage:
        k = idx_d2(c1, c2, a, x, y, p_i)
        d2fd2x[k] = ...

    Parameters:
        coord1, coord2: Coordinate indices (0..number_coordinates-1).
        field: Field component index.
        i, j: Lattice indices.
        p_i: Device int parameter array (uses p_i[0], p_i[1], p_i[3]=number_coordinates).

    Outputs:
        - Returns the flat index into a second-derivative buffer laid out by (field, coord1, coord2, x, y).
    """
    xlen = p_i[0]; ylen = p_i[1]
    number_coordinates = p_i[3]
    return i + j * xlen + coord1 * xlen * ylen + coord2 * xlen * ylen * number_coordinates + field * xlen * ylen * number_coordinates * number_coordinates

# ---------------- Bounds check ----------------
@cuda.jit(device=True, inline=True)
def in_bounds(x, y, p_i):
    """
    Check whether (x, y) lies within the active lattice extents.

    Usage:
        if not in_bounds(x, y, p_i):
            return

    Parameters:
        x, y: Lattice indices.
        p_i: Device int parameter array (uses p_i[0]=xlen, p_i[1]=ylen).

    Outputs:
        - Returns True if 0 <= x < xlen and 0 <= y < ylen, else False.
    """
    xlen = p_i[0]; ylen = p_i[1]
    return (0 <= x < xlen) and (0 <= y < ylen)

# ---------------- Nullify field ----------------
@cuda.jit
def set_field_zero_kernel(Field, p_i):
    """
    Set all field components to zero at each lattice site.

    Usage:
        set_field_zero_kernel[grid2d, block2d](Field, p_i)

    Parameters:
        Field: Device array storing flattened field components.
        p_i: Device int parameter array (uses p_i[4]=number_total_fields and grid sizes).

    Outputs:
        - Writes zeros into Field for all components and all in-bounds lattice sites.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Field[idx_field(a, x, y, p_i)] = 0.0

# ---------------- Launches 2D CUDA grid ----------------
def launch_2d(p_i_h, threads=(16, 32)):
    """
    Compute a 2D CUDA launch configuration covering the full lattice.

    Usage:
        grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))
        kernel[grid2d, block2d](...)

    Parameters:
        p_i_h: Host int parameter array (uses p_i_h[0]=xlen, p_i_h[1]=ylen).
        threads: (bx, by) threads per block.

    Outputs:
        - Returns (grid2d, block2d) suitable for kernels using cuda.grid(2).
    """
    xlen = int(p_i_h[0]); ylen = int(p_i_h[1])
    bx, by = threads
    grid = ((xlen + bx - 1)//bx, (ylen + by - 1)//by)
    return grid, (bx, by)

# ---------------- Reduction to sum ----------------
@cuda.jit
def reduce_sum_kernel(var, partial, size):
    """
    Block-level reduction kernel computing partial sums over a 1D device array.

    Usage:
        reduce_sum_kernel[blocks, tpb](var_d, partial_d, size)

    Parameters:
        var: 1D device array to reduce.
        partial: 1D device array of length >= number of blocks to store block sums.
        size: Number of valid entries in var.

    Outputs:
        - Writes one partial sum per block into `partial`.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid

    v = 0.0
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()

    # conservative stride
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

# ---------------- Reduction to find maximum ----------------
@cuda.jit
def reduce_max_kernel(var, partial, size):
    """
    Block-level reduction kernel computing partial maxima over a 1D device array.

    Usage:
        reduce_max_kernel[blocks, tpb](var_d, partial_d, size)

    Parameters:
        var: 1D device array to reduce.
        partial: 1D device array of length >= number of blocks to store block maxima.
        size: Number of valid entries in var.

    Outputs:
        - Writes one partial maximum per block into `partial`.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid

    v = -1.0e300
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride and sdata[tid] < sdata[tid + stride]:
            sdata[tid] = sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

# ---------------- Reduction to determine minimum ----------------
@cuda.jit
def reduce_min_kernel(var, partial, size):
    """
    Block-level reduction kernel computing partial minima over a 1D device array.

    Usage:
        reduce_min_kernel[blocks, tpb](var_d, partial_d, size)

    Parameters:
        var: 1D device array to reduce.
        partial: 1D device array of length >= number of blocks to store block minima.
        size: Number of valid entries in var.

    Outputs:
        - Writes one partial minimum per block into `partial`.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid

    v = 1.0e300
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride and sdata[tid] > sdata[tid + stride]:
            sdata[tid] = sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

# ---------------- Compute sum of density ----------------
def compute_sum(var_d, partial_d, size: int) -> float:
    """
    Compute the sum of a 1D device array via a two-stage reduction.

    Usage:
        total = compute_sum(var_d, partial_d, size)

    Parameters:
        var_d: 1D device array to reduce.
        partial_d: Device array used to store block partials (length >= ceil(size/tpb)).
        size: Number of valid entries in var_d.

    Outputs:
        - Launches reduce_sum_kernel to fill partial_d.
        - Copies partial_d to host and returns the total sum as a float.
    """
    tpb = 1024
    blocks = (size + tpb - 1)//tpb
    reduce_sum_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host().sum())

# ---------------- Compute maximum of array ----------------
def compute_max(var_d, partial_d, size: int) -> float:
    """
    Compute the maximum of a 1D device array via a two-stage reduction.

    Usage:
        m = compute_max(var_d, partial_d, size)

    Parameters:
        var_d: 1D device array to reduce.
        partial_d: Device array used to store block partials (length >= ceil(size/tpb)).
        size: Number of valid entries in var_d.

    Outputs:
        - Launches reduce_max_kernel to fill partial_d.
        - Copies partial_d to host and returns the global maximum as a float.
    """
    tpb = 1024
    blocks = (size + tpb - 1)//tpb
    reduce_max_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host().max())

# ---------------- Compute minimum of array ----------------
def compute_min(var_d, partial_d, size):
    """
    Compute the minimum of a 1D device array via a two-stage reduction.

    Usage:
        m = compute_min(var_d, partial_d, size)

    Parameters:
        var_d: 1D device array to reduce.
        partial_d: Device array used to store block partials (length >= ceil(size/tpb)).
        size: Number of valid entries in var_d.

    Outputs:
        - Launches reduce_min_kernel to fill partial_d.
        - Copies partial_d to host and returns the global minimum as a float.
    """
    tpb = 1024
    blocks = (size + tpb - 1)//tpb
    reduce_min_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host().min())

# ---------------- Compute max(field) ----------------
def compute_max_field(EnergyGradient, max_partial, p_i_h):
    """
    Compute the maximum entry of the flattened EnergyGradient field.

    Usage:
        err = compute_max_field(EnergyGradient, max_partial, p_i_h)

    Parameters:
        EnergyGradient: Device array holding per-component gradient values.
        max_partial: Device array used to store block maxima for reduction.
        p_i_h: Host int parameter array (uses p_i_h[6]=dim_fields).

    Outputs:
        - Reduces EnergyGradient over dim_fields entries and returns the maximum as a float.
    """
    dim_fields = p_i_h[6]
    return compute_max(EnergyGradient, max_partial, int(dim_fields))