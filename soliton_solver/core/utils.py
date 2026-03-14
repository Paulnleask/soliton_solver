"""
Core CUDA utilities for indexing, launch configuration, and simple reductions.

Examples
--------
Use ``idx_field``, ``idx_d1``, and ``idx_d2`` inside CUDA kernels to map multi index access to flattened arrays.
Use ``launch_2d`` to construct a two dimensional CUDA launch configuration.
Use ``compute_sum``, ``compute_max``, and ``compute_min`` to reduce one dimensional device arrays to scalars.
"""

from __future__ import annotations
from numba import cuda, float64

MAX_TPB = 1024

@cuda.jit(device=True, inline=True)
def idx_field(a, i, j, p_i):
    """
    Map a field component and lattice site to a flattened field index.

    Parameters
    ----------
    a : int
        Field component index.
    i : int
        Lattice index along the x direction.
    j : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions.

    Returns
    -------
    int
        Flattened index into a field like array.

    Examples
    --------
    Use ``k = idx_field(a, x, y, p_i)`` to access a flattened field buffer inside a CUDA kernel.
    """
    xlen = p_i[0]; ylen = p_i[1]
    return j + i * ylen + a * xlen * ylen

@cuda.jit(device=True, inline=True)
def idx_d1(coord, field, i, j, p_i):
    """
    Map a coordinate, field component, and lattice site to a flattened first derivative index.

    Parameters
    ----------
    coord : int
        Coordinate index.
    field : int
        Field component index.
    i : int
        Lattice index along the x direction.
    j : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions and coordinate count.

    Returns
    -------
    int
        Flattened index into a first derivative buffer.

    Examples
    --------
    Use ``k = idx_d1(coord, a, x, y, p_i)`` to access a first derivative buffer inside a CUDA kernel.
    """
    xlen = p_i[0]; ylen = p_i[1]
    number_coordinates = p_i[3]
    return i + j * xlen + coord * xlen * ylen + field * xlen * ylen * number_coordinates

@cuda.jit(device=True, inline=True)
def idx_d2(coord1, coord2, field, i, j, p_i):
    """
    Map two coordinates, a field component, and a lattice site to a flattened second derivative index.

    Parameters
    ----------
    coord1 : int
        First coordinate index.
    coord2 : int
        Second coordinate index.
    field : int
        Field component index.
    i : int
        Lattice index along the x direction.
    j : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions and coordinate count.

    Returns
    -------
    int
        Flattened index into a second derivative buffer.

    Examples
    --------
    Use ``k = idx_d2(c1, c2, a, x, y, p_i)`` to access a second derivative buffer inside a CUDA kernel.
    """
    xlen = p_i[0]; ylen = p_i[1]
    number_coordinates = p_i[3]
    return i + j * xlen + coord1 * xlen * ylen + coord2 * xlen * ylen * number_coordinates + field * xlen * ylen * number_coordinates * number_coordinates

@cuda.jit(device=True, inline=True)
def in_bounds(x, y, p_i):
    """
    Check whether a lattice site lies inside the active grid.

    Parameters
    ----------
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing the lattice dimensions.

    Returns
    -------
    bool
        ``True`` if the site lies inside the lattice and ``False`` otherwise.

    Examples
    --------
    Use ``if not in_bounds(x, y, p_i): return`` to guard CUDA kernel work outside the lattice.
    """
    xlen = p_i[0]; ylen = p_i[1]
    return (0 <= x < xlen) and (0 <= y < ylen)

@cuda.jit
def set_field_zero_kernel(Field, p_i):
    """
    Set all field components to zero at each lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field buffer.
    p_i : device array
        Integer parameter array containing the grid dimensions and field count.

    Returns
    -------
    None
        The field buffer is set to zero in place.

    Examples
    --------
    Launch ``set_field_zero_kernel[grid2d, block2d](Field, p_i)`` to zero a device field buffer.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Field[idx_field(a, x, y, p_i)] = 0.0

def launch_2d(p_i_h, threads=(16, 32)):
    """
    Compute a two dimensional CUDA launch configuration covering the full lattice.

    Parameters
    ----------
    p_i_h : host array
        Integer parameter array containing the lattice dimensions.
    threads : tuple, optional
        Threads per block given as ``(bx, by)``.

    Returns
    -------
    tuple
        Pair ``(grid2d, block2d)`` suitable for kernels using ``cuda.grid(2)``.

    Examples
    --------
    Use ``grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))`` before launching a two dimensional CUDA kernel.
    """
    xlen = int(p_i_h[0]); ylen = int(p_i_h[1])
    bx, by = threads
    grid = ((xlen + bx - 1)//bx, (ylen + by - 1)//by)
    return grid, (bx, by)

@cuda.jit
def reduce_sum_kernel(var, partial, size):
    """
    Compute block level partial sums for a one dimensional device array.

    Parameters
    ----------
    var : device array
        One dimensional device array to reduce.
    partial : device array
        Output array storing one partial sum per block.
    size : int
        Number of valid entries in ``var``.

    Returns
    -------
    None
        Partial sums are written into ``partial``.

    Examples
    --------
    Launch ``reduce_sum_kernel[blocks, tpb](var_d, partial_d, size)`` to compute partial sums on the device.
    """
    sdata = cuda.shared.array(MAX_TPB, float64)
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid

    v = 0.0
    if idx < size:
        v = var[idx]
    sdata[tid] = v
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial[cuda.blockIdx.x] = sdata[0]

@cuda.jit
def reduce_max_kernel(var, partial, size):
    """
    Compute block level partial maxima for a one dimensional device array.

    Parameters
    ----------
    var : device array
        One dimensional device array to reduce.
    partial : device array
        Output array storing one partial maximum per block.
    size : int
        Number of valid entries in ``var``.

    Returns
    -------
    None
        Partial maxima are written into ``partial``.

    Examples
    --------
    Launch ``reduce_max_kernel[blocks, tpb](var_d, partial_d, size)`` to compute partial maxima on the device.
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

@cuda.jit
def reduce_min_kernel(var, partial, size):
    """
    Compute block level partial minima for a one dimensional device array.

    Parameters
    ----------
    var : device array
        One dimensional device array to reduce.
    partial : device array
        Output array storing one partial minimum per block.
    size : int
        Number of valid entries in ``var``.

    Returns
    -------
    None
        Partial minima are written into ``partial``.

    Examples
    --------
    Launch ``reduce_min_kernel[blocks, tpb](var_d, partial_d, size)`` to compute partial minima on the device.
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

def compute_sum(var_d, partial_d, size: int) -> float:
    """
    Compute the sum of a one dimensional device array.

    Parameters
    ----------
    var_d : device array
        One dimensional device array to reduce.
    partial_d : device array
        Device array used to store partial sums.
    size : int
        Number of valid entries in ``var_d``.

    Returns
    -------
    float
        Sum of the array entries.

    Examples
    --------
    Use ``total = compute_sum(var_d, partial_d, size)`` to reduce a device array to a scalar sum.
    """
    tpb = 1024
    blocks = (size + tpb - 1)//tpb
    reduce_sum_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host().sum())

def compute_max(var_d, partial_d, size: int) -> float:
    """
    Compute the maximum of a one dimensional device array.

    Parameters
    ----------
    var_d : device array
        One dimensional device array to reduce.
    partial_d : device array
        Device array used to store partial maxima.
    size : int
        Number of valid entries in ``var_d``.

    Returns
    -------
    float
        Maximum array entry.

    Examples
    --------
    Use ``m = compute_max(var_d, partial_d, size)`` to reduce a device array to its maximum value.
    """
    tpb = 1024
    blocks = (size + tpb - 1)//tpb
    reduce_max_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host().max())

def compute_min(var_d, partial_d, size):
    """
    Compute the minimum of a one dimensional device array.

    Parameters
    ----------
    var_d : device array
        One dimensional device array to reduce.
    partial_d : device array
        Device array used to store partial minima.
    size : int
        Number of valid entries in ``var_d``.

    Returns
    -------
    float
        Minimum array entry.

    Examples
    --------
    Use ``m = compute_min(var_d, partial_d, size)`` to reduce a device array to its minimum value.
    """
    tpb = 1024
    blocks = (size + tpb - 1)//tpb
    reduce_min_kernel[blocks, tpb](var_d, partial_d, size)
    return float(partial_d.copy_to_host().min())

def compute_max_field(EnergyGradient, max_partial, p_i_h):
    """
    Compute the maximum entry of the flattened energy gradient field.

    Parameters
    ----------
    EnergyGradient : device array
        Device array containing the flattened energy gradient.
    max_partial : device array
        Device array used to store partial maxima.
    p_i_h : host array
        Integer parameter array containing the flattened field size.

    Returns
    -------
    float
        Maximum entry of ``EnergyGradient``.

    Examples
    --------
    Use ``err = compute_max_field(EnergyGradient, max_partial, p_i_h)`` to compute the maximum gradient entry.
    """
    dim_fields = p_i_h[6]
    return compute_max(EnergyGradient, max_partial, int(dim_fields))