"""
CUDA device helpers for 2D finite difference derivatives on flattened multi component fields.

Examples
--------
Use ``compute_derivative_first`` inside a CUDA kernel to compute first spatial derivatives of a field component.
Use ``compute_derivative_second`` inside a CUDA kernel to compute second spatial derivatives of a field component.
"""

from numba import cuda
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2

@cuda.jit(device=True)
def compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f):
    """
    Compute fourth order central first derivatives of a field component at a lattice site.

    Parameters
    ----------
    d1fd1x : device array
        Flattened buffer storing first derivatives.
    Field : device array
        Flattened buffer storing field values.
    a : int
        Field component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing grid dimensions and halo size.
    p_f : device array
        Float parameter array containing lattice spacings.

    Returns
    -------
    None
        The derivative values are written into ``d1fd1x`` in place.

    Raises
    ------
    None.

    Examples
    --------
    Call ``compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)`` from within a CUDA kernel over lattice sites.
    """
    xlen = p_i[0]; ylen = p_i[1]; halo = p_i[2]
    lsx = p_f[2]; lsy = p_f[3]
    if x > halo - 1 and x < xlen - halo:
        d1fd1x[idx_d1(0, a, x, y, p_i)] = ((1.0/12.0) * Field[idx_field(a, x-2, y, p_i)] - (2.0/3.0) * Field[idx_field(a, x-1, y, p_i)] + (2.0/3.0) * Field[idx_field(a, x+1, y, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]) / lsx
    else:
        d1fd1x[idx_d1(0, a, x, y, p_i)] = 0.0
    if y > halo - 1 and y < ylen - halo:
        d1fd1x[idx_d1(1, a, x, y, p_i)] = ((1.0/12.0) * Field[idx_field(a, x, y-2, p_i)] - (2.0/3.0) * Field[idx_field(a, x, y-1, p_i)] + (2.0/3.0) * Field[idx_field(a, x, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x, y+2, p_i)]) / lsy
    else:
        d1fd1x[idx_d1(1, a, x, y, p_i)] = 0.0

@cuda.jit(device=True)
def compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f):
    """
    Compute fourth order central second derivatives of a field component at a lattice site.

    Parameters
    ----------
    d2fd2x : device array
        Flattened buffer storing second derivatives.
    Field : device array
        Flattened buffer storing field values.
    a : int
        Field component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing grid dimensions and halo size.
    p_f : device array
        Float parameter array containing lattice spacings and grid volume.

    Returns
    -------
    None
        The derivative values are written into ``d2fd2x`` in place.

    Raises
    ------
    None.

    Examples
    --------
    Call ``compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)`` from within a CUDA kernel over lattice sites.
    """
    xlen = p_i[0]; ylen = p_i[1]; halo = p_i[2]
    lsx = p_f[2]; lsy = p_f[3]
    grid_volume = p_f[4]
    if x > halo - 1 and x < xlen - halo:
        d2fd2x[idx_d2(0, 0, a, x, y, p_i)] = (-(1.0/12.0) * Field[idx_field(a, x-2, y, p_i)] + (4.0/3.0) * Field[idx_field(a, x-1, y, p_i)] - (5.0/2.0) * Field[idx_field(a, x,   y, p_i)] + (4.0/3.0) * Field[idx_field(a, x+1, y, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]) / (lsx * lsx)
    else:
        d2fd2x[idx_d2(0, 0, a, x, y, p_i)] = 0.0
    if y > halo - 1 and y < ylen - halo:
        d2fd2x[idx_d2(1, 1, a, x, y, p_i)] = (-(1.0/12.0) * Field[idx_field(a, x, y-2, p_i)] + (4.0/3.0) * Field[idx_field(a, x, y-1, p_i)] - (5.0/2.0) * Field[idx_field(a, x, y,   p_i)] + (4.0/3.0) * Field[idx_field(a, x, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x, y+2, p_i)]) / (lsy * lsy)
    else:
        d2fd2x[idx_d2(1, 1, a, x, y, p_i)] = 0.0
    if (x > halo) and (y > halo) and (x < xlen - halo) and (y < ylen - halo):
        diag = (-(1.0/12.0) * Field[idx_field(a, x-2, y-2, p_i)] + (4.0/3.0) * Field[idx_field(a, x-1, y-1, p_i)] - (5.0/2.0) * Field[idx_field(a, x,   y,   p_i)] + (4.0/3.0) * Field[idx_field(a, x+1, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y+2, p_i)]) / (2.0 * grid_volume)

        dxx_half = (-(1.0/12.0) * Field[idx_field(a, x-2, y, p_i)] + (4.0/3.0) * Field[idx_field(a, x-1, y, p_i)] - (5.0/2.0) * Field[idx_field(a, x,   y, p_i)] + (4.0/3.0) * Field[idx_field(a, x+1, y, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]) / (2.0 * lsx * lsx)

        dyy_half = (-(1.0/12.0) * Field[idx_field(a, x, y-2, p_i)] + (4.0/3.0) * Field[idx_field(a, x, y-1, p_i)] - (5.0/2.0) * Field[idx_field(a, x, y,   p_i)] + (4.0/3.0) * Field[idx_field(a, x, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x, y+2, p_i)]) / (2.0 * lsy * lsy)

        d2fd2x[idx_d2(1, 0, a, x, y, p_i)] = diag - dxx_half - dyy_half
    else:
        d2fd2x[idx_d2(1, 0, a, x, y, p_i)] = 0.0
    d2fd2x[idx_d2(0, 1, a, x, y, p_i)] = d2fd2x[idx_d2(1, 0, a, x, y, p_i)]