# =====================================================================================
# soliton_solver/core/derivatives.py
# =====================================================================================
"""
CUDA device helpers for 2D finite-difference derivatives on flattened multi-component fields.

Purpose:
    Provides device functions that compute first and second spatial derivatives of
    Field[a](x, y) using 4th-order central finite differences on interior points.

Usage:
    - Call compute_derivative_first(...) and compute_derivative_second(...) from within a
      global CUDA kernel that iterates over (x, y) and loops over field components `a`.
    - Derivatives are written into preallocated flattened buffers (d1fd1x, d2fd2x).

Outputs:
    - Writes first-derivative components (dx, dy) into d1fd1x.
    - Writes second-derivative components (xx, yy, xy, yx) into d2fd2x.
    - Sets derivatives to 0.0 inside the halo region where the stencil would go out of bounds.
"""

# ---------------- Imports ----------------
from numba import cuda
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2

# ---------------- Compute first derivatives ----------------
@cuda.jit(device=True)
def compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f):
    """
    Compute 4th-order central first derivatives of Field[a] at a single lattice site.

    Usage:
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)

    Parameters:
        d1fd1x: Flat output buffer for first derivatives (coord, field, x, y).
        Field: Flat input buffer for field values (field, x, y).
        a: Field component index.
        x, y: Lattice indices.
        p_i: Device int params (expects p_i[0]=xlen, p_i[1]=ylen, p_i[2]=halo).
        p_f: Device float params (expects p_f[2]=lsx, p_f[3]=lsy).

    Outputs:
        - Writes ∂x Field[a](x, y) into d1fd1x at idx_d1(0, a, x, y, p_i) if stencil is valid, else 0.0.
        - Writes ∂y Field[a](x, y) into d1fd1x at idx_d1(1, a, x, y, p_i) if stencil is valid, else 0.0.
    """
    xlen = p_i[0]; ylen = p_i[1]; halo = p_i[2]
    lsx = p_f[2]; lsy = p_f[3]
    # x derivative
    if x > halo - 1 and x < xlen - halo:
        d1fd1x[idx_d1(0, a, x, y, p_i)] = ((1.0/12.0) * Field[idx_field(a, x-2, y, p_i)] - (2.0/3.0) * Field[idx_field(a, x-1, y, p_i)] + (2.0/3.0) * Field[idx_field(a, x+1, y, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]) / lsx
    else:
        d1fd1x[idx_d1(0, a, x, y, p_i)] = 0.0
    # y derivative
    if y > halo - 1 and y < ylen - halo:
        d1fd1x[idx_d1(1, a, x, y, p_i)] = ((1.0/12.0) * Field[idx_field(a, x, y-2, p_i)] - (2.0/3.0) * Field[idx_field(a, x, y-1, p_i)] + (2.0/3.0) * Field[idx_field(a, x, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x, y+2, p_i)]) / lsy
    else:
        d1fd1x[idx_d1(1, a, x, y, p_i)] = 0.0

# ---------------- Compute second derivatives ----------------
@cuda.jit(device=True)
def compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f):
    """
    Compute 4th-order central second derivatives of Field[a] at a single lattice site.

    Usage:
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)

    Parameters:
        d2fd2x: Flat output buffer for second derivatives (coord1, coord2, field, x, y).
        Field: Flat input buffer for field values (field, x, y).
        a: Field component index.
        x, y: Lattice indices.
        p_i: Device int params (expects p_i[0]=xlen, p_i[1]=ylen, p_i[2]=halo).
        p_f: Device float params (expects p_f[2]=lsx, p_f[3]=lsy, p_f[4]=grid_volume).

    Outputs:
        - Writes ∂xx Field[a](x, y) into d2fd2x at idx_d2(0, 0, a, x, y, p_i) if stencil is valid, else 0.0.
        - Writes ∂yy Field[a](x, y) into d2fd2x at idx_d2(1, 1, a, x, y, p_i) if stencil is valid, else 0.0.
        - Writes ∂xy Field[a](x, y) into d2fd2x at idx_d2(1, 0, a, x, y, p_i) if stencil is valid, else 0.0.
        - Mirrors ∂xy into the ∂yx slot at idx_d2(0, 1, a, x, y, p_i).
    """
    xlen = p_i[0]; ylen = p_i[1]; halo = p_i[2]
    lsx = p_f[2]; lsy = p_f[3]
    grid_volume = p_f[4]
    # xx derivative
    if x > halo - 1 and x < xlen - halo:
        d2fd2x[idx_d2(0, 0, a, x, y, p_i)] = (-(1.0/12.0) * Field[idx_field(a, x-2, y, p_i)] + (4.0/3.0) * Field[idx_field(a, x-1, y, p_i)] - (5.0/2.0) * Field[idx_field(a, x,   y, p_i)] + (4.0/3.0) * Field[idx_field(a, x+1, y, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]) / (lsx * lsx)
    else:
        d2fd2x[idx_d2(0, 0, a, x, y, p_i)] = 0.0
    # yy derivative
    if y > halo - 1 and y < ylen - halo:
        d2fd2x[idx_d2(1, 1, a, x, y, p_i)] = (-(1.0/12.0) * Field[idx_field(a, x, y-2, p_i)] + (4.0/3.0) * Field[idx_field(a, x, y-1, p_i)] - (5.0/2.0) * Field[idx_field(a, x, y,   p_i)] + (4.0/3.0) * Field[idx_field(a, x, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x, y+2, p_i)]) / (lsy * lsy)
    else:
        d2fd2x[idx_d2(1, 1, a, x, y, p_i)] = 0.0
    # xy derivative
    if (x > halo) and (y > halo) and (x < xlen - halo) and (y < ylen - halo):
        diag = (-(1.0/12.0) * Field[idx_field(a, x-2, y-2, p_i)] + (4.0/3.0) * Field[idx_field(a, x-1, y-1, p_i)] - (5.0/2.0) * Field[idx_field(a, x,   y,   p_i)] + (4.0/3.0) * Field[idx_field(a, x+1, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y+2, p_i)]) / (2.0 * grid_volume)

        dxx_half = (-(1.0/12.0) * Field[idx_field(a, x-2, y, p_i)] + (4.0/3.0) * Field[idx_field(a, x-1, y, p_i)] - (5.0/2.0) * Field[idx_field(a, x,   y, p_i)] + (4.0/3.0) * Field[idx_field(a, x+1, y, p_i)] - (1.0/12.0) * Field[idx_field(a, x+2, y, p_i)]) / (2.0 * lsx * lsx)

        dyy_half = (-(1.0/12.0) * Field[idx_field(a, x, y-2, p_i)] + (4.0/3.0) * Field[idx_field(a, x, y-1, p_i)] - (5.0/2.0) * Field[idx_field(a, x, y,   p_i)] + (4.0/3.0) * Field[idx_field(a, x, y+1, p_i)] - (1.0/12.0) * Field[idx_field(a, x, y+2, p_i)]) / (2.0 * lsy * lsy)

        d2fd2x[idx_d2(1, 0, a, x, y, p_i)] = diag - dxx_half - dyy_half
    else:
        d2fd2x[idx_d2(1, 0, a, x, y, p_i)] = 0.0
    # yx derivative
    d2fd2x[idx_d2(0, 1, a, x, y, p_i)] = d2fd2x[idx_d2(1, 0, a, x, y, p_i)]