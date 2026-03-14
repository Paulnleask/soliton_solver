"""
Define CUDA kernels for the Ginzburg-Landau superconductor theory.

Examples
--------
Use ``compute_energy_kernel`` to evaluate the energy density.
Use ``compute_supercurrent_kernel`` to evaluate the supercurrent.
"""
import math
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds, launch_2d
from soliton_solver.core.integrator import make_do_gradient_step_kernel

@cuda.jit
def create_grid_kernel(grid, p_i, p_f):
    """
    Populate the physical coordinate grid.

    Parameters
    ----------
    grid : device array
        Flattened coordinate array.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the lattice spacings.

    Returns
    -------
    None
        The coordinate grid is written in place.

    Examples
    --------
    Launch ``create_grid_kernel[grid2d, block2d](grid, p_i, p_f)`` to build the coordinate grid.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    lsx = p_f[2]; lsy = p_f[3]
    grid[idx_field(0, x, y, p_i)] = lsx * float(x)
    grid[idx_field(1, x, y, p_i)] = lsy * float(y)

@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the cell integrated energy at one lattice site.

    Parameters
    ----------
    Field : device array
        Flattened state array.
    d1fd1x : device array
        First derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    float
        Cell integrated energy at the lattice site.

    Examples
    --------
    Use ``e = compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local energy.
    """
    q = p_f[6]
    lam = p_f[7]
    u1 = p_f[9]
    grid_volume = p_f[4]

    A1 = Field[idx_field(0, x, y, p_i)]
    A2 = Field[idx_field(1, x, y, p_i)]
    psi1 = Field[idx_field(2, x, y, p_i)]
    psi2 = Field[idx_field(3, x, y, p_i)]

    dpsi1dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]
    dpsi1dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dpsi2dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]
    dpsi2dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]

    dA1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]
    dA1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dA2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]
    dA2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]

    psi_sq = psi1*psi1 + psi2*psi2
    A_sq = A1*A1 + A2*A2

    energy = 0.0
    energy += 0.5*(dpsi1dx*dpsi1dx + dpsi1dy*dpsi1dy + dpsi2dx*dpsi2dx + dpsi2dy*dpsi2dy)
    energy += q*A1*(psi1*dpsi2dx - psi2*dpsi1dx)
    energy += q*A2*(psi1*dpsi2dy - psi2*dpsi1dy)
    energy += 0.5*q*q*A_sq*psi_sq
    energy += lam/8.0*(u1*u1 - psi_sq)*(u1*u1 - psi_sq)

    curlA = dA2dx - dA1dy
    energy += 0.5*curlA*curlA

    energy *= grid_volume
    return energy

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the energy density across the grid.

    Parameters
    ----------
    en : device array
        Output energy array.
    Field : device array
        Flattened state array.
    d1fd1x : device array
        First derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local energy is written into ``en``.

    Examples
    --------
    Launch ``compute_energy_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to compute the energy density.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)

@cuda.jit(device=True)
def compute_vortex_density(d1fd1x, which, x, y, p_i, p_f):
    """
    Compute the local magnetic flux density.

    Parameters
    ----------
    d1fd1x : device array
        First derivative buffer.
    which : int
        Component selector.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    float
        Local magnetic flux density.

    Examples
    --------
    Use ``rho = compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the flux density.
    """
    grid_volume = p_f[4]
    q = p_f[6]
    magneticField = d1fd1x[idx_d1(0, 1, x, y, p_i)] - d1fd1x[idx_d1(1, 0, x, y, p_i)]
    magneticField *= grid_volume / (2.0 * math.pi * q)
    return magneticField

@cuda.jit
def compute_vortex_number_kernel(en, Field, d1fd1x, which, p_i, p_f):
    """
    Compute the magnetic flux density across the grid.

    Parameters
    ----------
    en : device array
        Output density array.
    Field : device array
        Flattened state array.
    d1fd1x : device array
        First derivative buffer.
    which : int
        Component selector.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local magnetic flux density is written into ``en``.

    Examples
    --------
    Launch ``compute_vortex_number_kernel[grid2d, block2d](en, Field, d1fd1x, which, p_i, p_f)`` to compute the flux density.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[11]
    for a in range(number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)

@cuda.jit(device=True)
def compute_norm_higgs_point(Field, x, y, p_i):
    """
    Compute the Higgs norm at one lattice site.

    Parameters
    ----------
    Field : device array
        Flattened state array.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    float
        Squared Higgs norm at the lattice site.

    Examples
    --------
    Use ``norm = compute_norm_higgs_point(Field, x, y, p_i)`` inside a CUDA kernel to evaluate ``|psi|^2``.
    """
    psi1 = Field[idx_field(2, x, y, p_i)]
    psi2 = Field[idx_field(3, x, y, p_i)]
    return psi1*psi1 + psi2*psi2

@cuda.jit
def compute_norm_higgs_kernel(en, Field, p_i):
    """
    Compute the Higgs norm across the grid.

    Parameters
    ----------
    en : device array
        Output norm array.
    Field : device array
        Flattened state array.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The Higgs norm is written into ``en``.

    Examples
    --------
    Launch ``compute_norm_higgs_kernel[grid2d, block2d](en, Field, p_i)`` to compute ``|psi|^2`` across the grid.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs_point(Field, x, y, p_i)

@cuda.jit(device=True)
def compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i):
    """
    Compute the local supercurrent from second derivatives of the gauge field.

    Parameters
    ----------
    Supercurrent : device array
        Output supercurrent array.
    d2fd2x : device array
        Second derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The supercurrent is written into ``Supercurrent``.

    Examples
    --------
    Use ``compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)`` inside a CUDA kernel to evaluate the local supercurrent.
    """
    number_gauge_fields = p_i[11]
    for a in range(number_gauge_fields):
        Supercurrent[idx_field(a, x, y, p_i)] = 0.0
    Supercurrent[idx_field(0, x, y, p_i)] += d2fd2x[idx_d2(1, 0, 1, x, y, p_i)] - d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    Supercurrent[idx_field(1, x, y, p_i)] += d2fd2x[idx_d2(0, 1, 0, x, y, p_i)] - d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]

@cuda.jit
def compute_supercurrent_kernel(Field, d2fd2x, Supercurrent, p_i, p_f):
    """
    Compute the supercurrent across the grid.

    Parameters
    ----------
    Field : device array
        Flattened state array.
    d2fd2x : device array
        Second derivative buffer.
    Supercurrent : device array
        Output supercurrent array.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The supercurrent is written into ``Supercurrent``.

    Examples
    --------
    Launch ``compute_supercurrent_kernel[grid2d, block2d](Field, d2fd2x, Supercurrent, p_i, p_f)`` to compute the supercurrent.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[11]
    for a in range(number_gauge_fields):
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
    compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient and update the velocity.

    Parameters
    ----------
    Velocity : device array
        Velocity field buffer.
    Field : device array
        Flattened state array.
    EnergyGradient : device array
        Output energy gradient array.
    d1fd1x : device array
        First derivative buffer.
    d2fd2x : device array
        Second derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local energy gradient and velocity are written in place.

    Examples
    --------
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local gradient step.
    """
    q = p_f[6]
    lam = p_f[7]
    u1 = p_f[8]
    time_step = p_f[5]

    A1 = Field[idx_field(0, x, y, p_i)]
    A2 = Field[idx_field(1, x, y, p_i)]
    psi1 = Field[idx_field(2, x, y, p_i)]
    psi2 = Field[idx_field(3, x, y, p_i)]

    psi_sq = psi1*psi1 + psi2*psi2

    dA1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]
    dA1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dA2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]
    dA2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dpsi1dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]
    dpsi1dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dpsi2dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]
    dpsi2dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]

    d2A1d2y = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    d2A2d2x = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]
    d2psi1d2x = d2fd2x[idx_d2(0, 0, 2, x, y, p_i)]
    d2psi1d2y = d2fd2x[idx_d2(1, 1, 2, x, y, p_i)]
    d2psi2d2x = d2fd2x[idx_d2(0, 0, 3, x, y, p_i)]
    d2psi2d2y = d2fd2x[idx_d2(1, 1, 3, x, y, p_i)]
    d2A1dxdy = d2fd2x[idx_d2(1, 0, 0, x, y, p_i)]
    d2A2dxdy = d2fd2x[idx_d2(1, 0, 1, x, y, p_i)]

    g0 = (-d2A1d2y + d2A2dxdy + q*(psi1*dpsi2dx - psi2*dpsi1dx) + q*q*A1*psi_sq)
    g1 = (-d2A2d2x + d2A1dxdy + q*(psi1*dpsi2dy - psi2*dpsi1dy) + q*q*A2*psi_sq)
    g2 = (-d2psi1d2x - d2psi1d2y + 2.0*q*(A1*dpsi2dx + A2*dpsi2dy) + q*psi2*(dA1dx + dA2dy) + q*q*(A1*A1 + A2*A2)*psi1 - lam/2.0*(u1*u1 - psi_sq)*psi1)
    g3 = (-d2psi2d2x - d2psi2d2y - 2.0*q*(A1*dpsi1dx + A2*dpsi1dy) - q*psi1*(dA1dx + dA2dy) + q*q*(A1*A1 + A2*A2)*psi2 - lam/2.0*(u1*u1 - psi_sq)*psi2)

    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1
    EnergyGradient[idx_field(2, x, y, p_i)] = g2
    EnergyGradient[idx_field(3, x, y, p_i)] = g3

    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)