"""
CUDA kernels for the Bose-Einstein condensate theory.

Examples
--------
Use ``create_grid_kernel`` to build the coordinate grid on the device.
Use ``compute_energy_kernel`` to evaluate the energy density.
"""
import math
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds, launch_2d
from soliton_solver.core.integrator import make_do_gradient_step_kernel

@cuda.jit
def create_grid_kernel(grid, p_i, p_f):
    """
    Populate the physical coordinate grid on the device.

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
        The coordinate values are written into ``grid``.

    Examples
    --------
    Launch ``create_grid_kernel[grid2d, block2d](grid, p_i, p_f)`` to initialize the coordinate grid.
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
    Compute the local cell energy at one lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array.
    d1fd1x : device array
        Flattened first derivative buffer.
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
    Use ``compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local energy.
    """
    lsx = p_f[2]; lsy = p_f[3]
    xlen = p_i[0]; ylen = p_i[1]
    xcent = (0.0 + lsx * float(xlen - 1)) / 2.0
    ycent = (0.0 + lsy * float(ylen - 1)) / 2.0
    xpos = lsx * float(x) - xcent
    ypos = lsy * float(y) - ycent
    r1 = math.sqrt(xpos * xpos + ypos * ypos)
    
    beta = p_f[6]
    omega_rot = p_f[8]
    grid_volume = p_f[4]

    psir = Field[idx_field(0, x, y, p_i)]
    psii = Field[idx_field(1, x, y, p_i)]

    dpsirdx = d1fd1x[idx_d1(0, 0, x, y, p_i)]
    dpsirdy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dpsiidx = d1fd1x[idx_d1(0, 1, x, y, p_i)]
    dpsiidy = d1fd1x[idx_d1(1, 1, x, y, p_i)]

    psi_sq = psir*psir + psii*psii

    energy = 0.0
    energy += 0.5*(dpsirdx*dpsirdx + dpsirdy*dpsirdy + dpsiidx*dpsiidx + dpsiidy*dpsiidy)
    energy += 0.5 * r1 * r1 * psi_sq
    energy += 0.5 * beta * psi_sq * psi_sq
    energy -= omega_rot * (xpos * (psir * dpsiidy - psii * dpsirdy) - ypos * (psir * dpsiidx - psii * dpsirdx))
    energy *= grid_volume
    return energy

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the energy density across the grid.

    Parameters
    ----------
    en : device array
        Output array for the local energy values.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        First derivative buffer written in place.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local energies are written into ``en``.

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
def compute_norm_higgs_point(Field, x, y, p_i):
    """
    Compute the Higgs norm at one lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    float
        Value of ``|psi|^2`` at the lattice site.

    Examples
    --------
    Use ``compute_norm_higgs_point(Field, x, y, p_i)`` inside a CUDA kernel to evaluate the Higgs norm.
    """
    psir = Field[idx_field(0, x, y, p_i)]
    psii = Field[idx_field(1, x, y, p_i)]
    return psir*psir + psii*psii

@cuda.jit
def compute_norm_higgs_kernel(en, Field, p_i):
    """
    Compute the Higgs norm across the grid.

    Parameters
    ----------
    en : device array
        Output array for the Higgs norm.
    Field : device array
        Flattened field array.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The Higgs norm is written into ``en``.

    Examples
    --------
    Launch ``compute_norm_higgs_kernel[grid2d, block2d](en, Field, p_i)`` to compute ``|psi|^2`` on the grid.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs_point(Field, x, y, p_i)

@cuda.jit
def compute_norm_kernel(en, Field, p_i, p_f):
    """
    Compute the integrated Higgs norm density across the grid.

    Parameters
    ----------
    en : device array
        Output array for the weighted Higgs norm.
    Field : device array
        Flattened field array.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the grid volume.

    Returns
    -------
    None
        The weighted Higgs norm is written into ``en``.

    Examples
    --------
    Launch ``compute_norm_kernel[grid2d, block2d](en, Field, p_i, p_f)`` to compute the weighted norm density.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    grid_volume = p_f[4]
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs_point(Field, x, y, p_i) * grid_volume

@cuda.jit
def do_norm_kernel(Field, norm, p_i):
    """
    Normalize the Higgs field by a global norm.

    Parameters
    ----------
    Field : device array
        Flattened field array.
    norm : float
        Global normalization factor.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The Higgs field components are rescaled in place.

    Examples
    --------
    Launch ``do_norm_kernel[grid2d, block2d](Field, norm, p_i)`` to normalize the Higgs field.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    Field[idx_field(0, x, y, p_i)] /= math.sqrt(norm)
    Field[idx_field(1, x, y, p_i)] /= math.sqrt(norm)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient and velocity update at one lattice site.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Flattened field array.
    EnergyGradient : device array
        Output array for the energy gradient.
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
        The local gradient and velocity are written in place.

    Examples
    --------
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local gradient step.
    """
    lsx = p_f[2]; lsy = p_f[3]
    xlen = p_i[0]; ylen = p_i[1]
    xcent = (0.0 + lsx * float(xlen - 1)) / 2.0
    ycent = (0.0 + lsy * float(ylen - 1)) / 2.0
    xpos = lsx * float(x) - xcent
    ypos = lsy * float(y) - ycent
    r1 = math.sqrt(xpos * xpos + ypos * ypos)
    
    beta = p_f[6]
    omega_rot = p_f[8]
    time_step = p_f[5]

    psir = Field[idx_field(0, x, y, p_i)]
    psii = Field[idx_field(1, x, y, p_i)]

    dpsirdx = d1fd1x[idx_d1(0, 0, x, y, p_i)]
    dpsirdy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dpsiidx = d1fd1x[idx_d1(0, 1, x, y, p_i)]
    dpsiidy = d1fd1x[idx_d1(1, 1, x, y, p_i)]

    d2psird2x = d2fd2x[idx_d2(0, 0, 0, x, y, p_i)]
    d2psird2y = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    d2psiid2x = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]
    d2psiid2y = d2fd2x[idx_d2(1, 1, 1, x, y, p_i)]

    psi_sq = psir*psir + psii*psii

    g0 = psir * r1 * r1 + 2.0 * beta * psir * psi_sq - d2psird2x - d2psird2y
    g1 = psii * r1 * r1 + 2.0 * beta * psii * psi_sq - d2psiid2x - d2psiid2y
    g0 -= 2.0 * omega_rot * (xpos * dpsiidy - ypos * dpsiidx)
    g1 += 2.0 * omega_rot * (xpos * dpsirdy - ypos * dpsirdx)

    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1

    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)