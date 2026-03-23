"""
CUDA kernels for the anisotropic s+id superconductor theory.

Examples
--------
Use ``compute_energy_kernel`` to evaluate the local energy density.
Use ``do_gradient_step_kernel`` to compute the local energy gradient.
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
        Flattened coordinate grid.
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
def gamma_entry(i, j, a, b, p_f):
    """
    Return one entry of the anisotropy tensor.

    Parameters
    ----------
    i : int
        First coordinate index.
    j : int
        Second coordinate index.
    a : int
        First condensate index.
    b : int
        Second condensate index.
    p_f : device array
        Float parameter array.

    Returns
    -------
    float
        Value of gamma[i, j, a, b].

    Examples
    --------
    Use ``gamma_entry(i, j, a, b, p_f)`` inside device functions that need the anisotropy tensor.
    """
    gamma1 = p_f[14]; gamma2 = p_f[15]; gamma12 = p_f[16]

    if i == 0 and j == 0:
        if a == 0 and b == 0:
            return 2.0 * gamma1
        elif a == 0 and b == 1:
            return -2.0 * gamma12
        elif a == 1 and b == 0:
            return -2.0 * gamma12
        else:
            return 2.0 * gamma2

    if i == 1 and j == 1:
        if a == 0 and b == 0:
            return 2.0 * gamma1
        elif a == 0 and b == 1:
            return 2.0 * gamma12
        elif a == 1 and b == 0:
            return 2.0 * gamma12
        else:
            return 2.0 * gamma2

    return 0.0

@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the local cell energy at a lattice site.

    Parameters
    ----------
    Field : device array
        State field array.
    d1fd1x : device array
        First-derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    float
        Cell-integrated energy at the lattice site.

    Examples
    --------
    Use ``compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)`` inside a device kernel after computing first derivatives.
    """
    kappa = p_f[6]; Q = p_f[7]
    alpha1 = p_f[8]; alpha2 = p_f[9]
    beta1 = p_f[10]; beta2 = p_f[11]; beta3 = p_f[12]; beta4 = p_f[13]
    grid_volume = p_f[4]
    u1 = p_f[17]; u2 = p_f[18]; theta_diff = p_f[19]

    A1 = Field[idx_field(0, x, y, p_i)]; A2 = Field[idx_field(1, x, y, p_i)]
    psi1r = Field[idx_field(2, x, y, p_i)]; psi1i = Field[idx_field(3, x, y, p_i)]
    psi2r = Field[idx_field(4, x, y, p_i)]; psi2i = Field[idx_field(5, x, y, p_i)]

    psi1_sq = psi1r * psi1r + psi1i * psi1i
    psi2_sq = psi2r * psi2r + psi2i * psi2i

    F_mag = 0.5 * kappa * kappa * (d1fd1x[idx_d1(0, 1, x, y, p_i)] - d1fd1x[idx_d1(1, 0, x, y, p_i)]) ** 2

    F_potential = 0.0
    F_potential += alpha1 * psi1_sq + alpha2 * psi2_sq
    F_potential += beta1 * psi1_sq * psi1_sq + beta2 * psi2_sq * psi2_sq + beta3 * psi1_sq * psi2_sq
    F_potential += 2.0 * beta4 * (psi1r * psi1r * psi2r * psi2r + psi1i * psi1i * psi2i * psi2i - psi1i * psi1i * psi2r * psi2r - psi1r * psi1r * psi2i * psi2i + 4.0 * psi1r * psi1i * psi2r * psi2i)

    F_kinetic = 0.0
    number_coordinates = p_i[3]
    for i in range(number_coordinates):
        Ai = A1 if i == 0 else A2
        for j in range(number_coordinates):
            Aj = A1 if j == 0 else A2

            g00 = gamma_entry(i, j, 0, 0, p_f)
            g01 = gamma_entry(i, j, 0, 1, p_f)
            g10 = gamma_entry(i, j, 1, 0, p_f)
            g11 = gamma_entry(i, j, 1, 1, p_f)

            dpsi1r_i = d1fd1x[idx_d1(i, 2, x, y, p_i)]; dpsi1r_j = d1fd1x[idx_d1(j, 2, x, y, p_i)]
            dpsi1i_i = d1fd1x[idx_d1(i, 3, x, y, p_i)]; dpsi1i_j = d1fd1x[idx_d1(j, 3, x, y, p_i)]
            dpsi2r_i = d1fd1x[idx_d1(i, 4, x, y, p_i)]; dpsi2r_j = d1fd1x[idx_d1(j, 4, x, y, p_i)]
            dpsi2i_i = d1fd1x[idx_d1(i, 5, x, y, p_i)]; dpsi2i_j = d1fd1x[idx_d1(j, 5, x, y, p_i)]

            F_kinetic += 0.5 * g00 * (dpsi1r_i * dpsi1r_j + dpsi1i_i * dpsi1i_j + Q * Ai * (psi1r * dpsi1i_j - psi1i * dpsi1r_j) + Q * Aj * (psi1r * dpsi1i_i - psi1i * dpsi1r_i) + Q * Q * Ai * Aj * psi1_sq)
            F_kinetic += 0.5 * g01 * (dpsi1r_i * dpsi2r_j + dpsi1i_i * dpsi2i_j + Q * Ai * (psi1r * dpsi2i_j - psi1i * dpsi2r_j) + Q * Aj * (psi2r * dpsi1i_i - psi2i * dpsi1r_i) + Q * Q * Ai * Aj * (psi1r * psi2r + psi1i * psi2i))
            F_kinetic += 0.5 * g10 * (dpsi2r_i * dpsi1r_j + dpsi2i_i * dpsi1i_j + Q * Ai * (psi2r * dpsi1i_j - psi2i * dpsi1r_j) + Q * Aj * (psi1r * dpsi2i_i - psi1i * dpsi2r_i) + Q * Q * Ai * Aj * (psi2r * psi1r + psi2i * psi1i))
            F_kinetic += 0.5 * g11 * (dpsi2r_i * dpsi2r_j + dpsi2i_i * dpsi2i_j + Q * Ai * (psi2r * dpsi2i_j - psi2i * dpsi2r_j) + Q * Aj * (psi2r * dpsi2i_i - psi2i * dpsi2r_i) + Q * Q * Ai * Aj * psi2_sq)

    energy = F_kinetic + F_mag + F_potential
    energy -= alpha1 * u1 * u1 + alpha2 * u2 * u2 + beta1 * u1 * u1 * u1 * u1 + beta2 * u2 * u2 * u2 * u2 + beta3 * u1 * u1 * u2 * u2 + 2.0 * beta4 * u1 * u1 * u2 * u2 * math.cos(2.0 * theta_diff)
    energy *= grid_volume
    return energy

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the local energy density across the grid.

    Parameters
    ----------
    en : device array
        Output energy density buffer.
    Field : device array
        State field buffer.
    d1fd1x : device array
        First derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local energy density is written in place.

    Examples
    --------
    Launch ``compute_energy_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to evaluate the energy density.
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
    Compute a local magnetic flux density component.

    Parameters
    ----------
    d1fd1x : device array
        First derivative buffer.
    which : int
        Flux component selector.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the cell volume and charge.

    Returns
    -------
    float
        Local magnetic flux density contribution.

    Examples
    --------
    Use ``compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the flux density.
    """
    grid_volume = p_f[4]
    q = p_f[7]
    magneticField = d1fd1x[idx_d1(0, 1, x, y, p_i)] - d1fd1x[idx_d1(1, 0, x, y, p_i)]
    magneticField *= grid_volume / (2.0 * math.pi * q)
    return magneticField

@cuda.jit
def compute_vortex_number_kernel(en, Field, d1fd1x, which, p_i, p_f):
    """
    Compute a selected magnetic flux density component across the grid.

    Parameters
    ----------
    en : device array
        Output density buffer.
    Field : device array
        State field buffer.
    d1fd1x : device array
        First derivative buffer.
    which : int
        Flux component selector.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The selected flux density is written in place.

    Examples
    --------
    Launch ``compute_vortex_number_kernel[grid2d, block2d](en, Field, d1fd1x, which, p_i, p_f)`` to evaluate a flux density component.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[12]
    for a in range(number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)

@cuda.jit
def compute_magnetic_field_kernel(Field, d1fd1x, MagneticFluxDensity, p_i, p_f):
    """
    Compute the magnetic flux density vector field across the grid.

    Parameters
    ----------
    Field : device array
        State field buffer.
    d1fd1x : device array
        First derivative buffer.
    MagneticFluxDensity : device array
        Output magnetic flux density buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The magnetic flux density vector field is written in place.

    Examples
    --------
    Launch ``compute_magnetic_field_kernel[grid2d, block2d](Field, d1fd1x, MagneticFluxDensity, p_i, p_f)`` to evaluate the magnetic field.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[12]
    for a in range(number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    MagneticFluxDensity[idx_field(0, x, y, p_i)] = compute_vortex_density(d1fd1x, 2, x, y, p_i, p_f)

@cuda.jit(device=True)
def compute_norm_higgs1_point(Field, x, y, p_i):
    """
    Compute the squared norm of the first Higgs field at one lattice site.

    Parameters
    ----------
    Field : device array
        State field buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    float
        Squared norm of the first Higgs field.

    Examples
    --------
    Use ``compute_norm_higgs1_point(Field, x, y, p_i)`` inside a CUDA kernel to evaluate ``|psi_1|^2``.
    """
    psi1r = Field[idx_field(2, x, y, p_i)]
    psi1i = Field[idx_field(3, x, y, p_i)]
    return psi1r*psi1r + psi1i*psi1i

@cuda.jit
def compute_norm_higgs1_kernel(en, Field, p_i):
    """
    Compute the squared norm of the first Higgs field across the grid.

    Parameters
    ----------
    en : device array
        Output density buffer.
    Field : device array
        State field buffer.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The squared norm is written in place.

    Examples
    --------
    Launch ``compute_norm_higgs1_kernel[grid2d, block2d](en, Field, p_i)`` to evaluate ``|psi_1|^2``.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs1_point(Field, x, y, p_i)

@cuda.jit(device=True)
def compute_norm_higgs2_point(Field, x, y, p_i):
    """
    Compute the squared norm of the second Higgs field at one lattice site.

    Parameters
    ----------
    Field : device array
        State field buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    float
        Squared norm of the second Higgs field.

    Examples
    --------
    Use ``compute_norm_higgs2_point(Field, x, y, p_i)`` inside a CUDA kernel to evaluate ``|psi_2|^2``.
    """
    psi2r = Field[idx_field(4, x, y, p_i)]
    psi2i = Field[idx_field(5, x, y, p_i)]
    return psi2r*psi2r + psi2i*psi2i

@cuda.jit
def compute_norm_higgs2_kernel(en, Field, p_i):
    """
    Compute the squared norm of the second Higgs field across the grid.

    Parameters
    ----------
    en : device array
        Output density buffer.
    Field : device array
        State field buffer.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The squared norm is written in place.

    Examples
    --------
    Launch ``compute_norm_higgs2_kernel[grid2d, block2d](en, Field, p_i)`` to evaluate ``|psi_2|^2``.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs2_point(Field, x, y, p_i)

@cuda.jit(device=True)
def compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i):
    """
    Compute the local supercurrent vector from gauge field second derivatives.

    Parameters
    ----------
    Supercurrent : device array
        Output supercurrent buffer.
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
        The local supercurrent vector is written in place.

    Examples
    --------
    Use ``compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)`` inside a CUDA kernel to evaluate the local supercurrent.
    """
    number_gauge_fields = p_i[12]
    for a in range(number_gauge_fields):
        Supercurrent[idx_field(a, x, y, p_i)] = 0.0
    Supercurrent[idx_field(0, x, y, p_i)] += d2fd2x[idx_d2(1, 0, 1, x, y, p_i)] - d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    Supercurrent[idx_field(1, x, y, p_i)] += d2fd2x[idx_d2(0, 1, 0, x, y, p_i)] - d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]

@cuda.jit
def compute_supercurrent_kernel(Field, d2fd2x, Supercurrent, p_i, p_f):
    """
    Compute the supercurrent field across the grid.

    Parameters
    ----------
    Field : device array
        State field buffer.
    d2fd2x : device array
        Second derivative buffer.
    Supercurrent : device array
        Output supercurrent buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The supercurrent field is written in place.

    Examples
    --------
    Launch ``compute_supercurrent_kernel[grid2d, block2d](Field, d2fd2x, Supercurrent, p_i, p_f)`` to evaluate the supercurrent.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[12]
    for a in range(number_gauge_fields):
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
    compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)

@cuda.jit(device=True)
def compute_phase_difference_point(Field, x, y, p_i):
    """
    Compute the phase difference at one lattice site.

    Parameters
    ----------
    Field : device array
        State field buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    float
        Phase difference between the condensates.

    Examples
    --------
    Use ``compute_phase_difference_point(Field, x, y, p_i)`` inside a CUDA kernel to evaluate ``cos(theta1-theta2)``.
    """
    psi1r = Field[idx_field(2, x, y, p_i)]; psi1i = Field[idx_field(3, x, y, p_i)]
    psi2r = Field[idx_field(4, x, y, p_i)]; psi2i = Field[idx_field(5, x, y, p_i)]
    theta1 = math.atan2(psi1i, psi1r)
    theta2 = math.atan2(psi2i, psi2r)
    return math.cos(theta1 - theta2)

@cuda.jit
def compute_phase_difference_kernel(en, Field, p_i):
    """
    Compute the phase difference between the condensates across the grid.

    Parameters
    ----------
    en : device array
        Output density buffer.
    Field : device array
        State field buffer.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The phase difference is written in place.

    Examples
    --------
    Launch ``compute_phase_difference_kernel[grid2d, block2d](en, Field, p_i)`` to evaluate ``cos(theta1-theta2)``.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_phase_difference_point(Field, x, y, p_i)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient and update the local velocity.

    Parameters
    ----------
    Velocity : device array
        Velocity buffer to update.
    Field : device array
        State field array.
    EnergyGradient : device array
        Output gradient buffer.
    d1fd1x : device array
        First-derivative buffer.
    d2fd2x : device array
        Second-derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    None
        The local gradient is written into ``EnergyGradient`` and the velocity is updated in ``Velocity``.

    Examples
    --------
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)`` inside a device kernel after computing derivatives.
    """
    kappa = p_f[6]; Q = p_f[7] 
    alpha1 = p_f[8]; alpha2 = p_f[9]
    beta1 = p_f[10]; beta2 = p_f[11]; beta3 = p_f[12]; beta4 = p_f[13]
    time_step = p_f[5]

    A1 = Field[idx_field(0, x, y, p_i)]; A2 = Field[idx_field(1, x, y, p_i)]
    psi1r = Field[idx_field(2, x, y, p_i)]; psi1i = Field[idx_field(3, x, y, p_i)]
    psi2r = Field[idx_field(4, x, y, p_i)]; psi2i = Field[idx_field(5, x, y, p_i)]

    EG0 = 0.0; EG1 = 0.0; EG2 = 0.0; EG3 = 0.0; EG4 = 0.0; EG5 = 0.0

    number_coordinates = p_i[3]

    for j in range(number_coordinates):
        for k in range(number_coordinates):
            Ak = A1 if k == 0 else A2

            g00 = gamma_entry(k, j, 0, 0, p_f)
            g01 = gamma_entry(k, j, 0, 1, p_f)
            g10 = gamma_entry(k, j, 1, 0, p_f)
            g11 = gamma_entry(k, j, 1, 1, p_f)

            val = kappa * kappa * (d2fd2x[idx_d2(j, k, k, x, y, p_i)] - d2fd2x[idx_d2(k, k, j, x, y, p_i)])
            val -= Q * g00 * (psi1i * d1fd1x[idx_d1(k, 2, x, y, p_i)] - psi1r * d1fd1x[idx_d1(k, 3, x, y, p_i)] - Q * Ak * (psi1r * psi1r + psi1i * psi1i))
            val -= Q * g01 * (psi2i * d1fd1x[idx_d1(k, 2, x, y, p_i)] - psi2r * d1fd1x[idx_d1(k, 3, x, y, p_i)] - Q * Ak * (psi1r * psi2r + psi1i * psi2i))
            val -= Q * g10 * (psi1i * d1fd1x[idx_d1(k, 4, x, y, p_i)] - psi1r * d1fd1x[idx_d1(k, 5, x, y, p_i)] - Q * Ak * (psi2r * psi1r + psi2i * psi1i))
            val -= Q * g11 * (psi2i * d1fd1x[idx_d1(k, 4, x, y, p_i)] - psi2r * d1fd1x[idx_d1(k, 5, x, y, p_i)] - Q * Ak * (psi2r * psi2r + psi2i * psi2i))

            if j == 0:
                EG0 += val
            else:
                EG1 += val

    EG2 += 2.0 * (2.0 * alpha1 * psi1r + 4.0 * beta1 * psi1r * (psi1r * psi1r + psi1i * psi1i) + 2.0 * beta3 * psi1r * (psi2r * psi2r + psi2i * psi2i) + 2.0 * beta4 * (2.0 * psi1r * (psi2r * psi2r - psi2i * psi2i) + 4.0 * psi1i * psi2r * psi2i))
    EG3 += 2.0 * (2.0 * alpha1 * psi1i + 4.0 * beta1 * psi1i * (psi1r * psi1r + psi1i * psi1i) + 2.0 * beta3 * psi1i * (psi2r * psi2r + psi2i * psi2i) + 2.0 * beta4 * (-2.0 * psi1i * (psi2r * psi2r - psi2i * psi2i) + 4.0 * psi1r * psi2r * psi2i))
    EG4 += 2.0 * (2.0 * alpha2 * psi2r + 4.0 * beta2 * psi2r * (psi2r * psi2r + psi2i * psi2i) + 2.0 * beta3 * psi2r * (psi1r * psi1r + psi1i * psi1i) + 2.0 * beta4 * (2.0 * psi2r * (psi1r * psi1r - psi1i * psi1i) + 4.0 * psi1r * psi1i * psi2i))
    EG5 += 2.0 * (2.0 * alpha2 * psi2i + 4.0 * beta2 * psi2i * (psi2r * psi2r + psi2i * psi2i) + 2.0 * beta3 * psi2i * (psi1r * psi1r + psi1i * psi1i) + 2.0 * beta4 * (-2.0 * psi2i * (psi1r * psi1r - psi1i * psi1i) + 4.0 * psi1r * psi1i * psi2r))

    for i in range(number_coordinates):
        Ai = A1 if i == 0 else A2
        for j in range(number_coordinates):
            Aj = A1 if j == 0 else A2
            gij00 = gamma_entry(i, j, 0, 0, p_f)
            gij01 = gamma_entry(i, j, 0, 1, p_f)
            gij10 = gamma_entry(i, j, 1, 0, p_f)
            gij11 = gamma_entry(i, j, 1, 1, p_f)
            dAijdx = d1fd1x[idx_d1(i, j, x, y, p_i)]

            EG2 -= gij00 * (d2fd2x[idx_d2(i, j, 2, x, y, p_i)] - Q * Q * Ai * Aj * psi1r - Q * (psi1i * dAijdx + Ai * d1fd1x[idx_d1(j, 3, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 3, x, y, p_i)]))
            EG2 -= gij01 * (d2fd2x[idx_d2(i, j, 4, x, y, p_i)] - Q * Q * Ai * Aj * psi2r - Q * (psi2i * dAijdx + Ai * d1fd1x[idx_d1(j, 5, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 5, x, y, p_i)]))

            EG3 -= gij00 * (d2fd2x[idx_d2(i, j, 3, x, y, p_i)] - Q * Q * Ai * Aj * psi1i + Q * (psi1r * dAijdx + Ai * d1fd1x[idx_d1(j, 2, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 2, x, y, p_i)]))
            EG3 -= gij01 * (d2fd2x[idx_d2(i, j, 5, x, y, p_i)] - Q * Q * Ai * Aj * psi2i + Q * (psi2r * dAijdx + Ai * d1fd1x[idx_d1(j, 4, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 4, x, y, p_i)]))

            EG4 -= gij10 * (d2fd2x[idx_d2(i, j, 2, x, y, p_i)] - Q * Q * Ai * Aj * psi1r - Q * (psi1i * dAijdx + Ai * d1fd1x[idx_d1(j, 3, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 3, x, y, p_i)]))
            EG4 -= gij11 * (d2fd2x[idx_d2(i, j, 4, x, y, p_i)] - Q * Q * Ai * Aj * psi2r - Q * (psi2i * dAijdx + Ai * d1fd1x[idx_d1(j, 5, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 5, x, y, p_i)]))

            EG5 -= gij10 * (d2fd2x[idx_d2(i, j, 3, x, y, p_i)] - Q * Q * Ai * Aj * psi1i + Q * (psi1r * dAijdx + Ai * d1fd1x[idx_d1(j, 2, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 2, x, y, p_i)]))
            EG5 -= gij11 * (d2fd2x[idx_d2(i, j, 5, x, y, p_i)] - Q * Q * Ai * Aj * psi2i + Q * (psi2r * dAijdx + Ai * d1fd1x[idx_d1(j, 4, x, y, p_i)] + Aj * d1fd1x[idx_d1(i, 4, x, y, p_i)]))

    EnergyGradient[idx_field(0, x, y, p_i)] = EG0
    EnergyGradient[idx_field(1, x, y, p_i)] = EG1
    EnergyGradient[idx_field(2, x, y, p_i)] = EG2
    EnergyGradient[idx_field(3, x, y, p_i)] = EG3
    EnergyGradient[idx_field(4, x, y, p_i)] = EG4
    EnergyGradient[idx_field(5, x, y, p_i)] = EG5
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)