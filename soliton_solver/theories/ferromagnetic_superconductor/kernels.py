"""
Core CUDA kernels for the ferromagnetic superconductor theory.

Examples
--------
Use ``compute_energy_kernel`` to evaluate the energy density.
Use ``do_gradient_step_kernel`` to evaluate the local energy gradient.
"""
import math
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds, launch_2d
from soliton_solver.core.integrator import make_do_gradient_step_kernel
from soliton_solver.core.integrator import make_do_rk4_kernel

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
        The coordinate grid is written in place.

    Examples
    --------
    Launch ``create_grid_kernel[grid2d, block2d](grid, p_i, p_f)`` to populate the coordinate grid.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    lsx = p_f[2]; lsy = p_f[3]
    grid[idx_field(0, x, y, p_i)] = lsx * float(x)
    grid[idx_field(1, x, y, p_i)] = lsy * float(y)

@cuda.jit(device=True)
def compute_norm_magnetization(Field, x, y, p_i, p_f):
    """
    Normalize the magnetization at a lattice site to fixed length.

    Parameters
    ----------
    Field : device array
        State field array.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the target magnetization length.

    Returns
    -------
    None
        The magnetization components are rescaled in place.

    Examples
    --------
    Use ``compute_norm_magnetization(Field, x, y, p_i, p_f)`` inside a device kernel to enforce the magnetization norm.
    """
    number_magnetization_fields = p_i[10]
    M0 = p_f[17]
    s = 0.0
    for a in range(number_magnetization_fields):
        v = Field[idx_field(a, x, y, p_i)]
        s += v * v
    s = math.sqrt(s)
    if s == 0.0:
        return
    for a in range(number_magnetization_fields):
        Field[idx_field(a, x, y, p_i)] *= (M0 / s)

@cuda.jit(device=True)
def project_orthogonal_magnetization(func, Field, x, y, p_i, p_f):
    """
    Project a vector field orthogonally to the magnetization direction at a lattice site.

    Parameters
    ----------
    func : device array
        Field to project.
    Field : device array
        State field array containing the magnetization.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the magnetization scale.

    Returns
    -------
    None
        The projected field is written in place.

    Examples
    --------
    Use ``project_orthogonal_magnetization(func, Field, x, y, p_i, p_f)`` inside a device kernel to enforce the tangent-space constraint.
    """
    number_magnetization_fields = p_i[10]
    M0 = p_f[17]
    lm = 0.0
    for a in range(number_magnetization_fields):
        lm += func[idx_field(a, x, y, p_i)] * Field[idx_field(a, x, y, p_i)] / M0
    for a in range(number_magnetization_fields):
        func[idx_field(a, x, y, p_i)] -= lm * Field[idx_field(a, x, y, p_i)] / M0

do_rk4_kernel = make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization)

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
    ha = p_f[7]; hb = p_f[8]; q = p_f[6]
    alpha = p_f[14]; beta = p_f[15]; gamma = p_f[16]
    eta1 = p_f[9]; eta2 = p_f[10]
    grid_volume = p_f[4]
    m1 = Field[idx_field(0, x, y, p_i)]; m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    A1 = Field[idx_field(3, x, y, p_i)]; A2 = Field[idx_field(4, x, y, p_i)]; A3 = Field[idx_field(5, x, y, p_i)]
    psi1 = Field[idx_field(6, x, y, p_i)]; psi2 = Field[idx_field(7, x, y, p_i)]
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]; dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]; dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dA1dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]; dA1dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]; dA2dx = d1fd1x[idx_d1(0, 4, x, y, p_i)]; dA2dy = d1fd1x[idx_d1(1, 4, x, y, p_i)]; dA3dx = d1fd1x[idx_d1(0, 5, x, y, p_i)]; dA3dy = d1fd1x[idx_d1(1, 5, x, y, p_i)]
    dpsi1dx = d1fd1x[idx_d1(0, 6, x, y, p_i)]; dpsi1dy = d1fd1x[idx_d1(1, 6, x, y, p_i)]; dpsi2dx = d1fd1x[idx_d1(0, 7, x, y, p_i)]; dpsi2dy = d1fd1x[idx_d1(1, 7, x, y, p_i)]
    psi_sq = psi1*psi1 + psi2*psi2
    m_sq = m1*m1 + m2*m2 + m3*m3
    A_sq = A1*A1 + A2*A2 + A3*A3
    energy = 0.0
    energy += ha/2.0 * psi_sq
    energy += hb/4.0 * psi_sq * psi_sq
    energy += 0.5*dpsi1dx*dpsi1dx + 0.5*dpsi1dy*dpsi1dy + 0.5*dpsi2dx*dpsi2dx + 0.5*dpsi2dy*dpsi2dy
    energy += 0.5*q*q*A_sq * psi_sq
    energy += q*A1*(psi1*dpsi2dx - psi2*dpsi1dx) + q*A2*(psi1*dpsi2dy - psi2*dpsi1dy)
    energy += 0.5*dA2dx*dA2dx + 0.5*dA1dy*dA1dy - dA2dx*dA1dy + 0.5*dA3dx*dA3dx + 0.5*dA3dy*dA3dy
    energy += alpha/2.0*m_sq + beta/4.0*m_sq*m_sq
    energy += gamma*gamma/2.0*(dm1dx*dm1dx + dm2dx*dm2dx + dm3dx*dm3dx + dm1dy*dm1dy + dm2dy*dm2dy + dm3dy*dm3dy)
    energy += m2*dA3dx - m1*dA3dy - m3*dA2dx + m3*dA1dy
    energy += eta1 * m_sq * psi_sq
    energy += eta2 * (dm1dx*dm1dx + dm2dx*dm2dx + dm3dx*dm3dx + dm1dy*dm1dy + dm2dy*dm2dy + dm3dy*dm3dy) * psi_sq
    denom = (16.0*eta1*eta1 - 4.0*hb*beta)
    energy -= ((alpha*alpha*hb + ha*ha*beta - 4.0*ha*alpha*eta1) / denom)
    energy *= grid_volume
    return energy

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per-site energy contributions across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field for the local energy.
    Field : device array
        State field array.
    d1fd1x : device array
        First-derivative buffer.
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
def compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the local skyrmion density at a lattice site.

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
        Float parameter array.

    Returns
    -------
    float
        Local skyrmion density contribution.

    Examples
    --------
    Use ``compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)`` inside a device kernel after computing magnetization derivatives.
    """
    M0 = p_f[17]
    grid_volume = p_f[4]
    m1 = Field[idx_field(0, x, y, p_i)];m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    cx0 = dm2dx*dm3dy - dm3dx*dm2dy; cx1 = dm3dx*dm1dy - dm1dx*dm3dy; cx2 = dm1dx*dm2dy - dm2dx*dm1dy
    charge = m1*cx0 + m2*cx1 + m3*cx2
    return charge * (grid_volume / (4.0 * math.pi * M0 * M0 * M0))

@cuda.jit
def compute_skyrmion_number_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per-site skyrmion density across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field for the skyrmion density.
    Field : device array
        State field array.
    d1fd1x : device array
        First-derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local skyrmion density is written into ``en``.

    Examples
    --------
    Launch ``compute_skyrmion_number_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to evaluate the skyrmion density.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)

@cuda.jit(device=True)
def compute_vortex_density(d1fd1x, which, x, y, p_i, p_f):
    """
    Compute a local magnetic flux density component from gauge-field derivatives.

    Parameters
    ----------
    d1fd1x : device array
        First-derivative buffer.
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
        Local magnetic flux density contribution.

    Examples
    --------
    Use ``compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)`` inside a device kernel after computing gauge derivatives.
    """
    grid_volume = p_f[4]
    q = p_f[6]
    magneticField = 0.0
    if which == 0:
        magneticField += d1fd1x[idx_d1(1, 5, x, y, p_i)]
    elif which == 1:
        magneticField -= d1fd1x[idx_d1(0, 5, x, y, p_i)]
    else:
        magneticField += d1fd1x[idx_d1(0, 4, x, y, p_i)] - d1fd1x[idx_d1(1, 3, x, y, p_i)]
    magneticField *= grid_volume / (2.0 * math.pi * q)
    return magneticField

@cuda.jit
def compute_vortex_number_kernel(en, Field, d1fd1x, which, p_i, p_f):
    """
    Compute the per-site magnetic flux density for a selected component.

    Parameters
    ----------
    en : device array
        Output scalar field for the flux density.
    Field : device array
        State field array.
    d1fd1x : device array
        First-derivative buffer.
    which : int
        Component selector.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The selected magnetic flux density is written into ``en``.

    Examples
    --------
    Launch ``compute_vortex_number_kernel[grid2d, block2d](en, Field, d1fd1x, which, p_i, p_f)`` to evaluate one flux component.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    number_gauge_fields = p_i[12]
    for a in range(number_magnetization_fields, number_magnetization_fields + number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)

@cuda.jit
def compute_magnetic_field_kernel(Field, d1fd1x, MagneticFluxDensity, p_i, p_f):
    """
    Compute the magnetic flux density vector field across the grid.

    Parameters
    ----------
    Field : device array
        State field array.
    d1fd1x : device array
        First-derivative buffer.
    MagneticFluxDensity : device array
        Output array for the magnetic flux density vector.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The magnetic flux density vector is written into ``MagneticFluxDensity``.

    Examples
    --------
    Launch ``compute_magnetic_field_kernel[grid2d, block2d](Field, d1fd1x, MagneticFluxDensity, p_i, p_f)`` to evaluate the magnetic field.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    number_gauge_fields = p_i[12]
    for a in range(number_magnetization_fields, number_magnetization_fields + number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    for a in range(number_magnetization_fields):
        MagneticFluxDensity[idx_field(a, x, y, p_i)] = compute_vortex_density(d1fd1x, a, x, y, p_i, p_f)

@cuda.jit(device=True)
def compute_norm_higgs_point(Field, x, y, p_i):
    """
    Compute the Higgs norm at a lattice site.

    Parameters
    ----------
    Field : device array
        State field array.
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
    Use ``compute_norm_higgs_point(Field, x, y, p_i)`` inside a device kernel to evaluate the Higgs norm.
    """
    psi1 = Field[idx_field(6, x, y, p_i)]
    psi2 = Field[idx_field(7, x, y, p_i)]
    return psi1*psi1 + psi2*psi2

@cuda.jit
def compute_norm_higgs_kernel(en, Field, p_i):
    """
    Compute the Higgs norm across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field for the Higgs norm.
    Field : device array
        State field array.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The Higgs norm is written into ``en``.

    Examples
    --------
    Launch ``compute_norm_higgs_kernel[grid2d, block2d](en, Field, p_i)`` to evaluate ``|psi|^2``.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs_point(Field, x, y, p_i)

@cuda.jit(device=True)
def compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i):
    """
    Compute the local supercurrent vector from second derivatives of the gauge fields.

    Parameters
    ----------
    Supercurrent : device array
        Output array for the supercurrent.
    d2fd2x : device array
        Second-derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.

    Returns
    -------
    None
        The local supercurrent is written into ``Supercurrent``.

    Examples
    --------
    Use ``compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)`` inside a device kernel after computing gauge second derivatives.
    """
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        Supercurrent[idx_field(a, x, y, p_i)] = 0.0
    Supercurrent[idx_field(0, x, y, p_i)] += d2fd2x[idx_d2(1, 0, 4, x, y, p_i)] - d2fd2x[idx_d2(1, 1, 3, x, y, p_i)]
    Supercurrent[idx_field(1, x, y, p_i)] += d2fd2x[idx_d2(0, 1, 3, x, y, p_i)] - d2fd2x[idx_d2(0, 0, 4, x, y, p_i)]
    Supercurrent[idx_field(2, x, y, p_i)] -= d2fd2x[idx_d2(0, 0, 5, x, y, p_i)] + d2fd2x[idx_d2(1, 1, 5, x, y, p_i)]

@cuda.jit
def compute_supercurrent_kernel(Field, d2fd2x, Supercurrent, p_i, p_f):
    """
    Compute the supercurrent across the grid.

    Parameters
    ----------
    Field : device array
        State field array.
    d2fd2x : device array
        Second-derivative buffer.
    Supercurrent : device array
        Output array for the supercurrent.
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
    Launch ``compute_supercurrent_kernel[grid2d, block2d](Field, d2fd2x, Supercurrent, p_i, p_f)`` to evaluate the supercurrent.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    number_gauge_fields = p_i[12]
    for a in range(number_magnetization_fields, number_magnetization_fields + number_gauge_fields):
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
    compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient and update the velocity descent direction.

    Parameters
    ----------
    Velocity : device array
        Velocity field buffer.
    Field : device array
        State field array.
    EnergyGradient : device array
        Output array for the energy gradient.
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
        Float parameter array.

    Returns
    -------
    None
        The local energy gradient and velocity are written in place.

    Examples
    --------
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)`` inside a device kernel after computing derivatives.
    """
    time_step = p_f[5]
    alpha = p_f[14]; beta = p_f[15]; gamma = p_f[16]
    ha = p_f[7]; hb = p_f[8]; q = p_f[6]
    eta1 = p_f[9]; eta2 = p_f[10]
    m1 = Field[idx_field(0, x, y, p_i)]
    m2 = Field[idx_field(1, x, y, p_i)]
    m3 = Field[idx_field(2, x, y, p_i)]
    A1 = Field[idx_field(3, x, y, p_i)]
    A2 = Field[idx_field(4, x, y, p_i)]
    A3 = Field[idx_field(5, x, y, p_i)]
    psi1 = Field[idx_field(6, x, y, p_i)]
    psi2 = Field[idx_field(7, x, y, p_i)]
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dA1dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]; dA1dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]
    dA2dx = d1fd1x[idx_d1(0, 4, x, y, p_i)]; dA2dy = d1fd1x[idx_d1(1, 4, x, y, p_i)]
    dA3dx = d1fd1x[idx_d1(0, 5, x, y, p_i)]; dA3dy = d1fd1x[idx_d1(1, 5, x, y, p_i)]
    dpsi1dx = d1fd1x[idx_d1(0, 6, x, y, p_i)]; dpsi1dy = d1fd1x[idx_d1(1, 6, x, y, p_i)]
    dpsi2dx = d1fd1x[idx_d1(0, 7, x, y, p_i)]; dpsi2dy = d1fd1x[idx_d1(1, 7, x, y, p_i)]
    d2m1d2x = d2fd2x[idx_d2(0, 0, 0, x, y, p_i)]; d2m1d2y = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    d2m2d2x = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]; d2m2d2y = d2fd2x[idx_d2(1, 1, 1, x, y, p_i)]
    d2m3d2x = d2fd2x[idx_d2(0, 0, 2, x, y, p_i)]; d2m3d2y = d2fd2x[idx_d2(1, 1, 2, x, y, p_i)]
    d2A1dxdy = d2fd2x[idx_d2(0, 1, 3, x, y, p_i)]; d2A1d2y = d2fd2x[idx_d2(1, 1, 3, x, y, p_i)]
    d2A2d2x  = d2fd2x[idx_d2(0, 0, 4, x, y, p_i)]; d2A2dxdy = d2fd2x[idx_d2(1, 0, 4, x, y, p_i)]
    d2A3d2x  = d2fd2x[idx_d2(0, 0, 5, x, y, p_i)]; d2A3d2y  = d2fd2x[idx_d2(1, 1, 5, x, y, p_i)]
    d2psi1d2x = d2fd2x[idx_d2(0, 0, 6, x, y, p_i)]; d2psi1d2y = d2fd2x[idx_d2(1, 1, 6, x, y, p_i)]
    d2psi2d2x = d2fd2x[idx_d2(0, 0, 7, x, y, p_i)]; d2psi2d2y = d2fd2x[idx_d2(1, 1, 7, x, y, p_i)]
    psi_sq = psi1*psi1 + psi2*psi2
    m_sq = m1*m1 + m2*m2 + m3*m3
    dm_sq = (dm1dx*dm1dx + dm2dx*dm2dx + dm3dx*dm3dx + dm1dy*dm1dy + dm2dy*dm2dy + dm3dy*dm3dy)
    A_sq = A1*A1 + A2*A2 + A3*A3
    g0 = alpha*m1 + beta*m1*m_sq - gamma*gamma*d2m1d2x - gamma*gamma*d2m1d2y - dA3dy + 2.0*eta1*psi_sq*m1 - 2.0*eta2*(d2m1d2x + d2m1d2y)*psi_sq
    g1 = alpha*m2 + beta*m2*m_sq - gamma*gamma*d2m2d2x - gamma*gamma*d2m2d2y + dA3dx + 2.0*eta1*psi_sq*m2 - 2.0*eta2*(d2m2d2x + d2m2d2y)*psi_sq
    g2 = alpha*m3 + beta*m3*m_sq - gamma*gamma*d2m3d2x - gamma*gamma*d2m3d2y - dA2dx + dA1dy + 2.0*eta1*psi_sq*m3 - 2.0*eta2*(d2m3d2x + d2m3d2y)*psi_sq
    g3 = q*q*A1*psi_sq + q*(psi1*dpsi2dx - psi2*dpsi1dx) + d2A2dxdy - d2A1d2y - dm3dy
    g4 = q*q*A2*psi_sq + q*(psi1*dpsi2dy - psi2*dpsi1dy) + d2A1dxdy - d2A2d2x + dm3dx
    g5 = q*q*A3*psi_sq - d2A3d2x - d2A3d2y - dm2dx + dm1dy
    g6 = ha*psi1 + hb*psi1*psi_sq - d2psi1d2x - d2psi1d2y + q*q*A_sq*psi1 + 2.0*q*A1*dpsi2dx + 2.0*q*A2*dpsi2dy + q*psi2*(dA1dx + dA2dy) + 2.0*eta1*m_sq*psi1 + 2.0*eta2*psi1*psi1*dm_sq
    g7 = ha*psi2 + hb*psi2*psi_sq - d2psi2d2x - d2psi2d2y + q*q*A_sq*psi2 - 2.0*q*A1*dpsi1dx - 2.0*q*A2*dpsi1dy - q*psi1*(dA1dx + dA2dy) + 2.0*eta1*m_sq*psi2 + 2.0*eta2*psi2*psi2*dm_sq
    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1
    EnergyGradient[idx_field(2, x, y, p_i)] = g2
    EnergyGradient[idx_field(3, x, y, p_i)] = g3
    EnergyGradient[idx_field(4, x, y, p_i)] = g4
    EnergyGradient[idx_field(5, x, y, p_i)] = g5
    EnergyGradient[idx_field(6, x, y, p_i)] = g6
    EnergyGradient[idx_field(7, x, y, p_i)] = g7
    project_orthogonal_magnetization(EnergyGradient, Field, x, y, p_i, p_f)
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)