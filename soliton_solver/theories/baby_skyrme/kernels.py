"""
Core CUDA kernels and device helpers for the Baby Skyrme model.

Examples
--------
Use ``create_grid_kernel`` to construct the physical coordinate grid on the device.
Use ``compute_energy_kernel`` to evaluate the per site energy contributions.
Use ``compute_skyrmion_number_kernel`` to evaluate the per site skyrmion density contributions.
Use ``do_gradient_step_kernel`` to compute the local energy gradient for relaxation.
"""

import math
from math import sin, cos, atan2, sqrt
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds
from soliton_solver.core.integrator import make_do_gradient_step_kernel
from soliton_solver.core.integrator import make_do_rk4_kernel

@cuda.jit
def create_grid_kernel(grid, p_i, p_f):
    """
    Populate the physical coordinate grid on the device.

    Parameters
    ----------
    grid : device array
        Flattened coordinate array with components for the spatial coordinates.
    p_i : device array
        Integer parameter array used for indexing and bounds checks.
    p_f : device array
        Float parameter array containing the lattice spacings.

    Returns
    -------
    None
        The coordinate values are written into ``grid`` in place.

    Examples
    --------
    Launch ``create_grid_kernel[grid2d, block2d](grid, p_i, p_f)`` to construct the coordinate grid.
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
    Normalize the magnetization vector at a lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array containing the magnetization components.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing the number of magnetization components.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The magnetization components are normalized in place.

    Examples
    --------
    Use ``compute_norm_magnetization(Field, x, y, p_i, p_f)`` inside a CUDA kernel to normalize the local magnetization.
    """
    number_magnetization_fields = p_i[10]
    s = 0.0
    for a in range(number_magnetization_fields):
        v = Field[idx_field(a, x, y, p_i)]
        s += v * v
    s = math.sqrt(s)
    if s == 0.0:
        return
    for a in range(number_magnetization_fields):
        Field[idx_field(a, x, y, p_i)] /= (s)

@cuda.jit(device=True)
def project_orthogonal_magnetization(func, Field, x, y, p_i, p_f):
    """
    Project a local vector field orthogonally to the magnetization.

    Parameters
    ----------
    func : device array
        Flattened array containing the vector to project.
    Field : device array
        Flattened field array containing the magnetization components.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing the number of magnetization components.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The projected vector is written into ``func`` in place.

    Examples
    --------
    Use ``project_orthogonal_magnetization(func, Field, x, y, p_i, p_f)`` inside a CUDA kernel to enforce orthogonality.
    """
    number_magnetization_fields = p_i[10]
    lm = 0.0
    for a in range(number_magnetization_fields):
        lm += func[idx_field(a, x, y, p_i)] * Field[idx_field(a, x, y, p_i)]
    for a in range(number_magnetization_fields):
        func[idx_field(a, x, y, p_i)] -= lm * Field[idx_field(a, x, y, p_i)]

do_rk4_kernel = make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization)

@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the local energy contribution at a lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array containing the magnetization components.
    d1fd1x : device array
        Flattened first derivative buffer for the field.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing model flags and dimensions.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    float
        Cell integrated energy contribution at the lattice site.

    Examples
    --------
    Use ``compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel after the first derivatives have been computed.
    """
    grid_volume = p_f[4]
    mpi = p_f[6]; mpi2 = mpi * mpi
    kappa = p_f[7]; kappa2 = kappa * kappa
    m1 = Field[idx_field(0, x, y, p_i)]; m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    grad_x_sq = dm1dx*dm1dx + dm2dx*dm2dx + dm3dx*dm3dx
    grad_y_sq = dm1dy*dm1dy + dm2dy*dm2dy + dm3dy*dm3dy
    energy = 0.5 * (grad_x_sq + grad_y_sq)
    cx = dm2dx*dm3dy - dm3dx*dm2dy; cy = dm3dx*dm1dy - dm1dx*dm3dy; cz = dm1dx*dm2dy - dm2dx*dm1dy
    cross_sq = cx*cx + cy*cy + cz*cz
    energy += 0.5 * kappa2 * cross_sq
    Potential_Standard = p_i[11]
    Potential_Holomorphic = p_i[12]
    Potential_EasyPlane = p_i[13]
    Potential_Dihedral2 = p_i[14]
    Potential_Aloof = p_i[15]
    Potential_Dihedral3 = p_i[16]
    Potential_Broken = p_i[17]; N = p_i[19]
    Potential_DoubleVacua = p_i[18]
    if Potential_Standard:
        energy += mpi2 * (1.0 - m3)
    if Potential_Holomorphic:
        t = 1.0 - m3
        energy += mpi2 * t*t*t*t
    if Potential_EasyPlane:
        energy += 0.5 * mpi2 * m1*m1
    if Potential_Dihedral2:
        energy += 0.5 * mpi2 * (1.0 - m1*m1) * (1.0 - m3*m3)
    if Potential_Aloof:
        t = 1.0 - m3
        energy += 0.5 * mpi2 * t * (1.0 + t*t*t)
    if Potential_Dihedral3:
        energy += (16.0 * mpi2 * (1.0 - m3)* (1.0 + 3.0*m3*m3 + 3.0*m1*m2*m2 - m1*m1*m1))
    if Potential_Broken:
        zr, zi = m1, m2
        for _ in range(N - 1):
            zr, zi = zr*m1 - zi*m2, zr*m2 + zi*m1
        wr = 1.0 - zr
        wi = -zi
        energy += mpi2 * (1.0 - m3) * (wr*wr + wi*wi)
    if Potential_DoubleVacua:
        energy += mpi2 * (1.0 - m3*m3)
    return energy * grid_volume

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per site energy contributions across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field storing the local energy contributions.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        First derivative buffer written during the kernel and used for the energy evaluation.
    p_i : device array
        Integer parameter array containing the number of field components.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    None
        The local energy contributions are written into ``en`` in place.

    Examples
    --------
    Launch ``compute_energy_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to evaluate the energy density on the grid.
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
    Compute the local skyrmion charge density contribution at a lattice site.

    Parameters
    ----------
    Field : device array
        Flattened field array containing the magnetization components.
    d1fd1x : device array
        First derivative buffer for the magnetization field.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the cell volume.

    Returns
    -------
    float
        Per cell contribution to the skyrmion number.

    Examples
    --------
    Use ``compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel after the first derivatives have been computed.
    """
    grid_volume = p_f[4]
    m1 = Field[idx_field(0, x, y, p_i)];m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    cx0 = dm2dx*dm3dy - dm3dx*dm2dy; cx1 = dm3dx*dm1dy - dm1dx*dm3dy; cx2 = dm1dx*dm2dy - dm2dx*dm1dy
    charge = m1*cx0 + m2*cx1 + m3*cx2
    return charge * (grid_volume / (4.0 * math.pi))

@cuda.jit
def compute_skyrmion_number_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per site skyrmion charge density contributions across the grid.

    Parameters
    ----------
    en : device array
        Output scalar field storing the local skyrmion density contributions.
    Field : device array
        Flattened field array.
    d1fd1x : device array
        First derivative buffer written during the kernel and used for the skyrmion density evaluation.
    p_i : device array
        Integer parameter array containing the number of magnetization components.
    p_f : device array
        Float parameter array containing the model parameters.

    Returns
    -------
    None
        The local skyrmion density contributions are written into ``en`` in place.

    Examples
    --------
    Launch ``compute_skyrmion_number_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to evaluate the skyrmion density on the grid.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient and set the local velocity update.

    Parameters
    ----------
    Velocity : device array
        Output array receiving the local velocity update.
    Field : device array
        Flattened field array.
    EnergyGradient : device array
        Output array receiving the local energy gradient.
    d1fd1x : device array
        First derivative buffer.
    d2fd2x : device array
        Second derivative buffer.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    p_i : device array
        Integer parameter array containing model flags and dimensions.
    p_f : device array
        Float parameter array containing the time step and model parameters.

    Returns
    -------
    None
        The local energy gradient and velocity are written in place.

    Examples
    --------
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)`` inside a CUDA kernel after the derivatives have been computed.
    """
    time_step = p_f[5]
    mpi = p_f[6]; mpi2 = mpi * mpi
    kappa = p_f[7]; kappa2 = kappa * kappa
    Potential_Standard = p_i[11]
    Potential_Holomorphic = p_i[12]
    Potential_EasyPlane = p_i[13]
    Potential_Dihedral2 = p_i[14]
    Potential_Aloof = p_i[15]
    Potential_Dihedral3 = p_i[16]
    Potential_Broken = p_i[17]; N = p_i[19]
    Potential_DoubleVacua = p_i[18]
    m1 = Field[idx_field(0, x, y, p_i)]; m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    F00 = dm1dx; F01 = dm2dx; F02 = dm3dx
    F10 = dm1dy; F11 = dm2dy; F12 = dm3dy
    d2m1d2x = d2fd2x[idx_d2(0, 0, 0, x, y, p_i)]; d2m1d2y = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]; d2m1dxdy = d2fd2x[idx_d2(0, 1, 0, x, y, p_i)]; d2m1dydx = d2fd2x[idx_d2(1, 0, 0, x, y, p_i)]
    d2m2d2x = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]; d2m2d2y = d2fd2x[idx_d2(1, 1, 1, x, y, p_i)]; d2m2dxdy = d2fd2x[idx_d2(0, 1, 1, x, y, p_i)]; d2m3dxdy = d2fd2x[idx_d2(0, 1, 2, x, y, p_i)]
    d2m3d2x = d2fd2x[idx_d2(0, 0, 2, x, y, p_i)]; d2m3d2y = d2fd2x[idx_d2(1, 1, 2, x, y, p_i)]; d2m2dydx = d2fd2x[idx_d2(1, 0, 1, x, y, p_i)]; d2m3dydx = d2fd2x[idx_d2(1, 0, 2, x, y, p_i)]
    S000 = d2m1d2x; S001 = d2m2d2x; S002 = d2m3d2x
    S110 = d2m1d2y; S111 = d2m2d2y; S112 = d2m3d2y
    S010 = d2m1dxdy; S011 = d2m2dxdy; S012 = d2m3dxdy
    S100 = d2m1dydx; S101 = d2m2dydx; S102 = d2m3dydx
    sum_F0_sq = F00*F00 + F01*F01 + F02*F02
    sum_F1_sq = F10*F10 + F11*F11 + F12*F12
    g0 = 0.0; g1 = 0.0; g2 = 0.0

    L0_0 = S000 + S110
    L0_1 = S001 + S111
    L0_2 = S002 + S112

    g0 -= L0_0
    g1 -= L0_1
    g2 -= L0_2

    Spp0 = S000; Spp1 = S001; Spp2 = S002

    sum_all_sq = sum_F0_sq + sum_F1_sq
    g0 -= kappa2 * Spp0 * sum_all_sq
    g1 -= kappa2 * Spp1 * sum_all_sq
    g2 -= kappa2 * Spp2 * sum_all_sq

    g0 += kappa2 * S000 * sum_F0_sq
    g1 += kappa2 * S001 * sum_F0_sq
    g2 += kappa2 * S002 * sum_F0_sq

    A0 = S010 * F10 - S110 * F00
    A1 = S011 * F11 - S111 * F01
    A2 = S012 * F12 - S112 * F02

    sumA = A0 + A1 + A2
    dot01 = F00*F10 + F01*F11 + F02*F12

    g0 -= kappa2 * (F00 * sumA - S010 * dot01)
    g1 -= kappa2 * (F01 * sumA - S011 * dot01)
    g2 -= kappa2 * (F02 * sumA - S012 * dot01)

    Spp0 = S110; Spp1 = S111; Spp2 = S112
    g0 -= kappa2 * Spp0 * sum_all_sq
    g1 -= kappa2 * Spp1 * sum_all_sq
    g2 -= kappa2 * Spp2 * sum_all_sq

    g0 += kappa2 * S110 * sum_F1_sq
    g1 += kappa2 * S111 * sum_F1_sq
    g2 += kappa2 * S112 * sum_F1_sq

    B0 = S100 * F00 - S000 * F10
    B1 = S101 * F01 - S001 * F11
    B2 = S102 * F02 - S002 * F12
    sumB = B0 + B1 + B2

    g0 -= kappa2 * (F10 * sumB - S100 * dot01)
    g1 -= kappa2 * (F11 * sumB - S101 * dot01)
    g2 -= kappa2 * (F12 * sumB - S102 * dot01)

    if Potential_Standard:
        g2 -= mpi2
    if Potential_Holomorphic:
        t = 1.0 - m3
        g2 -= 4.0 * mpi2 * t * t * t
    if Potential_EasyPlane:
        g0 += mpi2 * m1
    if Potential_Dihedral2:
        g0 -= mpi2 * (1.0 - m3*m3) * m1
        g2 -= mpi2 * (1.0 - m1*m1) * m3
    if Potential_Aloof:
        t = 1.0 - m3
        t3 = t * t * t
        g2 -= mpi2 * (0.5 * (1.0 + t3) + 1.5 * t3)
    if Potential_Dihedral3:
        g0 += 48.0 * mpi2 * ((m2*m2) - (m1*m1)) * (1.0 - m3)
        g1 += 96.0 * mpi2 * m1 * m2 * (1.0 - m3)
        g2 -= 16.0 * mpi2 * (-(m1*m1*m1) + 3.0*m1*(m2*m2) + (1.0 - 3.0*m3)*(1.0 - 3.0*m3))
    if Potential_Broken:
        r2 = m1*m1 + m2*m2
        theta = atan2(m2, m1)
        rN2 = r2 ** (0.5 * N)
        rN = r2 ** (1.0 * N)
        c = cos(N * theta)
        s = sin(N * theta)
        denom = sqrt(1.0 - 2.0 * c * rN2 + rN)
        norm1mzN = (1.0 - 2.0 * c * rN2 + rN)
        if denom != 0.0 and r2 != 0.0:
            pref = 2.0 * mpi2 * N * sqrt(norm1mzN) * (r2 ** (-1.0 + 0.5 * N)) * (-1.0 + m3) / denom
            g0 += pref * (s * m2 + m1 * (c - rN2))
            g1 += (-pref) * (s * m1 + m2 * (-c + rN2))
        g2 -= mpi2 * norm1mzN
    if Potential_DoubleVacua:
        g2 -= 2.0 * mpi2 * m3
    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1
    EnergyGradient[idx_field(2, x, y, p_i)] = g2
    project_orthogonal_magnetization(EnergyGradient, Field, x, y, p_i, p_f)
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)