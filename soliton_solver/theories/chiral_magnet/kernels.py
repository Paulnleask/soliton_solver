"""
CUDA kernels and device helpers for the chiral magnet theory.

Examples
--------
Use ``create_grid_kernel`` to build the coordinate grid.
Use ``compute_energy_kernel`` to evaluate the energy density.
Use ``do_gradient_step_kernel`` to compute the local energy gradient update.
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
def compute_norm_magnetization(Field, x, y, p_i, p_f):
    """
    Normalize the magnetization vector at one lattice site.

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
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The magnetization components are normalized in place.

    Examples
    --------
    Use ``compute_norm_magnetization(Field, x, y, p_i, p_f)`` inside a CUDA kernel to enforce unit magnetization.
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
    Project a field orthogonally to the local magnetization.

    Parameters
    ----------
    func : device array
        Flattened field to project.
    Field : device array
        Flattened magnetization field.
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
        The projected field is written into ``func`` in place.

    Examples
    --------
    Use ``project_orthogonal_magnetization(func, Field, x, y, p_i, p_f)`` inside a CUDA kernel to remove the component parallel to the magnetization.
    """
    number_magnetization_fields = p_i[10]
    lm = 0.0
    for a in range(number_magnetization_fields):
        lm += func[idx_field(a, x, y, p_i)] * Field[idx_field(a, x, y, p_i)]
    for a in range(number_magnetization_fields):
        func[idx_field(a, x, y, p_i)] -= lm * Field[idx_field(a, x, y, p_i)]

# compute_norm_magnetization and project_orthogonal_magnetization must be @cuda.jit(device=True)
do_rk4_kernel = make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization)

@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the local chiral magnet energy contribution.

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
        Cell integrated energy contribution at the lattice site.

    Examples
    --------
    Use ``compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local energy.
    """

    grid_volume = p_f[4]
    coup_K  = p_f[6]; coup_h  = p_f[7]; coup_mu = p_f[8]
    number_coordinates = p_i[3]; number_total_fields = p_i[4]; number_magnetization_fields = p_i[10]

    dmi_dresselhaus = p_i[11]; dmi_rashba = p_i[12]; dmi_heusler = p_i[13]; dmi_hybrid = p_i[14]

    demag = p_i[15]

    def levi(i, j, k):
        if i == 0 and j == 1 and k == 2: return  1.0
        if i == 1 and j == 2 and k == 0: return  1.0
        if i == 2 and j == 0 and k == 1: return  1.0
        if i == 0 and j == 2 and k == 1: return -1.0
        if i == 2 and j == 1 and k == 0: return -1.0
        if i == 1 and j == 0 and k == 2: return -1.0
        return 0.0

    D00 = 0.0; D01 = 0.0
    D10 = 0.0; D11 = 0.0

    if dmi_dresselhaus:
        D00 = -1.0; D11 = -1.0
    elif dmi_rashba:
        D01 = -1.0; D10 =  1.0
    elif dmi_heusler:
        D00 =  1.0; D11 = -1.0
    elif dmi_hybrid:
        s = 0.7071067811865476  # sqrt(2)/2
        D00 = -s; D01 = -s
        D10 =  s; D11 = -s

    energy = 0.0
    for a in range(number_magnetization_fields):
        for i in range(number_coordinates):
            dfa = d1fd1x[idx_d1(i, a, x, y, p_i)]
            energy += 0.5 * dfa * dfa

    for i in range(number_coordinates):
        for j in range(number_coordinates):
            Dij = (
                D00 if (j == 0 and i == 0) else
                D01 if (j == 0 and i == 1) else
                D10 if (j == 1 and i == 0) else
                D11
            )

            if Dij == 0.0:
                continue

            for k in range(number_magnetization_fields):
                mk = Field[idx_field(k, x, y, p_i)]
                for l in range(number_magnetization_fields):
                    eps = levi(j, k, l)
                    if eps != 0.0:
                        energy += (Dij * eps * mk* d1fd1x[idx_d1(i, l, x, y, p_i)])

        if demag:
            energy -= Field[idx_field(3, x, y, p_i)] * d1fd1x[idx_d1(i, i, x, y, p_i)] + d1fd1x[idx_d1(i, 3, x, y, p_i)] * d1fd1x[idx_d1(i, 3, x, y, p_i)] / (2.0 * coup_mu)

    mz = Field[idx_field(number_magnetization_fields - 1, x, y, p_i)]
    energy += coup_h * (1.0 - mz)
    energy += coup_K * (1.0 - mz * mz)
    return energy * grid_volume

@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per site energy density field.

    Parameters
    ----------
    en : device array
        Output scalar field.
    Field : device array
        Flattened state array.
    d1fd1x : device array
        Flattened first derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local energy contributions are written into ``en``.

    Examples
    --------
    Launch ``compute_energy_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to compute the energy density field.
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
    Compute the local skyrmion charge density contribution.

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
        Local skyrmion charge density contribution.

    Examples
    --------
    Use ``compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local topological density.
    """
    grid_volume = p_f[4]
    # Define fields
    m1 = Field[idx_field(0, x, y, p_i)];m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    # Define derivatives
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    # cross = mx × my
    cx0 = dm2dx*dm3dy - dm3dx*dm2dy; cx1 = dm3dx*dm1dy - dm1dx*dm3dy; cx2 = dm1dx*dm2dy - dm2dx*dm1dy
    # Skyrmion charge density
    charge = m1*cx0 + m2*cx1 + m3*cx2
    return charge * (grid_volume / (4.0 * math.pi))

@cuda.jit
def compute_skyrmion_number_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute the per site skyrmion density field.

    Parameters
    ----------
    en : device array
        Output scalar field.
    Field : device array
        Flattened state array.
    d1fd1x : device array
        Flattened first derivative buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local skyrmion density contributions are written into ``en``.

    Examples
    --------
    Launch ``compute_skyrmion_number_kernel[grid2d, block2d](en, Field, d1fd1x, p_i, p_f)`` to compute the skyrmion density field.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)

@cuda.jit(device=True)
def compute_magnetic_charge_point(d1fd1x, x, y, p_i, p_f):
    """
    Compute the local magnetic charge contribution.

    Parameters
    ----------
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
        Local magnetic charge contribution.

    Examples
    --------
    Use ``compute_magnetic_charge_point(d1fd1x, x, y, p_i, p_f)`` inside a CUDA kernel to evaluate the local magnetic charge.
    """
    charge = 0.0
    grid_volume = p_f[4]
    number_coordinates = p_i[3]
    for i in range(number_coordinates):
        charge -= d1fd1x[idx_d1(i, i, x, y, p_i)]
    return charge * grid_volume

@cuda.jit
def compute_magnetic_charge_kernel(Field, d1fd1x, en, p_i, p_f):
    """
    Compute the per site magnetic charge field.

    Parameters
    ----------
    Field : device array
        Flattened state array.
    d1fd1x : device array
        Flattened first derivative buffer.
    en : device array
        Output scalar field.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The local magnetic charge contributions are written into ``en``.

    Examples
    --------
    Launch ``compute_magnetic_charge_kernel[grid2d, block2d](Field, d1fd1x, en, p_i, p_f)`` to compute the magnetic charge field.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_magnetic_charge_point(d1fd1x, x, y, p_i, p_f)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient and velocity update.

    Parameters
    ----------
    Velocity : device array
        Velocity field buffer.
    Field : device array
        Flattened state array.
    EnergyGradient : device array
        Energy gradient buffer.
    d1fd1x : device array
        Flattened first derivative buffer.
    d2fd2x : device array
        Flattened second derivative buffer.
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
    Use ``do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)`` inside a CUDA kernel to compute the local gradient update.
    """

    time_step = p_f[5]
    coup_K = p_f[6]; coup_h = p_f[7]; coup_mu = p_f[8]
    number_coordinates = p_i[3]; number_total_fields = p_i[4]; number_magnetization_fields = p_i[10]
    xlen = p_i[0]; ylen = p_i[1]; halo = p_i[2]

    dmi_dresselhaus = p_i[11]; dmi_rashba = p_i[12]; dmi_heusler = p_i[13]; dmi_hybrid = p_i[14]
    demag = p_i[15]

    def levi(i, j, k):
        if i == 0 and j == 1 and k == 2: return  1.0
        if i == 1 and j == 2 and k == 0: return  1.0
        if i == 2 and j == 0 and k == 1: return  1.0
        if i == 0 and j == 2 and k == 1: return -1.0
        if i == 2 and j == 1 and k == 0: return -1.0
        if i == 1 and j == 0 and k == 2: return -1.0
        return 0.0

    D00 = 0.0; D01 = 0.0
    D10 = 0.0; D11 = 0.0
    if dmi_dresselhaus:
        D00 = -1.0; D11 = -1.0
    elif dmi_rashba:
        D01 = -1.0; D10 =  1.0
    elif dmi_heusler:
        D00 =  1.0; D11 = -1.0
    elif dmi_hybrid:
        s = 0.7071067811865476  # sqrt(2)/2
        D00 = -s; D01 = -s
        D10 =  s; D11 = -s

    for a in range(number_total_fields):
        EnergyGradient[idx_field(a, x, y, p_i)] = 0.0

    for i in range(number_magnetization_fields):
        lap = 0.0
        for j in range(number_coordinates):
            lap += d2fd2x[idx_d2(j, j, i, x, y, p_i)]
        EnergyGradient[idx_field(i, x, y, p_i)] -= lap

    if (D00 != 0.0) or (D01 != 0.0) or (D10 != 0.0) or (D11 != 0.0):
        for i in range(number_magnetization_fields):
            for j in range(number_magnetization_fields):
                # a,b in {0,1} (coordinates)
                # a=0,b=0 => D00
                eps = levi(0, i, j)
                if eps != 0.0:
                    EnergyGradient[idx_field(i, x, y, p_i)] += 2.0 * D00 * eps * d1fd1x[idx_d1(0, j, x, y, p_i)]
                # a=1,b=0 => D01 uses D[b=0][a=1]
                eps = levi(0, i, j)
                if eps != 0.0 and D01 != 0.0:
                    EnergyGradient[idx_field(i, x, y, p_i)] += 2.0 * D01 * eps * d1fd1x[idx_d1(1, j, x, y, p_i)]
                # b=1,a=0 => D10 uses D[b=1][a=0]
                eps = levi(1, i, j)
                if eps != 0.0 and D10 != 0.0:
                    EnergyGradient[idx_field(i, x, y, p_i)] += 2.0 * D10 * eps * d1fd1x[idx_d1(0, j, x, y, p_i)]
                # b=1,a=1 => D11 uses D[b=1][a=1]
                eps = levi(1, i, j)
                if eps != 0.0 and D11 != 0.0:
                    EnergyGradient[idx_field(i, x, y, p_i)] += 2.0 * D11 * eps * d1fd1x[idx_d1(1, j, x, y, p_i)]

    last = number_magnetization_fields - 1
    mz = Field[idx_field(last, x, y, p_i)]
    EnergyGradient[idx_field(last, x, y, p_i)] -= coup_h
    EnergyGradient[idx_field(last, x, y, p_i)] -= 2.0 * coup_K * mz

    if demag:
        for i in range(number_coordinates):
            # Backreaction on magnetization
            EnergyGradient[idx_field(i, x, y, p_i)] += d1fd1x[idx_d1(i, 3, x, y, p_i)]
            # Magnetic potential gradient
            EnergyGradient[idx_field(3, x, y, p_i)] += coup_mu * d1fd1x[idx_d1(i, i, x, y, p_i)] - d2fd2x[idx_d2(i, i, 3, x, y, p_i)]

    if (x < halo) or (x > xlen - halo - 1) or (y < halo) or (y > ylen - halo - 1):
        for a in range(number_total_fields):
            EnergyGradient[idx_field(a, x, y, p_i)] = 0.0

    project_orthogonal_magnetization(EnergyGradient, Field, x, y, p_i, p_f)
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)