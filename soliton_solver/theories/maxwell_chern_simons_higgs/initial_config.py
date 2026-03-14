"""
Initial condition kernels for the Maxwell-Chern-Simons-Higgs theory.

Examples
--------
Use ``create_ground_state_kernel`` to initialize the uniform ground state.
Use ``create_initial_configuration_kernel`` to initialize the default vortex ansatz.
Use ``create_vortex_kernel`` to imprint a vortex or antivortex on the Higgs field.
"""
import math
from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds

@cuda.jit(device=True, inline=True)
def position(xcent, ycent, x, y, grid, p_i):
    """
    Compute the radial distance from a physical space center to a lattice site.

    Parameters
    ----------
    xcent : float
        Physical x coordinate of the center.
    ycent : float
        Physical y coordinate of the center.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    grid : device array
        Flattened coordinate array.
    p_i : device array
        Integer parameter array used for indexing.

    Returns
    -------
    float
        Radial distance from the center to the lattice site.

    Examples
    --------
    Use ``r = position(xcent, ycent, x, y, grid, p_i)`` to evaluate radial profiles.
    """
    dx = grid[idx_field(0, x, y, p_i)]
    dy = grid[idx_field(1, x, y, p_i)]
    return math.sqrt((dx - xcent) * (dx - xcent) + (dy - ycent) * (dy - ycent))

@cuda.jit(device=True, inline=True)
def profile_function_superconductor(r, max_r, value):
    """
    Compute the superconducting radial profile.

    Parameters
    ----------
    r : float
        Radial distance from the core.
    max_r : float
        Radius beyond which the profile is fixed to its asymptotic value.
    value : float
        Parameter controlling the profile width.

    Returns
    -------
    float
        Profile value at radius ``r``.

    Examples
    --------
    Use ``fs = profile_function_superconductor(r, max_r, value)`` to evaluate the Higgs amplitude profile.
    """
    if r > max_r:
        return 1.0
    return math.tanh(value * r)

@cuda.jit(device=True, inline=True)
def angle(r, xcent, ycent, x, y, grid, p_i):
    """
    Compute the azimuthal angle around a physical space center.

    Parameters
    ----------
    r : float
        Radial distance from the center.
    xcent : float
        Physical x coordinate of the center.
    ycent : float
        Physical y coordinate of the center.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    grid : device array
        Flattened coordinate array.
    p_i : device array
        Integer parameter array used for indexing.

    Returns
    -------
    float
        Azimuthal angle in the interval ``[0, 2π)``.

    Examples
    --------
    Use ``th = angle(r, xcent, ycent, x, y, grid, p_i)`` to build a phase field.
    """
    theta = 0.0
    if r == 0.0:
        return theta
    gx = grid[idx_field(0, x, y, p_i)] - xcent
    gy = grid[idx_field(1, x, y, p_i)] - ycent
    ang = math.atan2(gy, gx)
    if ang < 0.0:
        ang += 2.0 * math.pi
    return theta + ang

@cuda.jit
def create_ground_state_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i, p_f):
    """
    Initialize the uniform ground state and zero the auxiliary buffers.

    Parameters
    ----------
    Velocity : device array
        Velocity field buffer.
    Field : device array
        State field buffer.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The initial field and auxiliary buffers are written in place.

    Examples
    --------
    Launch ``create_ground_state_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i, p_f)`` to initialize the ground state.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    u1 = p_f[9]
    Field[idx_field(0, x, y, p_i)] = 0.0
    Field[idx_field(1, x, y, p_i)] = 0.0
    Field[idx_field(2, x, y, p_i)] = u1
    Field[idx_field(3, x, y, p_i)] = 0.0
    Field[idx_field(4, x, y, p_i)] = 0.0
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = 0.0
        k1[idx_field(a, x, y, p_i)] = 0.0
        k2[idx_field(a, x, y, p_i)] = 0.0
        k3[idx_field(a, x, y, p_i)] = 0.0
        k4[idx_field(a, x, y, p_i)] = 0.0
        l1[idx_field(a, x, y, p_i)] = 0.0
        l2[idx_field(a, x, y, p_i)] = 0.0
        l3[idx_field(a, x, y, p_i)] = 0.0
        l4[idx_field(a, x, y, p_i)] = 0.0
        Temp[idx_field(a, x, y, p_i)] = 0.0

@cuda.jit
def create_initial_configuration_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, vortex_number, p_i, p_f):
    """
    Initialize the default vortex configuration and zero the auxiliary buffers.

    Parameters
    ----------
    Velocity : device array
        Velocity field buffer.
    Field : device array
        State field buffer.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer.
    vortex_number : float
        Vortex winding number.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The initial field and auxiliary buffers are written in place.

    Examples
    --------
    Launch ``create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, vortex_number, p_i, p_f)`` to initialize the default configuration.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    ainf = p_f[11]
    u1 = p_f[9]
    xcent = (grid[idx_field(0, 0, 0, p_i)] + grid[idx_field(0, xlen-1, ylen-1, p_i)]) / 2.0
    ycent = (grid[idx_field(1, 0, 0, p_i)] + grid[idx_field(1, xlen-1, ylen-1, p_i)]) / 2.0
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, x, y, grid, p_i)
    fs = profile_function_superconductor(r1, rmax, 0.2)
    eps = 1e-12
    rr = r1 if r1 > eps else eps
    Field[idx_field(0, x, y, p_i)] = ainf * fs * math.sin(th) / rr
    Field[idx_field(1, x, y, p_i)] = -ainf * fs * math.cos(th) / rr
    Field[idx_field(2, x, y, p_i)] = u1 * fs * math.cos(vortex_number * th)
    Field[idx_field(3, x, y, p_i)] = u1 * fs * math.sin(vortex_number * th)
    Field[idx_field(4, x, y, p_i)] = 0.0
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = 0.0
        k1[idx_field(a, x, y, p_i)] = 0.0
        k2[idx_field(a, x, y, p_i)] = 0.0
        k3[idx_field(a, x, y, p_i)] = 0.0
        k4[idx_field(a, x, y, p_i)] = 0.0
        l1[idx_field(a, x, y, p_i)] = 0.0
        l2[idx_field(a, x, y, p_i)] = 0.0
        l3[idx_field(a, x, y, p_i)] = 0.0
        l4[idx_field(a, x, y, p_i)] = 0.0
        Temp[idx_field(a, x, y, p_i)] = 0.0

@cuda.jit
def create_vortex_kernel(Field, grid, pxi, pxj, vortex, p_i, p_f):
    """
    Multiply the Higgs field by a vortex or antivortex phase factor.

    Parameters
    ----------
    Field : device array
        State field buffer.
    grid : device array
        Flattened coordinate array.
    pxi : int
        Pixel index of the vortex center along the x direction.
    pxj : int
        Pixel index of the vortex center along the y direction.
    vortex : bool
        Flag selecting the phase sign.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The Higgs field is updated in place.

    Examples
    --------
    Launch ``create_vortex_kernel[grid2d, block2d](Field, grid, pxi, pxj, vortex, p_i, p_f)`` to imprint a vortex.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    u1 = p_f[9]
    vortex_number = p_f[10]
    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen:
        return
    xcent = grid[idx_field(0, pxi, pxj, p_i)]
    ycent = grid[idx_field(1, pxi, pxj, p_i)]
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, x, y, grid, p_i)
    fs = profile_function_superconductor(r1, rmax, 0.2)
    old_r = Field[idx_field(2, x, y, p_i)]
    old_i = Field[idx_field(3, x, y, p_i)]
    phase = (-1.0 if vortex else 1.0) * th
    int_r = u1 * fs * math.cos(vortex_number * phase)
    int_i = u1 * fs * math.sin(vortex_number * phase)
    new_r = (old_r * int_r - old_i * int_i) / u1
    new_i = (old_r * int_i + old_i * int_r) / u1
    Field[idx_field(2, x, y, p_i)] = new_r
    Field[idx_field(3, x, y, p_i)] = new_i

def initialize(*, Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d, p_i_h=None, p_f_h=None, grid2d, block2d, config: dict | None = None):
    """
    Initialize the theory fields using the requested configuration mode.

    Parameters
    ----------
    Velocity : device array
        Velocity field buffer.
    Field : device array
        State field buffer.
    grid : device array
        Flattened coordinate array.
    k1 : device array
        First RK buffer.
    k2 : device array
        Second RK buffer.
    k3 : device array
        Third RK buffer.
    k4 : device array
        Fourth RK buffer.
    l1 : device array
        First auxiliary RK buffer.
    l2 : device array
        Second auxiliary RK buffer.
    l3 : device array
        Third auxiliary RK buffer.
    l4 : device array
        Fourth auxiliary RK buffer.
    Temp : device array
        Temporary field buffer.
    p_i_d : device array
        Integer parameter array on the device.
    p_f_d : device array
        Float parameter array on the device.
    p_i_h : array-like, optional
        Integer parameter array on the host.
    p_f_h : array-like, optional
        Float parameter array on the host.
    grid2d : tuple
        CUDA grid configuration.
    block2d : tuple
        CUDA block configuration.
    config : dict or None, optional
        Initialization options.

    Returns
    -------
    None
        The selected initialization kernel is launched.

    Examples
    --------
    Use ``initialize(..., config={"mode": "ground"})`` to initialize the ground state.
    Use ``initialize(..., config={"mode": "initial", "vortex_number": 1.0})`` to initialize the default configuration.
    """
    cfg = config or {}

    mode = str(cfg.get("mode", "initial")).lower()
    if mode in ("ground", "uniform", "vacuum", "gs"):
        create_ground_state_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d)
        return

    vortex_number = float(cfg.get("vortex_number", 1.0))

    create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, vortex_number, p_i_d, p_f_d)