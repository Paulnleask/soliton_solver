"""
Initial condition kernels for the spin triplet superconducting ferromagnet theory.

Examples
--------
Use ``create_ground_state_kernel`` to initialize the uniform ground state.
Use ``create_initial_configuration_kernel`` to initialize the mixed soliton ansatz.
"""
import math
from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds
from soliton_solver.theories.spin_triplet_superconducting_magnet.kernels import compute_norm_magnetization

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
    Compute the superconducting amplitude profile.

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
    Use ``fs = profile_function_superconductor(r, max_r, value)`` to evaluate the condensate profile.
    """
    if r > max_r:
        return 1.0
    return math.tanh(value * r)

@cuda.jit(device=True, inline=True)
def profile_function_magnetization(r, max_r):
    """
    Compute the magnetization profile used in skyrmion ansatz fields.

    Parameters
    ----------
    r : float
        Radial distance from the center.
    max_r : float
        Characteristic profile radius.

    Returns
    -------
    float
        Polar angle profile value.

    Examples
    --------
    Use ``fm = profile_function_magnetization(r, max_r)`` to evaluate the skyrmion profile.
    """
    if r > max_r:
        return 0.0
    t = (2.0 * r / max_r)
    return math.pi * math.exp(-(t * t))

@cuda.jit(device=True, inline=True)
def angle(r, xcent, ycent, orientation, x, y, grid, p_i):
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
    orientation : float
        Rotation offset in radians.
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
        Azimuthal angle shifted by ``orientation``.

    Examples
    --------
    Use ``th = angle(r, xcent, ycent, orientation, x, y, grid, p_i)`` to build phase factors.
    """
    theta = -orientation
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
        The ground state and auxiliary buffers are written in place.

    Examples
    --------
    Launch ``create_ground_state_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i, p_f)`` to initialize the ground state.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    M0 = p_f[20]; u1 = p_f[11]; u2 = p_f[12]
    Field[idx_field(0, x, y, p_i)] = 0.0
    Field[idx_field(1, x, y, p_i)] = 0.0
    Field[idx_field(2, x, y, p_i)] = M0
    Field[idx_field(3, x, y, p_i)] = 0.0
    Field[idx_field(4, x, y, p_i)] = 0.0
    Field[idx_field(5, x, y, p_i)] = 0.0
    Field[idx_field(6, x, y, p_i)] = u1
    Field[idx_field(7, x, y, p_i)] = 0.0
    Field[idx_field(8, x, y, p_i)] = u2
    Field[idx_field(9, x, y, p_i)] = 0.0
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
def create_initial_configuration_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, ansatz_bloch, ansatz_neel, ansatz_anti, skyrmion_rotation, vortex_number, p_i, p_f):
    """
    Initialize the mixed skyrmion and vortex ansatz and zero the auxiliary buffers.

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
    ansatz_bloch : bool
        Flag selecting the Bloch ansatz.
    ansatz_neel : bool
        Flag selecting the Néel ansatz.
    ansatz_anti : bool
        Flag selecting the anti-skyrmion ansatz.
    skyrmion_rotation : float
        Rotation offset in radians.
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
    Launch ``create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, ansatz_bloch, ansatz_neel, ansatz_anti, skyrmion_rotation, vortex_number, p_i, p_f)`` to initialize the mixed soliton state.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    M0 = p_f[20]; u1 = p_f[11]; u2 = p_f[12]
    skN = p_f[21]
    ainf = p_f[16]; vortex1_number = p_f[13]; vortex2_number = p_f[14]
    xcent = (grid[idx_field(0, 0, 0, p_i)] + grid[idx_field(0, xlen-1, ylen-1, p_i)]) / 2.0
    ycent = (grid[idx_field(1, 0, 0, p_i)] + grid[idx_field(1, xlen-1, ylen-1, p_i)]) / 2.0
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, skyrmion_rotation, x, y, grid, p_i)
    fm = profile_function_magnetization(r1, rmax / 10.0)
    fs = profile_function_superconductor(r1, rmax, 0.2)
    if ansatz_bloch:
        Field[idx_field(0, x, y, p_i)] = -math.sin(fm) * math.sin(skN * th)
        Field[idx_field(1, x, y, p_i)] = math.sin(fm) * math.cos(skN * th)
        Field[idx_field(2, x, y, p_i)] = math.cos(fm)
        compute_norm_magnetization(Field, x, y, p_i, p_f)
    elif ansatz_neel:
        Field[idx_field(0, x, y, p_i)] = math.sin(fm) * math.cos(skN * th)
        Field[idx_field(1, x, y, p_i)] = math.sin(fm) * math.sin(skN * th)
        Field[idx_field(2, x, y, p_i)] = math.cos(fm)
        compute_norm_magnetization(Field, x, y, p_i, p_f)
    elif ansatz_anti:
        Field[idx_field(0, x, y, p_i)] = -math.sin(fm) * math.sin(skN * th)
        Field[idx_field(1, x, y, p_i)] = -math.sin(fm) * math.cos(skN * th)
        Field[idx_field(2, x, y, p_i)] = math.cos(fm)
        compute_norm_magnetization(Field, x, y, p_i, p_f)
    else:
        Field[idx_field(0, x, y, p_i)] = 0.0
        Field[idx_field(1, x, y, p_i)] = 0.0
        Field[idx_field(2, x, y, p_i)] = 1.0
    eps = 1e-12
    rr = r1 if r1 > eps else eps
    Field[idx_field(3, x, y, p_i)] = -ainf * fs * math.sin(th) / rr
    Field[idx_field(4, x, y, p_i)] = ainf * fs * math.cos(th) / rr
    Field[idx_field(5, x, y, p_i)] = 0.0
    Field[idx_field(6, x, y, p_i)] = u1 * fs * math.cos(vortex1_number * th)
    Field[idx_field(7, x, y, p_i)] = u1 * fs * math.sin(vortex1_number * th)
    Field[idx_field(8, x, y, p_i)] = u2 * fs * math.cos(vortex2_number * th)
    Field[idx_field(9, x, y, p_i)] = u2 * fs * math.sin(vortex2_number * th)
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
    Multiply the Higgs fields by a vortex or antivortex phase factor.

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
        The Higgs fields are updated in place.

    Examples
    --------
    Launch ``create_vortex_kernel[grid2d, block2d](Field, grid, pxi, pxj, vortex, p_i, p_f)`` to imprint a vortex.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    u1 = p_f[11]; u2 = p_f[12]
    vortex_number = p_f[15]
    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen:
        return
    xcent = grid[idx_field(0, pxi, pxj, p_i)]
    ycent = grid[idx_field(1, pxi, pxj, p_i)]
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, 0.0, x, y, grid, p_i)
    fs = profile_function_superconductor(r1, rmax, 0.2)
    old_r1 = Field[idx_field(6, x, y, p_i)]
    old_i1 = Field[idx_field(7, x, y, p_i)]
    phase = (-1.0 if vortex else 1.0) * th
    int_r1 = u1 * fs * math.cos(vortex_number * phase)
    int_i1 = u1 * fs * math.sin(vortex_number * phase)
    new_r1 = (old_r1 * int_r1 - old_i1 * int_i1) / u1
    new_i1 = (old_r1 * int_i1 + old_i1 * int_r1) / u1
    Field[idx_field(6, x, y, p_i)] = new_r1
    Field[idx_field(7, x, y, p_i)] = new_i1
    old_r2 = Field[idx_field(8, x, y, p_i)]
    old_i2 = Field[idx_field(9, x, y, p_i)]
    phase = (-1.0 if vortex else 1.0) * th
    int_r2 = u2 * fs * math.cos(vortex_number * phase)
    int_i2 = u2 * fs * math.sin(vortex_number * phase)
    new_r2 = (old_r2 * int_r2 - old_i2 * int_i2) / u2
    new_i2 = (old_r2 * int_i2 + old_i2 * int_r2) / u2
    Field[idx_field(8, x, y, p_i)] = new_r2
    Field[idx_field(9, x, y, p_i)] = new_i2

@cuda.jit
def create_skyrmion_kernel(Field, grid, pxi, pxj, skyrmion_rotation, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f):
    """
    Compose a skyrmion into the magnetization field.

    Parameters
    ----------
    Field : device array
        State field buffer.
    grid : device array
        Flattened coordinate array.
    pxi : int
        Pixel index of the skyrmion center along the x direction.
    pxj : int
        Pixel index of the skyrmion center along the y direction.
    skyrmion_rotation : float
        Rotation offset in radians.
    ansatz_bloch : bool
        Flag selecting the Bloch ansatz.
    ansatz_neel : bool
        Flag selecting the Néel ansatz.
    ansatz_anti : bool
        Flag selecting the anti-skyrmion ansatz.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The magnetization field is updated in place.

    Examples
    --------
    Launch ``create_skyrmion_kernel[grid2d, block2d](Field, grid, pxi, pxj, skyrmion_rotation, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f)`` to imprint a skyrmion.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    skN = p_f[21]
    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen:
        return
    xcent = grid[idx_field(0, pxi, pxj, p_i)]
    ycent = grid[idx_field(1, pxi, pxj, p_i)]
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, skyrmion_rotation, x, y, grid, p_i)
    if skN == 0:
        fm = profile_function_magnetization(r1, rmax / 10.0 * 2.0)
    else:
        fm = profile_function_magnetization(r1, rmax / 10.0 * skN)
    eps = 1e-12
    den = max(1.0 + Field[idx_field(2, x, y, p_i)], eps)
    old_wr = Field[idx_field(0, x, y, p_i)] / den
    old_wi = Field[idx_field(1, x, y, p_i)] / den
    if skN == 0:
        if ansatz_bloch:
            m1 = -math.sin(2.0 * fm) * math.sin(th)
            m2 = math.sin(2.0 * fm) * math.cos(th)
            m3 = math.cos(2.0 * fm)
        elif ansatz_neel:
            m1 = math.sin(2.0 * fm) * math.cos(th)
            m2 = math.sin(2.0 * fm) * math.sin(th)
            m3 = math.cos(2.0 * fm)
        elif ansatz_anti:
            m1 = -math.sin(2.0 * fm) * math.sin(th)
            m2 = -math.sin(2.0 * fm) * math.cos(th)
            m3 = math.cos(2.0 * fm)
        else:
            m1 = 0.0
            m2 = 0.0
            m3 = 1.0
    else:
        if ansatz_bloch:
            m1 = -math.sin(fm) * math.sin(skN * th)
            m2 = math.sin(fm) * math.cos(skN * th)
            m3 = math.cos(fm)
        elif ansatz_neel:
            m1 = math.sin(fm) * math.cos(skN * th)
            m2 = math.sin(fm) * math.sin(skN * th)
            m3 = math.cos(fm)
        elif ansatz_anti:
            m1 = -math.sin(fm) * math.sin(skN * th)
            m2 = -math.sin(fm) * math.cos(skN * th)
            m3 = math.cos(fm)
        else:
            m1 = 0.0
            m2 = 0.0
            m3 = 1.0
    int_den = max(1.0 + m3, eps)
    int_wr = m1 / int_den
    int_wi = m2 / int_den
    new_wr = old_wr + int_wr
    new_wi = old_wi + int_wi
    normW = new_wr * new_wr + new_wi * new_wi
    Field[idx_field(0, x, y, p_i)] = 2.0 * new_wr / (1.0 + normW)
    Field[idx_field(1, x, y, p_i)] = 2.0 * new_wi / (1.0 + normW)
    Field[idx_field(2, x, y, p_i)] = (1.0 - normW) / (1.0 + normW)
    compute_norm_magnetization(Field, x, y, p_i, p_f)

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
    Use ``initialize(..., config={"mode": "initial", "ansatz": "bloch"})`` to initialize the mixed soliton configuration.
    """
    cfg = config or {}

    mode = str(cfg.get("mode", "initial")).lower()
    if mode in ("ground", "uniform", "vacuum", "gs"):
        create_ground_state_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d)
        return

    ans = str(cfg.get("ansatz", "bloch")).lower()
    ansatz_bloch = (ans == "bloch")
    ansatz_neel = (ans == "neel")
    ansatz_anti = (ans == "anti")

    if not (ansatz_bloch or ansatz_neel or ansatz_anti):
        ansatz_bloch = True

    skyrmion_rotation = float(cfg.get("skyrmion_rotation", 0.0))
    vortex_number = float(cfg.get("vortex_number", 1.0))

    create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, ansatz_bloch, ansatz_neel, ansatz_anti, skyrmion_rotation, vortex_number, p_i_d, p_f_d)