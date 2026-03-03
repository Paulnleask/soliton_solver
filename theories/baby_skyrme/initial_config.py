# =====================================================================================
# soliton_solver/theories/baby_skyrme/initial_config.py
# =====================================================================================
"""
Initial condition / ansatz CUDA kernels for baby_skyrme simulations.

Usage:
    - Use device helpers (position/angle/profile) inside initialization kernels.
    - Use create_ground_state_kernel to set a uniform magnetization state and zero integrator buffers.
    - Use create_initial_configuration_kernel to write a skyrmion ansatz and zero integrator buffers.
    - Use create_skyrmion_kernel to compose an additional skyrmion into an existing configuration.
    - Use initialize(...) as the theory entrypoint called by the simulation driver.

Parameters (conventions):
    - grid: flattened coordinate array:
        grid[idx_field(0, x, y, p_i)] -> physical x-coordinate
        grid[idx_field(1, x, y, p_i)] -> physical y-coordinate
    - Field: flattened state array with at least magnetization at components (0,1,2).
    - All kernels guard with in_bounds(x, y, p_i).

Outputs:
    - Writes initial magnetization configurations into Field.
    - Zeros auxiliary integrator buffers (Velocity, k1..k4, l1..l4, Temp) where applicable.
    - Optionally composes additional skyrmions in-place into Field.
"""

# ---------------- Imports ----------------
import math
from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds
from soliton_solver.theories.baby_skyrme.kernels import compute_norm_magnetization

# ---------------- Initial configuration functions ----------------
@cuda.jit(device=True, inline=True)
def position(xcent, ycent, x, y, grid, p_i):
    """
    Compute radial distance from a physical-space center to lattice point (x, y).

    Usage:
        r = position(xcent, ycent, x, y, grid, p_i)

    Parameters:
        xcent, ycent: Physical coordinates of the center.
        x, y: Lattice indices.
        grid: Flattened coordinate array (component 0 -> x, 1 -> y).
        p_i: Integer parameter array for indexing.

    Outputs:
        - Returns radial distance r = sqrt((x_phys - xcent)^2 + (y_phys - ycent)^2).
    """
    dx = grid[idx_field(0, x, y, p_i)]
    dy = grid[idx_field(1, x, y, p_i)]
    return math.sqrt((dx - xcent) * (dx - xcent) + (dy - ycent) * (dy - ycent))

@cuda.jit(device=True, inline=True)
def profile_function_magnetization(r, max_r):
    """
    Compute radial profile for magnetization polar angle in a skyrmion ansatz.

    Usage:
        fm = profile_function_magnetization(r, max_r)

    Parameters:
        r: Radial distance from skyrmion center.
        max_r: Characteristic profile radius.

    Outputs:
        - Returns profile value:
            * 0 outside r > max_r.
            * π exp(-(2r/max_r)^2) inside.
    """
    if r > max_r:
        return 0.0
    t = (2.0 * r / max_r)
    return math.pi * math.exp(-(t * t))

@cuda.jit(device=True, inline=True)
def angle(r, xcent, ycent, orientation, x, y, grid, p_i):
    """
    Compute azimuthal angle around a center with rotation offset.

    Usage:
        theta = angle(r, xcent, ycent, orientation, x, y, grid, p_i)

    Parameters:
        r: Radial distance.
        xcent, ycent: Physical center coordinates.
        orientation: Rotation offset (radians).
        x, y: Lattice indices.
        grid: Flattened coordinate array.
        p_i: Integer parameter array.

    Outputs:
        - Returns angle θ = -orientation + atan2(y - ycent, x - xcent),
        mapped to [0, 2π) before shifting.
    """
    theta = -orientation
    if r == 0.0:
        return theta
    gx = grid[idx_field(0, x, y, p_i)] - xcent
    gy = grid[idx_field(1, x, y, p_i)] - ycent
    ang = math.atan2(gy, gx)   # (-pi, pi]
    if ang < 0.0:
        ang += 2.0 * math.pi  # -> [0, 2pi)
    return theta + ang

# ---------------- Create ground state configuration kernel ----------------
@cuda.jit
def create_ground_state_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i, p_f):
    """
    Initialize uniform ground state and zero all integrator buffers.

    Usage:
        create_ground_state_kernel[grid2d, block2d](...)

    Parameters:
        Velocity: Device array of velocities.
        Field: Device state array (magnetization at components 0,1,2).
        grid: Coordinate array (unused).
        k1..k4, l1..l4, Temp: Integrator buffers.
        p_i, p_f: Device parameter arrays.

    Outputs:
        - Sets magnetization to (0, 0, 1) at all lattice sites.
        - Zeros Velocity, k1..k4, l1..l4, and Temp for all components.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Magnetization
    Field[idx_field(0, x, y, p_i)] = 0.0
    Field[idx_field(1, x, y, p_i)] = 0.0
    Field[idx_field(2, x, y, p_i)] = 1.0
    # Initalize the other fields
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

#  ---------------- Create initial configuration kernel ----------------
@cuda.jit
def create_initial_configuration_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, ansatz_bloch, ansatz_neel, ansatz_anti, skyrmion_rotation, vortex_number, p_i, p_f):
    """
    Initialize skyrmion ansatz configuration and zero integrator buffers.

    Usage:
        create_initial_configuration_kernel[grid2d, block2d](...)

    Parameters:
        Velocity, Field: Device arrays.
        grid: Coordinate array.
        k1..k4, l1..l4, Temp: Integrator buffers.
        ansatz_bloch, ansatz_neel, ansatz_anti: Boolean flags selecting ansatz type.
        skyrmion_rotation: Rotation offset in radians.
        p_i, p_f: Device parameter arrays.

    Outputs:
        - Constructs magnetization skyrmion (Bloch / Néel / anti).
        - Normalizes magnetization using compute_norm_magnetization().
        - Defaults to uniform (0,0,1) if no ansatz selected.
        - Zeros Velocity, k1..k4, l1..l4, and Temp for all components.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Parameters
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    skN = p_f[8]
    # Grid and profiles
    xcent = (grid[idx_field(0, 0, 0, p_i)] + grid[idx_field(0, xlen-1, ylen-1, p_i)]) / 2.0
    ycent = (grid[idx_field(1, 0, 0, p_i)] + grid[idx_field(1, xlen-1, ylen-1, p_i)]) / 2.0
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, skyrmion_rotation, x, y, grid, p_i)
    fm = profile_function_magnetization(r1, rmax / 10.0 * skN)
    # Magnetization
    if ansatz_bloch:
        Field[idx_field(0, x, y, p_i)] = -math.sin(fm) * math.sin(skN * th)
        Field[idx_field(1, x, y, p_i)] =  math.sin(fm) * math.cos(skN * th)
        Field[idx_field(2, x, y, p_i)] =  math.cos(fm)
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
    # Initalize the other fields
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

#  ---------------- Create skyrmion kernel ----------------
@cuda.jit
def create_skyrmion_kernel(Field, grid, pxi, pxj, skyrmion_rotation, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f):
    """
    Compose a skyrmion into the existing magnetization field at a chosen pixel center.

    Usage:
        create_skyrmion_kernel[grid2d, block2d](...)

    Parameters:
        Field: Device state array (magnetization at components 0,1,2).
        grid: Coordinate array.
        pxi, pxj: Pixel indices of skyrmion center.
        skyrmion_rotation: Rotation offset in radians.
        ansatz_bloch, ansatz_neel, ansatz_anti: Boolean ansatz selectors.
        p_i, p_f: Device parameter arrays.

    Outputs:
        - Builds skyrmion magnetization using selected ansatz.
        - Converts old and new magnetization to CP¹ coordinates.
        - Composes configurations via stereographic addition.
        - Maps result back to S².
        - Normalizes magnetization in-place.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Parameters
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    skN = p_f[8]
    # Pixel bounds check
    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen:
        return
    # Grid and profiles
    xcent = grid[idx_field(0, pxi, pxj, p_i)]
    ycent = grid[idx_field(1, pxi, pxj, p_i)]
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, skyrmion_rotation, x, y, grid, p_i)
    fm = profile_function_magnetization(r1, rmax / 10.0 * skN)
    eps = 1e-12
    den = max(1.0 + Field[idx_field(2, x, y, p_i)], eps)
    # Old stereographic coordinate W
    old_wr = Field[idx_field(0, x, y, p_i)] / den
    old_wi = Field[idx_field(1, x, y, p_i)] / den
    # Skyrmion ansatz
    if ansatz_bloch:
        m1 = -math.sin(fm) * math.sin(skN * th)
        m2 =  math.sin(fm) * math.cos(skN * th)
        m3 =  math.cos(fm)
    elif ansatz_neel:
        m1 =  math.sin(fm) * math.cos(skN * th)
        m2 =  math.sin(fm) * math.sin(skN * th)
        m3 =  math.cos(fm)
    elif ansatz_anti:
        m1 = -math.sin(fm) * math.sin(skN * th)
        m2 = -math.sin(fm) * math.cos(skN * th)
        m3 =  math.cos(fm)
    else:
        m1 = 0.0
        m2 = 0.0
        m3 = 1.0
    # New skyrmion
    int_den = max(1.0 + m3, eps)
    int_wr = m1 / int_den
    int_wi = m2 / int_den
    # Composition
    new_wr = old_wr + int_wr
    new_wi = old_wi + int_wi
    normW = new_wr * new_wr + new_wi * new_wi
    # Map from CP1 to S2
    Field[idx_field(0, x, y, p_i)] = 2.0 * new_wr / (1.0 + normW)
    Field[idx_field(1, x, y, p_i)] = 2.0 * new_wi / (1.0 + normW)
    Field[idx_field(2, x, y, p_i)] = (1.0 - normW) / (1.0 + normW)
    compute_norm_magnetization(Field, x, y, p_i, p_f)

# ---------------- Initialize initial configuration ----------------
def initialize(*, Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d, p_i_h=None, p_f_h=None, grid2d, block2d, config: dict | None = None):
    """
    Theory-controlled initialization entrypoint.

    Usage:
        initialize(..., config=dict(...))

    Parameters:
        Velocity, Field: Device arrays (modified in-place).
        grid: Coordinate array.
        k1..k4, l1..l4, Temp: Integrator buffers.
        p_i_d, p_f_d: Device parameter arrays.
        p_i_h, p_f_h: Optional host parameter arrays.
        grid2d, block2d: CUDA launch configuration.
        config: Optional dictionary controlling mode and ansatz selection.

    Outputs:
        - If mode is "ground": launches ground-state initialization.
        - Otherwise: launches skyrmion ansatz initialization.
        - Selects ansatz type from config (default: Bloch).
        - Modifies Field and zeros integrator buffers via selected kernel.
    """
    cfg = config or {}

    # Decide mode (theory-defined)
    mode = str(cfg.get("mode", "initial")).lower()
    # Example supported modes: "ground", "initial"
    # If unknown -> default to ground
    if mode in ("ground", "uniform", "vacuum", "gs"):
        create_ground_state_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d)
        return

    # Otherwise do the combined mixed soliton initial configuration
    ans = str(cfg.get("ansatz", "bloch")).lower()
    ansatz_bloch = (ans == "bloch")
    ansatz_neel  = (ans == "neel")
    ansatz_anti  = (ans == "anti")

    # If the user asked for something baby_skyrme doesn't know, fall back cleanly
    if not (ansatz_bloch or ansatz_neel or ansatz_anti):
        # You can choose to raise instead, but defaulting is often nicer
        ansatz_bloch = True

    skyrmion_rotation = float(cfg.get("skyrmion_rotation", 0.0))

    create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, ansatz_bloch, ansatz_neel, ansatz_anti, skyrmion_rotation, p_i_d, p_f_d)