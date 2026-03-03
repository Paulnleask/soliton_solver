# =========================
# soliton_solver/theories/liquid_crystal/initial_config.py
# =========================
"""
Initial condition / ansatz CUDA kernels for the liquid_crystal simulation.

Usage:
- Provides small CUDA device helpers (distance/angle + profile functions) and several
  global kernels that write an initial configuration into `Field` (and zero auxiliary arrays).

Coordinate conventions:
- `grid` is a flattened array storing the physical coordinates at each lattice point:
    grid[idx_field(0, x, y, p_i)] -> physical x-coordinate
    grid[idx_field(1, x, y, p_i)] -> physical y-coordinate
- `Field` is a flattened array storing multiple physical fields per lattice point.
  This file assumes (by convention) at least:
    Magnetization: components 0,1,2  (m1, m2, m3)
  (Any additional components are zeroed by the kernels.)

Typical workflow:
- Call `create_ground_state_kernel` to initialize a uniform ground state.
- Call `create_initial_configuration_kernel` to initialize skyrmion ansatz.
- Use `create_skyrmion_kernel` to "paint" additional objects centered at a chosen pixel (pxi, pxj).

Boundary handling:
- All global kernels use `in_bounds(x, y, p_i)` and return early out-of-domain.

Notes:
- The docstring at the top of this file previously mentioned derivatives; this module is
  actually for initial conditions / ansätze.
"""

# ---------------- Imports ----------------
import math
from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds
from soliton_solver.theories.liquid_crystal.kernels import compute_norm_magnetization

# ---------------- Initial configuration functions ----------------
@cuda.jit(device=True, inline=True)
def position(xcent, ycent, x, y, grid, p_i):
    """
    Compute radial distance r from a physical-space center to lattice point (x, y).

    Usage:
    - Device helper for kernels building radial profiles (vortex/skyrmion).
    - `xcent, ycent` are physical coordinates (same units as `grid`).
    - Returns r = sqrt((x_phys-xcent)^2 + (y_phys-ycent)^2).

    Parameters:
    - grid: flattened coordinate array; grid has components 0->x, 1->y
    - p_i: integer parameter array used by idx_field
    """
    dx = grid[idx_field(0, x, y, p_i)]
    dy = grid[idx_field(1, x, y, p_i)]
    return math.sqrt((dx - xcent) * (dx - xcent) + (dy - ycent) * (dy - ycent))

@cuda.jit(device=True, inline=True)
def profile_function_magnetization(r, max_r):
    """
    Radial profile for the magnetization polar angle (used in skyrmion ansätze).

    Usage:
    - Returns 0 outside radius max_r (no perturbation).
    - Inside: returns pi * exp(-(2r/max_r)^2), which is near pi at r=0 and decays
      smoothly toward 0 with radius.

    Parameters:
    - r: radial distance
    - max_r: characteristic radius for the skyrmion profile
    """
    if r > max_r:
        return 0.0
    t = (2.0 * r / max_r)
    return math.pi * math.exp(-(t * t))

@cuda.jit(device=True, inline=True)
def angle(r, xcent, ycent, orientation, x, y, grid, p_i):
    """
    Compute the azimuthal angle theta around a physical-space center, with a rotation offset.

    Usage:
    - Device helper for vortex/skyrmion phase factors.
    - `orientation` is a rotation offset (in radians) applied as -orientation.
    - For r == 0: returns the base angle (-orientation) to avoid atan2(0,0).

    Returns:
    - theta = -orientation + atan2(y - ycent, x - xcent) mapped to [0, 2π) and then shifted.

    Notes:
    - atan2 range is (-π, π]; we map negatives by adding 2π.
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
    Initialize a uniform ground-state configuration and zero all auxiliary arrays.

    Usage:
    - Launch over the full lattice (2D grid of threads).
    - Sets:
        * Magnetization: (m1,m2,m3) = (0, 0, 1)
    - Zeros per-site auxiliary arrays used by the time integrator:
        Velocity, k1..k4, l1..l4, Temp
      for all `a` in [0, number_total_fields).

    Parameters:
    - Velocity, k*, l*, Temp: flattened arrays with same indexing as Field
    - Field: flattened state array (multiple components per lattice point)
    - grid: coordinate array (unused here, but kept for signature consistency)
    - p_i: integer params (expects p_i[4] = number_total_fields)
    - p_f: float params
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Magnetization
    Field[idx_field(0, x, y, p_i)] = 0.0
    Field[idx_field(1, x, y, p_i)] = 0.0
    Field[idx_field(2, x, y, p_i)] = 1.0
    Field[idx_field(3, x, y, p_i)] = 0.0
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
    Initialize skyrmion ansatz (and zero integrator buffers).

    Usage:
    - Launch over the full lattice (2D grid of threads).
    - Builds:
        * Magnetization skyrmion (Bloch / Néel / anti variants) using `profile_function_magnetization`
          and an azimuthal angle from `angle()`.
    - Normalizes magnetization (if a skyrmion ansatz is used) via compute_norm_magnetization().
    - Zeros Velocity/k*/l*/Temp for all components.

    Parameters / flags:
    - ansatz_bloch, ansatz_neel, ansatz_anti: booleans selecting the skyrmion type (first match wins).
      If none are True, magnetization defaults to (0,0,1).
    - skyrmion_rotation: rotation offset in radians for the skyrmion angle.
    - p_f expects:
        p_f[0]=xsize, p_f[1]=ysize, p_f[9]=skN (skyrmion number)
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Parameters
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    skN = p_f[11]
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
        Field[idx_field(3, x, y, p_i)] = 0.0
        compute_norm_magnetization(Field, x, y, p_i, p_f)
    elif ansatz_neel:
        Field[idx_field(0, x, y, p_i)] = math.sin(fm) * math.cos(skN * th)
        Field[idx_field(1, x, y, p_i)] = math.sin(fm) * math.sin(skN * th)
        Field[idx_field(2, x, y, p_i)] = math.cos(fm)
        Field[idx_field(3, x, y, p_i)] = 0.0
        compute_norm_magnetization(Field, x, y, p_i, p_f)
    elif ansatz_anti:
        Field[idx_field(0, x, y, p_i)] = -math.sin(fm) * math.sin(skN * th)
        Field[idx_field(1, x, y, p_i)] = -math.sin(fm) * math.cos(skN * th)
        Field[idx_field(2, x, y, p_i)] = math.cos(fm)
        Field[idx_field(3, x, y, p_i)] = 0.0
        compute_norm_magnetization(Field, x, y, p_i, p_f)
    else:
        Field[idx_field(0, x, y, p_i)] = 0.0
        Field[idx_field(1, x, y, p_i)] = 0.0
        Field[idx_field(2, x, y, p_i)] = 1.0
        Field[idx_field(3, x, y, p_i)] = 0.0
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
    Add (compose) a skyrmion into the existing magnetization field, centered at pixel (pxi, pxj).

    Usage:
    - Launch over the full lattice (2D grid of threads).
    - Builds a skyrmion magnetization (m1,m2,m3) using the chosen ansatz type and profile fm(r).
    - Converts both the old magnetization and the skyrmion ansatz to stereographic CP1 coordinate W,
      adds them (simple composition), then maps back to S^2.
    - Normalizes magnetization via compute_norm_magnetization() at the end.

    Parameters:
    - Field: flattened state array; magnetization is assumed at components 0,1,2
    - grid: coordinate array used to locate the physical center at (pxi, pxj)
    - pxi, pxj: integer pixel indices of the skyrmion center
    - skyrmion_rotation: rotation offset in radians for the skyrmion angle
    - ansatz_bloch / ansatz_neel / ansatz_anti: booleans selecting the skyrmion type
    - p_f expects p_f[9]=skN (skyrmion number)
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Parameters
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    skN = p_f[11]
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

def initialize(*, Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, p_i_d, p_f_d, p_i_h=None, p_f_h=None, grid2d, block2d, config: dict | None = None):
    """
    Theory-controlled initialization entrypoint.

    Simulation is theory-agnostic and calls ONLY this function.

    `config` is an opaque dict; this theory decides which keys matter.
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

    # If the user asked for something superferro doesn't know, fall back cleanly
    if not (ansatz_bloch or ansatz_neel or ansatz_anti):
        # You can choose to raise instead, but defaulting is often nicer
        ansatz_bloch = True

    skyrmion_rotation = float(cfg.get("skyrmion_rotation", 0.0))

    create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, ansatz_bloch, ansatz_neel, ansatz_anti, skyrmion_rotation, p_i_d, p_f_d)