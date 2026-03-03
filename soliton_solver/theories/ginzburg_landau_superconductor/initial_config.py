# =========================
# soliton_solver/theories/ginzburg_landau_superconductor/initial_config.py
# =========================
"""
Initial condition / ansatz CUDA kernels for the ginzburg_landau_superconductor simulation.

Usage:
- Provides small CUDA device helpers (distance/angle + profile functions) and several
  global kernels that write an initial configuration into `Field` (and zero auxiliary arrays).

Coordinate conventions:
- `grid` is a flattened array storing the physical coordinates at each lattice point:
    grid[idx_field(0, x, y, p_i)] -> physical x-coordinate
    grid[idx_field(1, x, y, p_i)] -> physical y-coordinate
- `Field` is a flattened array storing multiple physical fields per lattice point.
  This file assumes (by convention) at least:
    Gauge fields:  components 0,1   (A1, A2)
    Higgs fields:  components 2,3   (Re(psi), Im(psi))
    Potential field: component 4    (A0/V)
  (Any additional components are zeroed by the kernels.)

Typical workflow:
- Call `create_ground_state_kernel` to initialize a uniform ground state.
- Call `create_initial_configuration_kernel` to initialize a combined skyrmion + vortex ansatz.
- Use `create_vortex_kernel` and `create_skyrmion_kernel` to "paint" additional objects centered
  at a chosen pixel (pxi, pxj).

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
def profile_function_superconductor(r, max_r, value):
    """
    Radial profile for the superconducting/Higgs amplitude.

    Usage:
    - Used to build a smooth amplitude envelope fs(r).
    - For r > max_r: returns 1.0 (saturates to bulk value).
    - Otherwise: returns tanh(value * r) (ramps up from 0 at the core).

    Parameters:
    - r: radial distance
    - max_r: saturation radius beyond which profile is constant
    - value: controls core size / steepness of the tanh ramp
    """
    if r > max_r:
        return 1.0
    return math.tanh(value * r)

@cuda.jit(device=True, inline=True)
def angle(r, xcent, ycent, x, y, grid, p_i):
    """
    Compute the azimuthal angle theta around a physical-space center, with a rotation offset.

    Usage:
    - Device helper for vortex/skyrmion phase factors.
    - `orientation` is a rotation offset (in radians) applied as -orientation.
    - For r == 0: returns the base angle (-orientation) to avoid atan2(0,0).

    Returns:
    - theta = atan2(y - ycent, x - xcent) mapped to [0, 2π) and then shifted.

    Notes:
    - atan2 range is (-π, π]; we map negatives by adding 2π.
    """
    theta = 0.0
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
        * Magnetization: (m1,m2,m3) = (0, 0, M0)
        * Gauge fields:  A = 0
        * Higgs fields:  (Re,Im) = (u1, 0)
      where M0 and u1 are read from p_f.
    - Zeros per-site auxiliary arrays used by the time integrator:
        Velocity, k1..k4, l1..l4, Temp
      for all `a` in [0, number_total_fields).

    Parameters:
    - Velocity, k*, l*, Temp: flattened arrays with same indexing as Field
    - Field: flattened state array (multiple components per lattice point)
    - grid: coordinate array (unused here, but kept for signature consistency)
    - p_i: integer params (expects p_i[4] = number_total_fields)
    - p_f: float params (expects p_f[9]=u1)
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Ground state parameters
    u1 = p_f[8]
    # Gauge fields
    Field[idx_field(0, x, y, p_i)] = 0.0
    Field[idx_field(1, x, y, p_i)] = 0.0
    # Higgs fields
    Field[idx_field(2, x, y, p_i)] = u1
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
def create_initial_configuration_kernel(Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, vortex_number, p_i, p_f):
    """
    Initialize a combined skyrmion + vortex ansatz (and zero integrator buffers).

    Usage:
    - Launch over the full lattice (2D grid of threads).
    - Builds:
        * Magnetization skyrmion (Bloch / Néel / anti variants) using `profile_function_magnetization`
          and an azimuthal angle from `angle()`.
        * Gauge field with a vortex-like 1/r angular structure scaled by `ainf` and the SC profile `fs`.
        * Higgs field (complex order parameter) with phase winding `vortex_number`.
    - Normalizes magnetization (if a skyrmion ansatz is used) via compute_norm_magnetization().
    - Zeros Velocity/k*/l*/Temp for all components.

    Parameters / flags:
    - ansatz_bloch, ansatz_neel, ansatz_anti: booleans selecting the skyrmion type (first match wins).
      If none are True, magnetization defaults to (0,0,1).
    - skyrmion_rotation: rotation offset in radians for the skyrmion angle.
    - vortex_number: integer winding number used in the Higgs phase factor.
    - p_f expects:
        p_f[0]=xsize, p_f[1]=ysize, p_f[11]=u1, p_f[13]=ainf, p_f[18]=skN (skyrmion number)
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Parameters
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    ainf = p_f[11]
    u1 = p_f[8]
    # Grid and profiles
    xcent = (grid[idx_field(0, 0, 0, p_i)] + grid[idx_field(0, xlen-1, ylen-1, p_i)]) / 2.0
    ycent = (grid[idx_field(1, 0, 0, p_i)] + grid[idx_field(1, xlen-1, ylen-1, p_i)]) / 2.0
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, x, y, grid, p_i)
    fs = profile_function_superconductor(r1, rmax, 0.2)
    # Gauge fields
    eps = 1e-12
    rr = r1 if r1 > eps else eps
    Field[idx_field(0, x, y, p_i)] = ainf * fs * math.sin(th) / rr
    Field[idx_field(1, x, y, p_i)] = -ainf * fs * math.cos(th) / rr
    # Higgs fields
    Field[idx_field(2, x, y, p_i)] = u1 * fs * math.cos(vortex_number * th)
    Field[idx_field(3, x, y, p_i)] = u1 * fs * math.sin(vortex_number * th)
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

#  ---------------- Create vortex kernel ----------------
@cuda.jit
def create_vortex_kernel(Field, grid, pxi, pxj, vortex, p_i, p_f):
    """
    Multiply the existing Higgs field by an (anti-)vortex phase factor centered at pixel (pxi, pxj).

    Usage:
    - Launch over the full lattice (2D grid of threads).
    - `pxi, pxj` specify the vortex center in *pixel indices* (not physical coords).
      The kernel converts that to physical center via `grid`.
    - If `vortex` is True, uses a negative sign in the phase (anti-vortex vs vortex as coded).
    - Applies the vortex by complex-multiplying the current Higgs field by an "internal" vortex field,
      then dividing by u1 to keep the bulk amplitude consistent:
        new = (old * int) / u1

    Parameters:
    - Field: flattened state array; Higgs fields are assumed at components 6 (real) and 7 (imag)
    - grid: coordinate array used to locate the physical center at (pxi, pxj)
    - pxi, pxj: integer pixel indices of the vortex center
    - vortex: boolean controlling the sign of the phase
    - p_f expects p_f[9]=u1 and p_f[10]=vortex_number (winding)
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    # Parameters
    xlen = p_i[0]; ylen = p_i[1]
    xsize = p_f[0]; ysize = p_f[1]
    u1 = p_f[8]
    vortex_number = p_f[9]
    # Pixel bounds check
    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen:
        return
    # Center in physical coords
    xcent = grid[idx_field(0, pxi, pxj, p_i)]
    ycent = grid[idx_field(1, pxi, pxj, p_i)]
    r1 = position(xcent, ycent, x, y, grid, p_i)
    rmax = xsize if xsize > ysize else ysize
    th = angle(r1, xcent, ycent, x, y, grid, p_i)
    fs = profile_function_superconductor(r1, rmax, 0.2)
    # Old Higgs field
    old_r = Field[idx_field(2, x, y, p_i)]
    old_i = Field[idx_field(3, x, y, p_i)]
    # New Higgs field
    phase = (-1.0 if vortex else 1.0) * th
    int_r = u1 * fs * math.cos(vortex_number * phase)
    int_i = u1 * fs * math.sin(vortex_number * phase)
    # Complex composition: (old * int) / u1
    new_r = (old_r * int_r - old_i * int_i) / u1
    new_i = (old_r * int_i + old_i * int_r) / u1
    Field[idx_field(2, x, y, p_i)] = new_r
    Field[idx_field(3, x, y, p_i)] = new_i

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

    vortex_number = float(cfg.get("vortex_number", 1.0))

    create_initial_configuration_kernel[grid2d, block2d](Velocity, Field, grid, k1, k2, k3, k4, l1, l2, l3, l4, Temp, vortex_number, p_i_d, p_f_d)