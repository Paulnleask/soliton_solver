# =========================
# soliton_solver/theories/ginzburg_landau_superconductor/kernels.py
# =========================
"""
Core CUDA kernels and device helpers for the ginzburg_landau_superconductor model.

Usage:
- This module contains:
  1) Grid construction (physical coordinates stored in `grid`)
  2) Local energy density evaluation (device function) + a kernel wrapper
  3) Topological / diagnostic densities (skyrmion density, vortex/flux density, Higgs norm)
  4) Supercurrent computation from second derivatives
  5) The local energy-gradient evaluation used by relaxation / time stepping

Conventions:
- All field-like arrays are flattened with idx_field(a, x, y, p_i).
- Derivative buffers are flattened with idx_d1(mu, a, x, y, p_i) and idx_d2(mu, nu, a, x, y, p_i).
- Global kernels use a 2D CUDA grid and guard with in_bounds(x, y, p_i).
- Many kernels assume a fixed component layout for Field:
    Gauge:         a = 0,1   (A1, A2, A3)
    Higgs:         a = 2,3   (psi1, psi2)  (real/imag components)
  and that p_i carries counts such as number_total_fields, number_magnetization_fields, etc.
"""

# ---------------- Imports ----------------
import math
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds, launch_2d
from soliton_solver.core.integrator import make_do_gradient_step_kernel

# ---------------- Creat grid ----------------
@cuda.jit
def create_grid_kernel(grid, p_i, p_f):
    """
    Populate the physical coordinate grid array on the device.

    Usage:
    - Launch over (xlen, ylen).
    - Writes:
        grid[a=0, x, y] = lsx * x   (physical x coordinate)
        grid[a=1, x, y] = lsy * y   (physical y coordinate)
      where lsx, lsy are the lattice spacings stored in p_f.

    Parameters:
    - grid: device array (flattened) with at least 2 components (x and y coordinates)
    - p_i: integer params used by idx_field / in_bounds (includes xlen, ylen, halo, etc.)
    - p_f: float params; expects p_f[2]=lsx and p_f[3]=lsy
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    lsx = p_f[2]; lsy = p_f[3]
    grid[idx_field(0, x, y, p_i)] = lsx * float(x)
    grid[idx_field(1, x, y, p_i)] = lsy * float(y)

from numba import cuda
import math


@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """
    CUDA Python equivalent of the C++ __device__ compute_energy(...)
    Returns cell-integrated energy (multiplied by grid_volume).
    """

    # ---- parameters (match dparams.*) ----
    q = p_f[6]
    lam = p_f[7]        # lambda
    u1 = p_f[9]
    grid_volume = p_f[4]

    # ---- fields ----
    A1 = Field[idx_field(0, x, y, p_i)]
    A2 = Field[idx_field(1, x, y, p_i)]
    psi1 = Field[idx_field(2, x, y, p_i)]
    psi2 = Field[idx_field(3, x, y, p_i)]

    # ---- first derivatives of fields ----
    dpsi1dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]
    dpsi1dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dpsi2dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]
    dpsi2dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]

    dA1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]
    dA1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dA2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]
    dA2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]

    # ---- common scalars ----
    psi_sq = psi1*psi1 + psi2*psi2
    A_sq = A1*A1 + A2*A2

    # ---- energy ----
    energy = 0.0

    # 1/2 |∇ψ|²
    energy += 0.5*(dpsi1dx*dpsi1dx + dpsi1dy*dpsi1dy + dpsi2dx*dpsi2dx + dpsi2dy*dpsi2dy)

    # q A · (ψ × ∇ψ)
    energy += q*A1*(psi1*dpsi2dx - psi2*dpsi1dx)
    energy += q*A2*(psi1*dpsi2dy - psi2*dpsi1dy)

    # 1/2 q² A² |ψ|²
    energy += 0.5*q*q*A_sq*psi_sq

    # λ/8 (u₁² − |ψ|²)²
    energy += lam/8.0*(u1*u1 - psi_sq)*(u1*u1 - psi_sq)

    # 1/2 (∂ₓA₂ − ∂ᵧA₁)²
    curlA = dA2dx - dA1dy
    energy += 0.5*curlA*curlA

    # volume factor
    energy *= grid_volume
    return energy

# ---------------- Compute the energy kernel ----------------
@cuda.jit
def compute_energy_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute per-site energy contributions across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - For each (x, y):
        1) Compute first derivatives for all components (fills d1fd1x for that site).
        2) Store the local cell energy in en at component 0:
            en[a=0, x, y] = compute_energy_point(...)

    Parameters:
    - en: device array (flattened) used as an output scalar field (component 0)
    - Field: device state array
    - d1fd1x: device first-derivative buffer (written here, then consumed by compute_energy_point)
    - p_i: expects p_i[4]=number_total_fields
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_energy_point(Field, d1fd1x, x, y, p_i, p_f)


# ---------------- Compute the magnetic flux density ----------------
@cuda.jit(device=True)
def compute_vortex_density(d1fd1x, which, x, y, p_i, p_f):
    """
    Compute a local magnetic/flux density component derived from gauge-field derivatives.

    Usage:
    - Device helper used by compute_vortex_number_kernel() and compute_magnetic_field_kernel().
    - `which` selects the returned component:
        which==0: uses  ∂y A3
        which==1: uses -∂x A3
        else:     uses  ∂x A2 - ∂y A1   (curl-like out-of-plane component)
    - Scales by grid_volume / (2π q), yielding a per-cell contribution.

    Parameters:
    - d1fd1x: first derivative buffer (must contain gauge field derivatives)
    - which: integer selector for component
    - p_f: expects p_f[6]=q and p_f[4]=grid_volume
    """
    grid_volume = p_f[4]
    q = p_f[6]
    magneticField = d1fd1x[idx_d1(0, 1, x, y, p_i)] - d1fd1x[idx_d1(1, 0, x, y, p_i)]
    magneticField *= grid_volume / (2.0 * math.pi * q)
    return magneticField

# ---------------- Compute the magnetic flux kernel ----------------
@cuda.jit
def compute_vortex_number_kernel(en, Field, d1fd1x, which, p_i, p_f):
    """
    Compute per-site vortex/flux density contributions for a selected component.

    Usage:
    - Launch over (xlen, ylen).
    - Computes first derivatives only for gauge-field components, then writes:
        en[a=0, x, y] = compute_vortex_density(..., which, ...)

    Parameters:
    - which: selects the component as described in compute_vortex_density()
    - p_i: p_i[11]=number_gauge_fields
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[11]
    for a in range(number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)

# ---------------- Compute the Higgs norm density ----------------
@cuda.jit(device=True)
def compute_norm_higgs_point(Field, x, y, p_i):
    """
    Compute |psi|^2 at a single lattice site.

    Usage:
    - Device helper used by compute_norm_higgs_kernel().
    - Assumes Higgs components are stored at a=2 (psi1) and a=3 (psi2).
    """
    psi1 = Field[idx_field(2, x, y, p_i)]
    psi2 = Field[idx_field(3, x, y, p_i)]
    return psi1*psi1 + psi2*psi2

# ---------------- Compute the Higgs norm kernel ----------------
@cuda.jit
def compute_norm_higgs_kernel(en, Field, p_i):
    """
    Compute |psi|^2 across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - Writes output into en at component 0:
        en[a=0, x, y] = |psi(x,y)|^2
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    en[idx_field(0, x, y, p_i)] = compute_norm_higgs_point(Field, x, y, p_i)

# ---------------- Compute the local supercurrent ----------------
@cuda.jit(device=True)
def compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i):
    """
    Compute the local supercurrent vector from second derivatives of the gauge fields.

    Usage:
    - Device helper used by compute_supercurrent_kernel().
    - Zeros the first `number_magnetization_fields` components of Supercurrent at (x,y),
      then fills components 0,1,2 with expressions built from d2fd2x.

    Notes:
    - The specific combinations correspond to the model’s definition of current in terms of
      second derivatives (mixed and pure) of gauge components 3,4,5.
    """
    number_gauge_fields = p_i[11]
    for a in range(number_gauge_fields):
        Supercurrent[idx_field(a, x, y, p_i)] = 0.0
    # Compute the supercurrent
    Supercurrent[idx_field(0, x, y, p_i)] += d2fd2x[idx_d2(1, 0, 1, x, y, p_i)] - d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    Supercurrent[idx_field(1, x, y, p_i)] += d2fd2x[idx_d2(0, 1, 0, x, y, p_i)] - d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]

# ---------------- Compute the supercurrent kernel ----------------
@cuda.jit
def compute_supercurrent_kernel(Field, d2fd2x, Supercurrent, p_i, p_f):
    """
    Compute supercurrent across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - Computes second derivatives for gauge fields (components in the gauge block),
      then calls compute_supercurrent_point(...) to write Supercurrent at (x, y).

    Parameters:
    - Field: device state array
    - d2fd2x: device second-derivative buffer (written here for gauge components)
    - Supercurrent: device output array (vector field)
    - p_i: expects p_i[10]=number_magnetization_fields, p_i[12]=number_gauge_fields
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_gauge_fields = p_i[11]
    for a in range(number_gauge_fields):
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
    compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    CUDA Python equivalent of C++ __device__ do_gradient_step(...)

    Fields:
      0: A1
      1: A2
      2: psi1
      3: psi2
    """

    # ---- parameters (match dparams.*) ----
    q = p_f[6]
    lam = p_f[7]
    u1 = p_f[8]
    time_step = p_f[5]

    # ---- fields ----
    A1   = Field[idx_field(0, x, y, p_i)]
    A2   = Field[idx_field(1, x, y, p_i)]
    psi1 = Field[idx_field(2, x, y, p_i)]
    psi2 = Field[idx_field(3, x, y, p_i)]

    psi_sq = psi1*psi1 + psi2*psi2

    # ---- first derivatives ----
    dA1dx   = d1fd1x[idx_d1(0, 0, x, y, p_i)]
    dA1dy   = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dA2dx   = d1fd1x[idx_d1(0, 1, x, y, p_i)]
    dA2dy   = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dpsi1dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]
    dpsi1dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dpsi2dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]
    dpsi2dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]

    # ---- second derivatives ----
    d2A1d2y   = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    d2A2d2x   = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]
    d2psi1d2x = d2fd2x[idx_d2(0, 0, 2, x, y, p_i)]
    d2psi1d2y = d2fd2x[idx_d2(1, 1, 2, x, y, p_i)]
    d2psi2d2x = d2fd2x[idx_d2(0, 0, 3, x, y, p_i)]
    d2psi2d2y = d2fd2x[idx_d2(1, 1, 3, x, y, p_i)]
    d2A1dxdy  = d2fd2x[idx_d2(1, 0, 0, x, y, p_i)]
    d2A2dxdy  = d2fd2x[idx_d2(1, 0, 1, x, y, p_i)]

    # =========================
    # Energy gradient
    # =========================

    # A1
    g0 = (-d2A1d2y + d2A2dxdy + q*(psi1*dpsi2dx - psi2*dpsi1dx) + q*q*A1*psi_sq)
    # A2
    g1 = (-d2A2d2x + d2A1dxdy + q*(psi1*dpsi2dy - psi2*dpsi1dy) + q*q*A2*psi_sq)
    # psi1
    g2 = (-d2psi1d2x - d2psi1d2y + 2.0*q*(A1*dpsi2dx + A2*dpsi2dy) + q*psi2*(dA1dx + dA2dy) + q*q*(A1*A1 + A2*A2)*psi1 - lam/2.0*(u1*u1 - psi_sq)*psi1)
    # psi2
    g3 = (-d2psi2d2x - d2psi2d2y - 2.0*q*(A1*dpsi1dx + A2*dpsi1dy) - q*psi1*(dA1dx + dA2dy) + q*q*(A1*A1 + A2*A2)*psi2 - lam/2.0*(u1*u1 - psi_sq)*psi2)

    # ---- write gradient ----
    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1
    EnergyGradient[idx_field(2, x, y, p_i)] = g2
    EnergyGradient[idx_field(3, x, y, p_i)] = g3

    # Calculates the velocity at each lattice point
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)