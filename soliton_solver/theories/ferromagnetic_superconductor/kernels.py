# =========================
# soliton_solver/theories/ferromagnetic_superconductor/kernels.py
# =========================
"""
Core CUDA kernels and device helpers for the ferromagnetic_superconductor model.

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
    Magnetization: a = 0,1,2   (m1, m2, m3)
    Gauge:         a = 3,4,5   (A1, A2, A3)
    Higgs:         a = 6,7     (psi1, psi2)  (real/imag components)
  and that p_i carries counts such as number_total_fields, number_magnetization_fields, etc.
"""

# ---------------- Imports ----------------
import math
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds, launch_2d
from soliton_solver.core.integrator import make_do_gradient_step_kernel
from soliton_solver.core.integrator import make_do_rk4_kernel

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

# ---------------- Normalize the magnetization field ----------------
@cuda.jit(device=True)
def compute_norm_magnetization(Field, x, y, p_i, p_f):
    """
    Normalises the magnetization to be of fixed length at a lattice point (x, y).

    Usage:

    Parameters:
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

# ---------------- Project orthogonal to the magnetization field ----------------
@cuda.jit(device=True)
def project_orthogonal_magnetization(func, Field, x, y, p_i, p_f):
    number_magnetization_fields = p_i[10]
    M0 = p_f[17]
    lm = 0.0
    for a in range(number_magnetization_fields):
        lm += func[idx_field(a, x, y, p_i)] * Field[idx_field(a, x, y, p_i)] / M0
    for a in range(number_magnetization_fields):
        func[idx_field(a, x, y, p_i)] -= lm * Field[idx_field(a, x, y, p_i)] / M0

# compute_norm_magnetization and project_orthogonal_magnetization must be @cuda.jit(device=True)
do_rk4_kernel = make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization)

# ---------------- Compute the energy density ----------------
@cuda.jit(device=True)
def compute_energy_point(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the local energy contribution at a single lattice site (x, y).

    Usage:
    - Device function called from compute_energy_kernel().
    - Assumes first-derivative buffer d1fd1x has already been computed for all relevant fields
      at (x, y) using compute_derivative_first(...).
    - Returns the *cell-integrated* energy (multiplied by grid_volume).

    Notes:
    - This implements the model-specific energy density for the coupled magnetization + gauge + Higgs system.
    - At the end, subtracts a constant so that the homogeneous ground state has zero energy
      (normalization using alpha, beta, ha, hb, eta1).
    """
    # Parameters
    ha = p_f[7]; hb = p_f[8]; q = p_f[6]
    alpha = p_f[14]; beta = p_f[15]; gamma = p_f[16]
    eta1 = p_f[9]; eta2 = p_f[10]
    grid_volume = p_f[4]
    #  Define fields
    m1 = Field[idx_field(0, x, y, p_i)]; m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    A1 = Field[idx_field(3, x, y, p_i)]; A2 = Field[idx_field(4, x, y, p_i)]; A3 = Field[idx_field(5, x, y, p_i)]
    psi1 = Field[idx_field(6, x, y, p_i)]; psi2 = Field[idx_field(7, x, y, p_i)]
    # Define derivatives
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]; dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]; dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dA1dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]; dA1dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]; dA2dx = d1fd1x[idx_d1(0, 4, x, y, p_i)]; dA2dy = d1fd1x[idx_d1(1, 4, x, y, p_i)]; dA3dx = d1fd1x[idx_d1(0, 5, x, y, p_i)]; dA3dy = d1fd1x[idx_d1(1, 5, x, y, p_i)]
    dpsi1dx = d1fd1x[idx_d1(0, 6, x, y, p_i)]; dpsi1dy = d1fd1x[idx_d1(1, 6, x, y, p_i)]; dpsi2dx = d1fd1x[idx_d1(0, 7, x, y, p_i)]; dpsi2dy = d1fd1x[idx_d1(1, 7, x, y, p_i)]
    # Squared values
    psi_sq = psi1*psi1 + psi2*psi2
    m_sq = m1*m1 + m2*m2 + m3*m3
    A_sq = A1*A1 + A2*A2 + A3*A3
    # Compute energy
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
    # Normalize energy to ground state
    denom = (16.0*eta1*eta1 - 4.0*hb*beta)
    energy -= ((alpha*alpha*hb + ha*ha*beta - 4.0*ha*alpha*eta1) / denom)
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

# ---------------- Compute the skyrmion number density ----------------
@cuda.jit(device=True)
def compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f):
    """
    Compute the local skyrmion (topological) charge density at (x, y).

    Usage:
    - Device function called from compute_skyrmion_number_kernel().
    - Assumes first derivatives for magnetization components have already been computed into d1fd1x.

    Returns:
    - A per-cell contribution to skyrmion number (includes grid_volume factor).
    """
    M0 = p_f[17]
    grid_volume = p_f[4]
    # Define fields
    m1 = Field[idx_field(0, x, y, p_i)];m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    # Define derivatives
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    # cross = mx × my
    cx0 = dm2dx*dm3dy - dm3dx*dm2dy; cx1 = dm3dx*dm1dy - dm1dx*dm3dy; cx2 = dm1dx*dm2dy - dm2dx*dm1dy
    # Skyrmion charge density
    charge = m1*cx0 + m2*cx1 + m3*cx2
    return charge * (grid_volume / (4.0 * math.pi * M0 * M0 * M0))

# ---------------- Compute the skyrmion number kernel ----------------
@cuda.jit
def compute_skyrmion_number_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute per-site skyrmion charge density contributions across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - Computes first derivatives only for magnetization components (0..number_magnetization_fields-1),
      then writes the local charge density into en at component 0.

    Parameters:
    - en: device output scalar field (component 0) storing local skyrmion density contribution
    - Field: device state array
    - d1fd1x: device first-derivative buffer (written here for magnetization components)
    - p_i: expects p_i[10]=number_magnetization_fields
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_skyrmion_density(Field, d1fd1x, x, y, p_i, p_f)

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
    magneticField = 0.0
    if which == 0:
        magneticField += d1fd1x[idx_d1(1, 5, x, y, p_i)]
    elif which == 1:
        magneticField -= d1fd1x[idx_d1(0, 5, x, y, p_i)]
    else:
        magneticField += d1fd1x[idx_d1(0, 4, x, y, p_i)] - d1fd1x[idx_d1(1, 3, x, y, p_i)]
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
    - p_i: expects p_i[10]=number_magnetization_fields and p_i[12]=number_gauge_fields
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    number_gauge_fields = p_i[12]
    for a in range(number_magnetization_fields, number_magnetization_fields + number_gauge_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_vortex_density(d1fd1x, which, x, y, p_i, p_f)

# ---------------- Compute the magnetic field kernel ----------------
@cuda.jit
def compute_magnetic_field_kernel(Field, d1fd1x, MagneticFluxDensity, p_i, p_f):
    """
    Compute the magnetic/flux density vector field across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - Computes derivatives for all gauge components once, then writes
      MagneticFluxDensity[a=0..number_magnetization_fields-1] using compute_vortex_density().
      (Here the number of output components is tied to `number_magnetization_fields` by convention.)

    Parameters:
    - MagneticFluxDensity: device output array storing the vector components
    - p_i: expects p_i[10]=number_magnetization_fields and p_i[12]=number_gauge_fields
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

# ---------------- Compute the Higgs norm density ----------------
@cuda.jit(device=True)
def compute_norm_higgs_point(Field, x, y, p_i):
    """
    Compute |psi|^2 at a single lattice site.

    Usage:
    - Device helper used by compute_norm_higgs_kernel().
    - Assumes Higgs components are stored at a=6 (psi1) and a=7 (psi2).
    """
    psi1 = Field[idx_field(6, x, y, p_i)]
    psi2 = Field[idx_field(7, x, y, p_i)]
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
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        Supercurrent[idx_field(a, x, y, p_i)] = 0.0
    # Compute the supercurrent
    Supercurrent[idx_field(0, x, y, p_i)] += d2fd2x[idx_d2(1, 0, 4, x, y, p_i)] - d2fd2x[idx_d2(1, 1, 3, x, y, p_i)]
    Supercurrent[idx_field(1, x, y, p_i)] += d2fd2x[idx_d2(0, 1, 3, x, y, p_i)] - d2fd2x[idx_d2(0, 0, 4, x, y, p_i)]
    Supercurrent[idx_field(2, x, y, p_i)] -= d2fd2x[idx_d2(0, 0, 5, x, y, p_i)] + d2fd2x[idx_d2(1, 1, 5, x, y, p_i)]

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
    number_magnetization_fields = p_i[10]
    number_gauge_fields = p_i[12]
    for a in range(number_magnetization_fields, number_magnetization_fields + number_gauge_fields):
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
    compute_supercurrent_point(Supercurrent, d2fd2x, x, y, p_i)

# ---------------- Compute the local energy gradient ----------------
@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient ∂E/∂Field and set the local Velocity update direction.

    Usage:
    - Device helper used by higher-level kernels (e.g. do_gradient_step_kernel in the integrator module).
    - Assumes d1fd1x and d2fd2x have already been filled at (x, y) for all required components.
    - Computes gradient components g0..g7 corresponding to:
        0..2: magnetization (m1,m2,m3)
        3..5: gauge fields (A1,A2,A3)
        6..7: Higgs fields (psi1,psi2)
    - Writes EnergyGradient[a, x, y] = g_a
    - Projects magnetization gradient to be orthogonal to the magnetization constraint.
    - Sets Velocity[a, x, y] = -time_step * EnergyGradient[a, x, y] for all a.

    Parameters:
    - Velocity: device array updated in-place (descent direction scaled by time_step)
    - Field: device state array
    - EnergyGradient: device array receiving the gradient
    - d1fd1x: first-derivative buffer
    - d2fd2x: second-derivative buffer
    - p_f: uses p_f[5]=time_step and other model parameters (alpha,beta,gamma,ha,hb,q,eta1,eta2)
    """
    # unpack used params
    time_step = p_f[5]
    alpha = p_f[14]; beta = p_f[15]; gamma = p_f[16]
    ha = p_f[7]; hb = p_f[8]; q = p_f[6]
    eta1 = p_f[9]; eta2 = p_f[10]
    # Define fields
    m1 = Field[idx_field(0, x, y, p_i)]
    m2 = Field[idx_field(1, x, y, p_i)]
    m3 = Field[idx_field(2, x, y, p_i)]
    A1 = Field[idx_field(3, x, y, p_i)]
    A2 = Field[idx_field(4, x, y, p_i)]
    A3 = Field[idx_field(5, x, y, p_i)]
    psi1 = Field[idx_field(6, x, y, p_i)]
    psi2 = Field[idx_field(7, x, y, p_i)]
    # Define first derivatives
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    dA1dx = d1fd1x[idx_d1(0, 3, x, y, p_i)]; dA1dy = d1fd1x[idx_d1(1, 3, x, y, p_i)]
    dA2dx = d1fd1x[idx_d1(0, 4, x, y, p_i)]; dA2dy = d1fd1x[idx_d1(1, 4, x, y, p_i)]
    dA3dx = d1fd1x[idx_d1(0, 5, x, y, p_i)]; dA3dy = d1fd1x[idx_d1(1, 5, x, y, p_i)]
    dpsi1dx = d1fd1x[idx_d1(0, 6, x, y, p_i)]; dpsi1dy = d1fd1x[idx_d1(1, 6, x, y, p_i)]
    dpsi2dx = d1fd1x[idx_d1(0, 7, x, y, p_i)]; dpsi2dy = d1fd1x[idx_d1(1, 7, x, y, p_i)]
    # Define second derivatives
    d2m1d2x = d2fd2x[idx_d2(0, 0, 0, x, y, p_i)]; d2m1d2y = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]
    d2m2d2x = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]; d2m2d2y = d2fd2x[idx_d2(1, 1, 1, x, y, p_i)]
    d2m3d2x = d2fd2x[idx_d2(0, 0, 2, x, y, p_i)]; d2m3d2y = d2fd2x[idx_d2(1, 1, 2, x, y, p_i)]
    d2A1dxdy = d2fd2x[idx_d2(0, 1, 3, x, y, p_i)]; d2A1d2y = d2fd2x[idx_d2(1, 1, 3, x, y, p_i)]
    d2A2d2x  = d2fd2x[idx_d2(0, 0, 4, x, y, p_i)]; d2A2dxdy = d2fd2x[idx_d2(1, 0, 4, x, y, p_i)]
    d2A3d2x  = d2fd2x[idx_d2(0, 0, 5, x, y, p_i)]; d2A3d2y  = d2fd2x[idx_d2(1, 1, 5, x, y, p_i)]
    d2psi1d2x = d2fd2x[idx_d2(0, 0, 6, x, y, p_i)]; d2psi1d2y = d2fd2x[idx_d2(1, 1, 6, x, y, p_i)]
    d2psi2d2x = d2fd2x[idx_d2(0, 0, 7, x, y, p_i)]; d2psi2d2y = d2fd2x[idx_d2(1, 1, 7, x, y, p_i)]
    # Squared values
    psi_sq = psi1*psi1 + psi2*psi2
    m_sq = m1*m1 + m2*m2 + m3*m3
    dm_sq = (dm1dx*dm1dx + dm2dx*dm2dx + dm3dx*dm3dx + dm1dy*dm1dy + dm2dy*dm2dy + dm3dy*dm3dy)
    A_sq = A1*A1 + A2*A2 + A3*A3
    # Magnetization
    g0 = alpha*m1 + beta*m1*m_sq - gamma*gamma*d2m1d2x - gamma*gamma*d2m1d2y - dA3dy + 2.0*eta1*psi_sq*m1 - 2.0*eta2*(d2m1d2x + d2m1d2y)*psi_sq
    g1 = alpha*m2 + beta*m2*m_sq - gamma*gamma*d2m2d2x - gamma*gamma*d2m2d2y + dA3dx + 2.0*eta1*psi_sq*m2 - 2.0*eta2*(d2m2d2x + d2m2d2y)*psi_sq
    g2 = alpha*m3 + beta*m3*m_sq - gamma*gamma*d2m3d2x - gamma*gamma*d2m3d2y - dA2dx + dA1dy + 2.0*eta1*psi_sq*m3 - 2.0*eta2*(d2m3d2x + d2m3d2y)*psi_sq
    # Gauge field
    g3 = q*q*A1*psi_sq + q*(psi1*dpsi2dx - psi2*dpsi1dx) + d2A2dxdy - d2A1d2y - dm3dy
    g4 = q*q*A2*psi_sq + q*(psi1*dpsi2dy - psi2*dpsi1dy) + d2A1dxdy - d2A2d2x + dm3dx
    g5 = q*q*A3*psi_sq - d2A3d2x - d2A3d2y - dm2dx + dm1dy
    # Higgs field
    g6 = ha*psi1 + hb*psi1*psi_sq - d2psi1d2x - d2psi1d2y + q*q*A_sq*psi1 + 2.0*q*A1*dpsi2dx + 2.0*q*A2*dpsi2dy + q*psi2*(dA1dx + dA2dy) + 2.0*eta1*m_sq*psi1 + 2.0*eta2*psi1*psi1*dm_sq
    g7 = ha*psi2 + hb*psi2*psi_sq - d2psi2d2x - d2psi2d2y + q*q*A_sq*psi2 - 2.0*q*A1*dpsi1dx - 2.0*q*A2*dpsi1dy - q*psi1*(dA1dx + dA2dy) + 2.0*eta1*m_sq*psi2 + 2.0*eta2*psi2*psi2*dm_sq
    # Write the gradient
    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1
    EnergyGradient[idx_field(2, x, y, p_i)] = g2
    EnergyGradient[idx_field(3, x, y, p_i)] = g3
    EnergyGradient[idx_field(4, x, y, p_i)] = g4
    EnergyGradient[idx_field(5, x, y, p_i)] = g5
    EnergyGradient[idx_field(6, x, y, p_i)] = g6
    EnergyGradient[idx_field(7, x, y, p_i)] = g7
    # Project magnetization
    project_orthogonal_magnetization(EnergyGradient, Field, x, y, p_i, p_f)
    # Calculates the velocity at each lattice point
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)