# =====================================================================================
# soliton_solver/theories/baby_skyrme/kernels.py
# =====================================================================================
"""
Core CUDA kernels and device helpers for the baby Skyrme model.

Usage:
- This module contains:
  1) Grid construction (physical coordinates stored in `grid`)
  2) Local energy density evaluation (device function) + a kernel wrapper
  3) Topological / diagnostic densities (skyrmion density)
  4) The local energy-gradient evaluation used by relaxation / time stepping

Conventions:
- All field-like arrays are flattened with idx_field(a, x, y, p_i).
- Derivative buffers are flattened with idx_d1(mu, a, x, y, p_i) and idx_d2(mu, nu, a, x, y, p_i).
- Global kernels use a 2D CUDA grid and guard with in_bounds(x, y, p_i).
- Many kernels assume a fixed component layout for Field:
    Magnetization: a = 0,1,2   (m1, m2, m3)
  and that p_i carries counts such as number_total_fields, number_magnetization_fields, etc.
"""

# ---------------- Imports ----------------
import math
from math import sin, cos, atan2, sqrt
from numba import cuda
from soliton_solver.core.derivatives import compute_derivative_first
from soliton_solver.core.utils import idx_field, idx_d1, idx_d2, in_bounds
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

# ---------------- Project orthogonal to the magnetization field ----------------
@cuda.jit(device=True)
def project_orthogonal_magnetization(func, Field, x, y, p_i, p_f):
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
    Compute the local energy contribution at a single lattice site (x, y).

    Usage:
    - Device function called from compute_energy_kernel().
    - Assumes first-derivative buffer d1fd1x has already been computed for all relevant fields
      at (x, y) using compute_derivative_first(...).
    - Returns the *cell-integrated* energy (multiplied by grid_volume).

    Notes:
    - This implements the model-specific energy density for the baby Skyrme model.
    - The energy is normalized such that the homogeneous ground state has zero energy.
    """
    # Parameters
    grid_volume = p_f[4]
    mpi = p_f[6]; mpi2 = mpi * mpi
    kappa = p_f[7]; kappa2 = kappa * kappa
    # Fields
    m1 = Field[idx_field(0, x, y, p_i)]; m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    # Derivatives
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    # Sigma-model term
    grad_x_sq = dm1dx*dm1dx + dm2dx*dm2dx + dm3dx*dm3dx
    grad_y_sq = dm1dy*dm1dy + dm2dy*dm2dy + dm3dy*dm3dy
    energy = 0.5 * (grad_x_sq + grad_y_sq)
    # Skyrme term: cross = ∂x m × ∂y m
    cx = dm2dx*dm3dy - dm3dx*dm2dy; cy = dm3dx*dm1dy - dm1dx*dm3dy; cz = dm1dx*dm2dy - dm2dx*dm1dy
    cross_sq = cx*cx + cy*cy + cz*cz
    energy += 0.5 * kappa2 * cross_sq
    # Potentials
    Potential_Standard = p_i[11]
    Potential_Holomorphic = p_i[12]
    Potential_EasyPlane = p_i[13]
    Potential_Dihedral2 = p_i[14]
    Potential_Aloof = p_i[15]
    Potential_Dihedral3 = p_i[16]
    Potential_Broken = p_i[17]; N = p_i[19]
    Potential_DoubleVacua = p_i[18]
    # Standard
    if Potential_Standard:
        energy += mpi2 * (1.0 - m3)
    # Holomorphic
    if Potential_Holomorphic:
        t = 1.0 - m3
        energy += mpi2 * t*t*t*t
    # Easy plane
    if Potential_EasyPlane:
        energy += 0.5 * mpi2 * m1*m1
    # Dihedral2
    if Potential_Dihedral2:
        energy += 0.5 * mpi2 * (1.0 - m1*m1) * (1.0 - m3*m3)
    # Aloof
    if Potential_Aloof:
        t = 1.0 - m3
        energy += 0.5 * mpi2 * t * (1.0 + t*t*t)
    # Dihedral3
    if Potential_Dihedral3:
        energy += (16.0 * mpi2 * (1.0 - m3)* (1.0 + 3.0*m3*m3 + 3.0*m1*m2*m2 - m1*m1*m1))
    # Broken
    if Potential_Broken:
        zr, zi = m1, m2
        for _ in range(N - 1):
            zr, zi = zr*m1 - zi*m2, zr*m2 + zi*m1
        wr = 1.0 - zr
        wi = -zi
        energy += mpi2 * (1.0 - m3) * (wr*wr + wi*wi)
    # Double vacua
    if Potential_DoubleVacua:
        energy += mpi2 * (1.0 - m3*m3)
    # Cell volume
    return energy * grid_volume

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

# ---------------- Compute the skyrmion number kernel ----------------
@cuda.jit
def compute_skyrmion_number_kernel(en, Field, d1fd1x, p_i, p_f):
    """
    Compute per-site skyrmion charge density contributions across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - Computes first derivatives only for magnetization components (0..2),
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

# ---------------- Compute the local energy gradient (Baby Skyrme) ----------------
@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Compute the local energy gradient ∂E/∂Field and set the local Velocity update direction.

    Usage:
    - Device helper used by higher-level kernels (e.g. do_gradient_step_kernel in the integrator module).
    - Assumes d1fd1x and d2fd2x have already been filled at (x, y) for all required components.
    - Computes gradient components g0..g2 corresponding to:
        0..2: magnetization (m1,m2,m3)
    - Writes EnergyGradient[a, x, y] = g_a
    - Projects magnetization gradient to be orthogonal to the magnetization constraint.
    - Sets Velocity[a, x, y] = -time_step * EnergyGradient[a, x, y] for all a.

    Parameters:
    - Velocity: device array updated in-place (descent direction scaled by time_step)
    - Field: device state array
    - EnergyGradient: device array receiving the gradient
    - d1fd1x: first-derivative buffer
    - d2fd2x: second-derivative buffer
    - p_f: uses p_f[5]=time_step and other model parameters (mpi, kappa)
    """
    # Parameters
    time_step = p_f[5]
    mpi = p_f[6]; mpi2 = mpi * mpi
    kappa = p_f[7]; kappa2 = kappa * kappa
    # Potential flags (0/1)
    Potential_Standard = p_i[11]
    Potential_Holomorphic = p_i[12]
    Potential_EasyPlane = p_i[13]
    Potential_Dihedral2 = p_i[14]
    Potential_Aloof = p_i[15]
    Potential_Dihedral3 = p_i[16]
    Potential_Broken = p_i[17]; N = p_i[19]
    Potential_DoubleVacua = p_i[18]
    # Fields
    m1 = Field[idx_field(0, x, y, p_i)]; m2 = Field[idx_field(1, x, y, p_i)]; m3 = Field[idx_field(2, x, y, p_i)]
    # First derivatives
    dm1dx = d1fd1x[idx_d1(0, 0, x, y, p_i)]; dm1dy = d1fd1x[idx_d1(1, 0, x, y, p_i)]
    dm2dx = d1fd1x[idx_d1(0, 1, x, y, p_i)]; dm2dy = d1fd1x[idx_d1(1, 1, x, y, p_i)]
    dm3dx = d1fd1x[idx_d1(0, 2, x, y, p_i)]; dm3dy = d1fd1x[idx_d1(1, 2, x, y, p_i)]
    # Alias notation: F[p][a]
    F00 = dm1dx; F01 = dm2dx; F02 = dm3dx
    F10 = dm1dy; F11 = dm2dy; F12 = dm3dy
    # Second derivatives
    d2m1d2x = d2fd2x[idx_d2(0, 0, 0, x, y, p_i)]; d2m1d2y = d2fd2x[idx_d2(1, 1, 0, x, y, p_i)]; d2m1dxdy = d2fd2x[idx_d2(0, 1, 0, x, y, p_i)]; d2m1dydx = d2fd2x[idx_d2(1, 0, 0, x, y, p_i)]
    d2m2d2x = d2fd2x[idx_d2(0, 0, 1, x, y, p_i)]; d2m2d2y = d2fd2x[idx_d2(1, 1, 1, x, y, p_i)]; d2m2dxdy = d2fd2x[idx_d2(0, 1, 1, x, y, p_i)]; d2m3dxdy = d2fd2x[idx_d2(0, 1, 2, x, y, p_i)]
    d2m3d2x = d2fd2x[idx_d2(0, 0, 2, x, y, p_i)]; d2m3d2y = d2fd2x[idx_d2(1, 1, 2, x, y, p_i)]; d2m2dydx = d2fd2x[idx_d2(1, 0, 1, x, y, p_i)]; d2m3dydx = d2fd2x[idx_d2(1, 0, 2, x, y, p_i)]
    # Alias notation: S[p][q][a]
    S000 = d2m1d2x; S001 = d2m2d2x; S002 = d2m3d2x
    S110 = d2m1d2y; S111 = d2m2d2y; S112 = d2m3d2y
    S010 = d2m1dxdy; S011 = d2m2dxdy; S012 = d2m3dxdy
    S100 = d2m1dydx; S101 = d2m2dydx; S102 = d2m3dydx
    # Precompute squares Σ_b (F[q][b]^2)
    sum_F0_sq = F00*F00 + F01*F01 + F02*F02
    sum_F1_sq = F10*F10 + F11*F11 + F12*F12
    # Energy gradients
    g0 = 0.0; g1 = 0.0; g2 = 0.0

    # ---- The C++ does:
    # for a:
    #   for p:
    #     g[a] -= S[p][p][a]
    #     for b,q:
    #       g[a] -= ( S[p][p][a] * (kappa2 * (F[q][b]^2)) + kappa2*( F[p][a]*(S[p][q][b]*F[q][b] - S[q][q][b]*F[p][b]) - S[p][q][a]*F[p][b]*F[q][b] ) )
    #
    # We'll compute it explicitly for (p,q) in {(0,0),(0,1),(1,0),(1,1)} and b in {0,1,2}.

    # Helpers for compactness:
    # Laplacians: S[p][p][a]
    L0_0 = S000 + S110  # (d2/dx2 + d2/dy2) of m1
    L0_1 = S001 + S111  # m2
    L0_2 = S002 + S112  # m3

    # First piece: - Σ_p S[p][p][a]  (i.e. -Laplacian)
    g0 -= L0_0
    g1 -= L0_1
    g2 -= L0_2

    # Now the large Skyrme corrections:
    # We'll do p=0 and p=1 separately (matches C++ structure), but still no big nested loops.

    # ---- p = 0 (x) ----
    # Spp[a] = S[0][0][a]
    Spp0 = S000; Spp1 = S001; Spp2 = S002

    # term: Spp[a] * (kappa2 * Σ_{q,b} F[q][b]^2)  but in C++ it's inside q,b loops,
    # so effectively it contributes: Σ_{q,b} Spp[a]*(kappa2*F[q][b]^2) = kappa2*Spp[a]*(sum_F0_sq + sum_F1_sq)
    sum_all_sq = sum_F0_sq + sum_F1_sq
    g0 -= kappa2 * Spp0 * sum_all_sq
    g1 -= kappa2 * Spp1 * sum_all_sq
    g2 -= kappa2 * Spp2 * sum_all_sq

    # Remaining: kappa2*( F[p][a]*(S[p][q][b]*F[q][b] - S[q][q][b]*F[p][b]) - S[p][q][a]*F[p][b]*F[q][b] )
    # summed over q in {0,1}, b in {0,1,2}. Here p=0.

    # For q=0:
    # S[p][q][b] = S[0][0][b] = (S000,S001,S002)
    # S[q][q][b] = S[0][0][b] same
    # So (S[p][q][b]*F[q][b] - S[q][q][b]*F[p][b]) = S00b*F0b - S00b*F0b = 0
    # leaving only: -kappa2*( - S[p][q][a]*F[p][b]*F[q][b]) = +kappa2*S00a*Σ_b(F0b*F0b)
    # but note C++ overall subtract: g[a] -= kappa2*( ... - S[p][q][a]*F[p][b]*F[q][b]) => g[a] -= kappa2*( ... ) and ... = -S00a*F0b*F0b,
    # so g[a] -= kappa2*(-S00a*sum_F0_sq) => g[a] += kappa2*S00a*sum_F0_sq.
    g0 += kappa2 * S000 * sum_F0_sq
    g1 += kappa2 * S001 * sum_F0_sq
    g2 += kappa2 * S002 * sum_F0_sq

    # For q=1:
    # S[p][q][b] = S[0][1][b] = (S010,S011,S012)
    # S[q][q][b] = S[1][1][b] = (S110,S111,S112)
    # Also S[p][q][a] = S[0][1][a] depends on a.

    # Compute A_b = (S[0][1][b]*F1b - S[1][1][b]*F0b)
    A0 = S010 * F10 - S110 * F00
    A1 = S011 * F11 - S111 * F01
    A2 = S012 * F12 - S112 * F02

    # Σ_b A_b over b weighted by nothing? No, it appears inside F[p][a]*A_b but A_b depends on b.
    # C++: Σ_b F[p][a] * A_b  and separately - Σ_b S[p][q][a]*F[p][b]*F[q][b]
    sumA = A0 + A1 + A2  # because F[p][a] is constant over b

    # Σ_b F[p][b]*F[q][b] with p=0,q=1 is dot(F0,F1)
    dot01 = F00*F10 + F01*F11 + F02*F12

    # Now apply for each a:
    # g[a] -= kappa2*( F0a*sumA - S01a*dot01 )
    g0 -= kappa2 * (F00 * sumA - S010 * dot01)
    g1 -= kappa2 * (F01 * sumA - S011 * dot01)
    g2 -= kappa2 * (F02 * sumA - S012 * dot01)

    # ---- p = 1 (y) ----
    Spp0 = S110; Spp1 = S111; Spp2 = S112
    g0 -= kappa2 * Spp0 * sum_all_sq
    g1 -= kappa2 * Spp1 * sum_all_sq
    g2 -= kappa2 * Spp2 * sum_all_sq

    # q=1 case analogous to q=0 above gives +kappa2*S11a*sum_F1_sq
    g0 += kappa2 * S110 * sum_F1_sq
    g1 += kappa2 * S111 * sum_F1_sq
    g2 += kappa2 * S112 * sum_F1_sq

    # q=0 cross terms with p=1,q=0:
    # S[p][q][b] = S[1][0][b] = (S100,S101,S102)
    # S[q][q][b] = S[0][0][b] = (S000,S001,S002)

    B0 = S100 * F00 - S000 * F10
    B1 = S101 * F01 - S001 * F11
    B2 = S102 * F02 - S002 * F12
    sumB = B0 + B1 + B2

    # dot(F1,F0) is same dot01
    # g[a] -= kappa2*( F1a*sumB - S10a*dot01 )
    g0 -= kappa2 * (F10 * sumB - S100 * dot01)
    g1 -= kappa2 * (F11 * sumB - S101 * dot01)
    g2 -= kappa2 * (F12 * sumB - S102 * dot01)

    # Standard
    if Potential_Standard:
        g2 -= mpi2
    # Holomorphic
    if Potential_Holomorphic:
        t = 1.0 - m3
        g2 -= 4.0 * mpi2 * t * t * t
    # Easy plane
    if Potential_EasyPlane:
        g0 += mpi2 * m1
    # Dihedral2
    if Potential_Dihedral2:
        g0 -= mpi2 * (1.0 - m3*m3) * m1
        g2 -= mpi2 * (1.0 - m1*m1) * m3
    # Aloof
    if Potential_Aloof:
        t = 1.0 - m3
        t3 = t * t * t
        g2 -= mpi2 * (0.5 * (1.0 + t3) + 1.5 * t3)
    # Dihedral3
    if Potential_Dihedral3:
        g0 += 48.0 * mpi2 * ((m2*m2) - (m1*m1)) * (1.0 - m3)
        g1 += 96.0 * mpi2 * m1 * m2 * (1.0 - m3)
        g2 -= 16.0 * mpi2 * (-(m1*m1*m1) + 3.0*m1*(m2*m2) + (1.0 - 3.0*m3)*(1.0 - 3.0*m3))
    # Broken
    if Potential_Broken:
        # NOTE: This block is a direct structural translation of the C++.
        # It is expensive and numerically delicate. Only keep if you actually use it.
        r2 = m1*m1 + m2*m2
        theta = atan2(m2, m1)

        # compute r^(N/2) and r^(N) via pow; avoid pow if N small and constant
        rN2 = r2 ** (0.5 * N)
        rN = r2 ** (1.0 * N)

        c = cos(N * theta)
        s = sin(N * theta)

        denom = sqrt(1.0 - 2.0 * c * rN2 + rN)
        # norm(1 - (m1+i m2)^N) in polar form equals (1 - 2 c r^(N/2) + r^N)
        norm1mzN = (1.0 - 2.0 * c * rN2 + rN)

        # These expressions match your C++ (just rewritten minimally)
        if denom != 0.0 and r2 != 0.0:
            pref = 2.0 * mpi2 * N * sqrt(norm1mzN) * (r2 ** (-1.0 + 0.5 * N)) * (-1.0 + m3) / denom
            g0 += pref * (s * m2 + m1 * (c - rN2))
            g1 += (-pref) * (s * m1 + m2 * (-c + rN2))
        g2 -= mpi2 * norm1mzN
    # Double vacua
    if Potential_DoubleVacua:
        g2 -= 2.0 * mpi2 * m3
    # Write gradient
    EnergyGradient[idx_field(0, x, y, p_i)] = g0
    EnergyGradient[idx_field(1, x, y, p_i)] = g1
    EnergyGradient[idx_field(2, x, y, p_i)] = g2
    # Project magnetization
    project_orthogonal_magnetization(EnergyGradient, Field, x, y, p_i, p_f)
    # Calculates the velocity at each lattice point
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)