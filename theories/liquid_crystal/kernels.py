# =========================
# soliton_solver/theories/liquid_crystal/kernels.py
# =========================
"""
Core CUDA kernels and device helpers for the liquid_crystal model.

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
    Depolarization: a = 3      (psi)
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
    Chiral magnet energy density (Dirichlet + DMI + Zeeman + anisotropy).

    Faithful translation of the CUDA C++ kernel with:
    - Dirichlet exchange
    - DMI (Dresselhaus / Rashba / Heusler / Hybrid)
    - Magnetic potential
    - Zeeman term
    - Uniaxial anisotropy

    Returns cell-integrated energy (× grid_volume).
    """

    # ---------------- Parameters ----------------
    grid_volume = p_f[4]
    coup_PotE = p_f[6]; coup_Potw0 = p_f[7]; coup_eps = p_f[8]; e1 = p_f[9]; e3 = p_f[10]
    number_coordinates = p_i[3]; number_total_fields = p_i[4]; number_magnetization_fields = p_i[10]

    # DMI flags
    dmi_dresselhaus = p_i[11]; dmi_rashba = p_i[12]

    # depol flag
    depol = p_i[13]

    # ---------------- Levi-Civita ε_{ijk} ----------------
    # Only nonzero entries hard-coded (faster than 3×3×3 array)
    # ε_{012}=+1 etc., but we only use magnetization indices 0,1,2
    def levi(i, j, k):
        if i == 0 and j == 1 and k == 2: return  1.0
        if i == 1 and j == 2 and k == 0: return  1.0
        if i == 2 and j == 0 and k == 1: return  1.0
        if i == 0 and j == 2 and k == 1: return -1.0
        if i == 2 and j == 1 and k == 0: return -1.0
        if i == 1 and j == 0 and k == 2: return -1.0
        return 0.0

    # ---------------- DMI tensor D_{ji} ----------------
    # Default: zero
    D00 = 0.0; D01 = 0.0
    D10 = 0.0; D11 = 0.0

    if dmi_dresselhaus:
        D00 = -1.0; D11 = -1.0
    elif dmi_rashba:
        D01 = -1.0; D10 =  1.0

    # ---------------- Dirichlet exchange ----------------
    energy = 0.0
    for a in range(number_magnetization_fields):
        for i in range(number_coordinates):
            dfa = d1fd1x[idx_d1(i, a, x, y, p_i)]
            energy += 0.5 * dfa * dfa

    # ---------------- Magnetic potential + DMI ----------------
    for i in range(number_coordinates):
        for j in range(number_coordinates):
            # pick DMI coefficient
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

    # electric potential term: 1/2 * P*Grad(psi)        
    if depol:
        for i in range(number_coordinates):
            for j in range(number_magnetization_fields):
                energy += (e3 / e1) * Field[idx_field(j, x, y, p_i)] * d1fd1x[idx_d1(i, j, x, y, p_i)] * d1fd1x[idx_d1(i, 3, x, y, p_i)] / 2.0
            for j in range(number_coordinates):
                energy += d1fd1x[idx_d1(i, i, x, y, p_i)] * Field[idx_field(j, x, y, p_i)] * d1fd1x[idx_d1(j, 3, x, y, p_i)] / 2.0
                energy -= (e3 / e1) * Field[idx_field(j, x, y, p_i)] * d1fd1x[idx_d1(j, i, x, y, p_i)] * d1fd1x[idx_d1(i, 3, x, y, p_i)] / 2.0

    # ---------------- Electric term ----------------
    mz = Field[idx_field(number_magnetization_fields - 1, x, y, p_i)]
    energy += 0.5 * coup_PotE * (1.0 - mz * mz)

    # ---------------- Homeotropic anchoring ----------------
    energy += 0.5 * coup_Potw0 * (1.0 - mz * mz)

    # ---------------- Cell volume ----------------
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

# ---------------- Compute the local supercurrent ----------------
@cuda.jit(device=True)
def compute_electric_charge_point(Field, d1fd1x, d2fd2x, x, y, p_i, p_f):
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
    charge = 0.0
    e1 = p_f[9]; e3 = p_f[10]
    grid_volume = p_f[4]
    number_coordinates = p_i[3]
    for i in range(number_coordinates):
        for j in range(number_coordinates):
            charge -= d1fd1x[idx_d1(i, i, x, y, p_i)] * d1fd1x[idx_d1(j, j, x, y, p_i)] + Field[idx_field(i, x, y, p_i)] * d2fd2x[idx_d2(i, j, j, x, y, p_i)] - (e3 / e1) * (d1fd1x[idx_d1(i, j, x, y, p_i)] * d1fd1x[idx_d1(j, i, x, y, p_i)] + Field[idx_field(j, x, y, p_i)] * d2fd2x[idx_d2(i, j, i, x, y, p_i)])
    return charge * grid_volume

# ---------------- Compute the supercurrent kernel ----------------
@cuda.jit
def compute_electric_charge_kernel(Field, d1fd1x, d2fd2x, en, p_i, p_f):
    """
    Compute supercurrent across the grid.

    Usage:
    - Launch over (xlen, ylen).
    - Computes second derivatives for gauge fields (components in the gauge block),
      then calls compute_supercurrent_point(...) to write Supercurrent at (x, y).

    Parameters:
    - Field: device state array
    - d1fd1x: device second-derivative buffer (written here for gauge components)
    - en: device output array (scalar field)
    - p_i: expects p_i[10]=number_magnetization_fields
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_magnetization_fields = p_i[10]
    for a in range(number_magnetization_fields):
        compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
        compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
    en[idx_field(0, x, y, p_i)] = compute_electric_charge_point(Field, d1fd1x, d2fd2x, x, y, p_i, p_f)

@cuda.jit(device=True)
def do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f):
    """
    Faithful translation of chiral-magnet CUDA C++ do_gradient_step.

    Computes EnergyGradient[a] = δE/δField[a] at (x,y) for:
    - Dirichlet (exchange):  -∂_j∂_j Field[a]
    - DMI: + 2 * D[b,a] * ε_{b,i,j} * ∂_a Field[j]   (as in C++)
    - Zeeman + anisotropy on last component (mz = Field[number_total_fields-1])
    - Magnetic potential: add FirstDerivative_Potential[i] to EnergyGradient[i] for i in [0..number_coordinates-1]
    - Boundary condition: zero gradient in halo/outside interior
    - Writes EnergyGradient and sets Velocity = -time_step * EnergyGradient

    Conventions:
    - Field idx: idx_field(a,x,y,p_i)
    - d1fd1x: idx_d1(mu,a,x,y,p_i)
    - d2fd2x: idx_d2(mu,nu,a,x,y,p_i)
    """

    # ---------------- Parameters ----------------
    time_step = p_f[5]
    coup_PotE = p_f[6]; coup_Potw0 = p_f[7]; coup_eps = p_f[8]; e1 = p_f[9]; e3 = p_f[10]
    number_coordinates = p_i[3]; number_total_fields = p_i[4]; number_magnetization_fields = p_i[10]
    xlen = p_i[0]; ylen = p_i[1]; halo = p_i[2]

    # DMI flags
    dmi_dresselhaus = p_i[11]; dmi_rashba = p_i[12]
    depol = p_i[13]

    # ---------------- Levi-Civita ε_{ijk} for i,j,k in {0,1,2} ----------------
    # Hard-coded (faster than array)
    def levi(i, j, k):
        if i == 0 and j == 1 and k == 2: return  1.0
        if i == 1 and j == 2 and k == 0: return  1.0
        if i == 2 and j == 0 and k == 1: return  1.0
        if i == 0 and j == 2 and k == 1: return -1.0
        if i == 2 and j == 1 and k == 0: return -1.0
        if i == 1 and j == 0 and k == 2: return -1.0
        return 0.0

    # ---------------- DMI tensor D[b][a] with a,b in {0,1} ----------------
    D00 = 0.0; D01 = 0.0
    D10 = 0.0; D11 = 0.0
    if dmi_dresselhaus:
        D00 = -1.0; D11 = -1.0
    elif dmi_rashba:
        D01 = -1.0; D10 =  1.0

    # ---------------- Initialize gradient ----------------
    # We assume number_total_fields is small; common case is 4 (m1,m2,m3,psi/mz).
    # Do explicit variables for speed if you know it is always 4; otherwise keep loop.
    for a in range(number_total_fields):
        EnergyGradient[idx_field(a, x, y, p_i)] = 0.0

    # ---------------- Dirichlet: g[i] -= Σ_j ∂_j∂_j Field[i] ----------------
    for i in range(number_magnetization_fields):
        lap = 0.0
        for j in range(number_coordinates):
            lap += d2fd2x[idx_d2(j, j, i, x, y, p_i)]
        EnergyGradient[idx_field(i, x, y, p_i)] -= lap

    # ---------------- DMI: g[i] += 2 * D[b][a] * ε_{b,i,j} * ∂_a Field[j] ----------------
    # In practice DMI couples magnetization components only (0..2), but C++ loops i,j over number_total_fields.
    # We'll keep it faithful but restrict epsilon evaluation to indices <3 (otherwise epsilon=0).
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

    # ---------------- Zeeman + anisotropy on last component ----------------
    last = number_magnetization_fields - 1
    mz = Field[idx_field(last, x, y, p_i)]
    EnergyGradient[idx_field(last, x, y, p_i)] -= coup_PotE * mz
    EnergyGradient[idx_field(last, x, y, p_i)] -= coup_Potw0 * mz

    # ---------------- Depolarization ----------------
    if depol:
        for j in range(number_coordinates):
            EnergyGradient[idx_field(3, x, y, p_i)] -= d2fd2x[idx_d2(j, j, 3, x, y, p_i)]
            for i in range(number_coordinates):
                EnergyGradient[idx_field(3, x, y, p_i)] += (1.0 / coup_eps) * (d1fd1x[idx_d1(i, i, x, y, p_i)] * d1fd1x[idx_d1(j, j, x, y, p_i)] + Field[idx_field(i, x, y, p_i)] * d2fd2x[idx_d2(i, j, j, x, y, p_i)] - (e3 / e1) * (d1fd1x[idx_d1(i, j, x, y, p_i)] * d1fd1x[idx_d1(j, i, x, y, p_i)] + Field[idx_field(j, x, y, p_i)] * d2fd2x[idx_d2(i, j, i, x, y, p_i)]))
                EnergyGradient[idx_field(i, x, y, p_i)] += d1fd1x[idx_d1(i, 3, x, y, p_i)] * d1fd1x[idx_d1(j, j, x, y, p_i)] - d1fd1x[idx_d1(j, 3, x, y, p_i)] * d1fd1x[idx_d1(i, j, x, y, p_i)] - Field[idx_field(j, x, y, p_i)] * d2fd2x[idx_d2(i, j, 3, x, y, p_i)]
                EnergyGradient[idx_field(i, x, y, p_i)] += (e3 / e1) * (d1fd1x[idx_d1(i, 3, x, y, p_i)] * d1fd1x[idx_d1(j, j, x, y, p_i)] - d1fd1x[idx_d1(j, 3, x, y, p_i)] * d1fd1x[idx_d1(i, j, x, y, p_i)] + Field[idx_field(j, x, y, p_i)] * d2fd2x[idx_d2(i, j, 3, x, y, p_i)])
            for i in range(number_magnetization_fields):
                EnergyGradient[idx_field(i, x, y, p_i)] -= (e3 / e1) * Field[idx_field(i, x, y, p_i)] * d2fd2x[idx_d2(j, j, 3, x, y, p_i)]

    # ---------------- Boundary conditions: zero gradient in halo/outside ----------------
    if (x < halo) or (x > xlen - halo - 1) or (y < halo) or (y > ylen - halo - 1):
        for a in range(number_total_fields):
            EnergyGradient[idx_field(a, x, y, p_i)] = 0.0

    # ---------------- Project magnetization gradient (constraint) ----------------
    project_orthogonal_magnetization(EnergyGradient, Field, x, y, p_i, p_f)

    # ---------------- Velocity update ----------------
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] = -time_step * EnergyGradient[idx_field(a, x, y, p_i)]

do_gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)