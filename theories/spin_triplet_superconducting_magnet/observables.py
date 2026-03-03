# =========================
# soliton_solver/theories/spin_triplet_superconducting_magnet/observables.py
# =========================
"""
Host-side observable wrappers (energy / topological charges).

Usage:
- These functions are *Python host* helpers that:
  1) launch a CUDA kernel that writes a per-site scalar field into `en`,
  2) reduce/sum that scalar field into a single number using compute_sum(...),
  3) clear temporary buffers for the next call.

Buffers:
- Field: device array containing the simulation fields.
- d1fd1x: device buffer for first derivatives (written by kernels as needed).
- en: device scalar field buffer (component 0 holds per-site contributions).
- entmp: device buffer used as input to compute_sum() (often a 1-component copy of `en`).
- gridsum_partial: device buffer for partial reductions used by compute_sum().

Parameters:
- p_i_d, p_f_d: device parameter arrays passed into kernels.
- p_i_h, p_f_h: host parameter arrays used for launch sizing and reduction dimensions.
"""

# ---------------- Imports ----------------
from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.spin_triplet_superconducting_magnet.kernels import compute_energy_kernel, compute_skyrmion_number_kernel, compute_vortex_number_kernel

# ---------------- Compute the energy wrapper ----------------
def compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h):
    """
    Compute the total energy by summing per-site energy contributions.

    Usage:
    - Call on the host when you want the total energy scalar:
        E = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)
    - Internally:
        1) compute_energy_kernel writes en[a=0, x, y] = local energy contribution
        2) entmp is filled from en (device-to-device copy via copy_to_device)
        3) compute_sum reduces entmp to a single float
        4) entmp and en are zeroed for reuse

    Parameters:
    - Field: device field array
    - d1fd1x: device first-derivative buffer (used by compute_energy_kernel)
    - en: device output buffer for local energy contributions
    - entmp: device scratch buffer passed into compute_sum (same shape as en's component 0)
    - gridsum_partial: device buffer for partial reduction results
    - p_i_d, p_f_d: device parameter arrays
    - p_i_h, p_f_h: host parameter arrays (p_i_h used for launch + reduction sizing)

    Returns:
    - energy (float): total energy over the grid.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))
    compute_energy_kernel[grid2d, block2d](en, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    energy = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return energy

# ---------------- Compute the skyrmion number wrapper ----------------
def compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h):
    """
    Compute the total skyrmion number (topological charge) by summing per-site density.

    Usage:
    - Call on the host:
        Q = compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)
    - Internally launches compute_skyrmion_number_kernel which writes a per-site charge density
      contribution into en[a=0, x, y], then reduces via compute_sum().

    Parameters:
    - Field: device field array (magnetization components are used)
    - d1fd1x: device first-derivative buffer (filled by the kernel)
    - en: device output buffer for per-site charge density contributions
    - entmp: device scratch buffer passed into compute_sum
    - gridsum_partial: device buffer for partial reduction results
    - p_i_d, p_f_d: device parameter arrays
    - p_i_h: host int parameter array (used for launch + reduction sizing)

    Returns:
    - charge (float): total skyrmion number.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))
    compute_skyrmion_number_kernel[grid2d, block2d](en, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge

# ---------------- Compute the vortex number wrapper ----------------
def compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h):
    """
    Compute the total vortex/flux number by summing a selected per-site flux-density component.

    Usage:
    - Call on the host:
        N = compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h)
    - `which` selects which flux component the kernel writes (see compute_vortex_number_kernel /
      compute_vortex_density in kernels.py).
    - Internally launches compute_vortex_number_kernel, then reduces via compute_sum().

    Parameters:
    - Field: device field array (gauge components are used)
    - d1fd1x: device first-derivative buffer (filled by the kernel)
    - en: device output buffer for per-site flux contributions
    - entmp: device scratch buffer passed into compute_sum
    - gridsum_partial: device buffer for partial reduction results
    - which: integer selector for the flux component (0/1/other)
    - p_i_d, p_f_d: device parameter arrays
    - p_i_h: host int parameter array (used for launch + reduction sizing)

    Returns:
    - charge (float): total vortex/flux number for the selected component.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))
    compute_vortex_number_kernel[grid2d, block2d](en, Field, d1fd1x, which, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge