"""
Host wrappers for Bose-Einstein condensate observables.

Examples
--------
Use ``compute_energy`` to compute the total energy.
Use ``compute_norm`` to compute the total norm.
"""
from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.bose_einstein_condensate.kernels import compute_energy_kernel, compute_norm_kernel

def compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h):
    """
    Compute the total energy.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer for per-site energy contributions.
    entmp : device array
        Device buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial sums.
    p_i_d : device array
        Integer parameter array on the device.
    p_f_d : device array
        Float parameter array on the device.
    p_i_h : array-like
        Integer parameter array on the host.
    p_f_h : array-like
        Float parameter array on the host.

    Returns
    -------
    float
        Total energy over the grid.

    Examples
    --------
    Use ``energy = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)`` to compute the total energy.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    compute_energy_kernel[grid2d, block2d](en, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    energy = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return energy

def compute_norm(Field, en, entmp, gridsum_partial, p_i_d, p_i_h, p_i_f):
    """
    Compute the total norm.

    Parameters
    ----------
    Field : device array
        Device field array.
    en : device array
        Device buffer for per-site norm contributions.
    entmp : device array
        Device buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial sums.
    p_i_d : device array
        Integer parameter array on the device.
    p_i_h : array-like
        Integer parameter array on the host.
    p_i_f : device array
        Float parameter array on the device.

    Returns
    -------
    float
        Total norm over the grid.

    Examples
    --------
    Use ``norm = compute_norm(Field, en, entmp, gridsum_partial, p_i_d, p_i_h, p_i_f)`` to compute the total norm.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    compute_norm_kernel[grid2d, block2d](en, Field, p_i_d, p_i_f)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    norm = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return norm