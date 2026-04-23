"""
Host side observable wrappers for energy and topological charge calculations.

Examples
--------
Use ``compute_energy`` to evaluate the total energy from the current field configuration.
Use ``compute_skyrmion_number`` to evaluate the total skyrmion number from the current field configuration.
"""

from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.baby_skyrme.kernels import compute_energy_kernel, compute_skyrmion_number_kernel

def compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h):
    """
    Compute the total energy by summing the per site energy contributions.

    Parameters
    ----------
    Field : device array
        Device field array containing the simulation fields.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer receiving the per site energy contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array
        Integer host parameter array.
    p_f_h : host array
        Float host parameter array.

    Returns
    -------
    float
        Total energy over the grid.

    Examples
    --------
    Use ``E = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)`` to compute the total energy.
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

def compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h):
    """
    Compute the total skyrmion number by summing the per site charge density contributions.

    Parameters
    ----------
    Field : device array
        Device field array containing the magnetization components.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer receiving the per site charge density contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Integer device parameter array.
    p_f_d : device array
        Float device parameter array.
    p_i_h : host array
        Integer host parameter array.

    Returns
    -------
    float
        Total skyrmion number over the grid.

    Examples
    --------
    Use ``Q = compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)`` to compute the total skyrmion number.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    compute_skyrmion_number_kernel[grid2d, block2d](en, Field, d1fd1x, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge