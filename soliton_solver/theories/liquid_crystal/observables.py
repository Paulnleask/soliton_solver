"""
Host side wrappers for liquid crystal observables.

Examples
--------
Use ``compute_energy`` to evaluate the total energy.
Use ``compute_skyrmion_number`` to evaluate the total skyrmion number.
Use ``compute_electric_charge`` to evaluate the total electric charge.
"""
from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.liquid_crystal.kernels import compute_energy_kernel, compute_skyrmion_number_kernel, compute_electric_charge_kernel

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
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : array-like
        Host integer parameter array.
    p_f_h : array-like
        Host float parameter array.

    Returns
    -------
    float
        Total energy.

    Examples
    --------
    Use ``E = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)`` to evaluate the total energy.
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
    Compute the total skyrmion number.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer for per-site skyrmion density contributions.
    entmp : device array
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : array-like
        Host integer parameter array.

    Returns
    -------
    float
        Total skyrmion number.

    Examples
    --------
    Use ``Q = compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)`` to evaluate the total skyrmion number.
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

def compute_electric_charge(Field, d1fd1x, d2fd2x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h):
    """
    Compute the total electric charge.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device buffer for first derivatives.
    d2fd2x : device array
        Device buffer for second derivatives.
    en : device array
        Device buffer for per-site electric charge contributions.
    entmp : device array
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : array-like
        Host integer parameter array.

    Returns
    -------
    float
        Total electric charge.

    Examples
    --------
    Use ``Qe = compute_electric_charge(Field, d1fd1x, d2fd2x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)`` to evaluate the total electric charge.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    compute_electric_charge_kernel[grid2d, block2d](Field, d1fd1x, d2fd2x, en, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge