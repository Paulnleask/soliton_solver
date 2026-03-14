"""
Host wrappers for observables in the spin triplet superconducting ferromagnet theory.

Examples
--------
Use ``compute_energy`` to evaluate the total energy.
Use ``compute_skyrmion_number`` to evaluate the total skyrmion number.
Use ``compute_vortex_number`` to evaluate the total vortex number.
"""
from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.spin_triplet_superconducting_magnet.kernels import compute_energy_kernel, compute_skyrmion_number_kernel, compute_vortex_number_kernel

def compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h):
    """
    Compute the total energy.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device first derivative buffer.
    en : device array
        Device buffer for local energy contributions.
    entmp : device array
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial sums.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : host array
        Host integer parameter array.
    p_f_h : host array
        Host float parameter array.

    Returns
    -------
    float
        Total energy.

    Examples
    --------
    Use ``energy = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)`` to evaluate the total energy.
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

def compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h):
    """
    Compute the total skyrmion number.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device first derivative buffer.
    en : device array
        Device buffer for local skyrmion density contributions.
    entmp : device array
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial sums.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : host array
        Host integer parameter array.

    Returns
    -------
    float
        Total skyrmion number.

    Examples
    --------
    Use ``charge = compute_skyrmion_number(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)`` to evaluate the total skyrmion number.
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

def compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h):
    """
    Compute the total vortex number for a selected flux component.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device first derivative buffer.
    en : device array
        Device buffer for local flux density contributions.
    entmp : device array
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial sums.
    which : int
        Flux component selector.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : host array
        Host integer parameter array.

    Returns
    -------
    float
        Total vortex number for the selected component.

    Examples
    --------
    Use ``charge = compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h)`` to evaluate the total vortex number.
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