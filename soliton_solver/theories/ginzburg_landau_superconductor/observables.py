"""
Compute host-side observables for the Ginzburg-Landau superconductor theory.

Examples
--------
Use ``compute_energy`` to compute the total energy.
Use ``compute_vortex_number`` to compute the total magnetic flux.
"""
from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.ginzburg_landau_superconductor.kernels import compute_energy_kernel, compute_vortex_number_kernel

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
        Device buffer for local energy contributions.
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

def compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h):
    """
    Compute the total magnetic flux.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer for local flux contributions.
    entmp : device array
        Device scratch buffer used for reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    which : int
        Flux component selector.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : array-like
        Host integer parameter array.

    Returns
    -------
    float
        Total magnetic flux for the selected component.

    Examples
    --------
    Use ``charge = compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h)`` to compute the total magnetic flux.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    compute_vortex_number_kernel[grid2d, block2d](en, Field, d1fd1x, which, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge