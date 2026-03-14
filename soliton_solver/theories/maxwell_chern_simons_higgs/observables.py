"""
Host wrappers for Maxwell-Chern-Simons-Higgs observables.

Examples
--------
Use ``compute_energy`` to evaluate the total energy.
Use ``compute_vortex_number`` to evaluate the total magnetic flux.
"""
from numba import cuda
from soliton_solver.core.utils import compute_sum
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.maxwell_chern_simons_higgs.kernels import compute_energy_kernel, compute_vortex_number_kernel, compute_electric_charge_kernel, compute_noether_charge_kernel

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
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
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
        Total energy over the grid.

    Examples
    --------
    Use ``E = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)`` to evaluate the total energy.
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

def compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h):
    """
    Compute the total vortex number.

    Parameters
    ----------
    Field : device array
        Device field array.
    d1fd1x : device array
        Device buffer for first derivatives.
    en : device array
        Device buffer for per-site flux contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
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
    Use ``N = compute_vortex_number(Field, d1fd1x, en, entmp, gridsum_partial, which, p_i_d, p_f_d, p_i_h)`` to evaluate the total vortex number.
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
        Device buffer for per-site charge contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : host array
        Host integer parameter array.

    Returns
    -------
    float
        Total electric charge over the grid.

    Examples
    --------
    Use ``Q = compute_electric_charge(Field, d1fd1x, d2fd2x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)`` to evaluate the total electric charge.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))
    compute_electric_charge_kernel[grid2d, block2d](Field, d1fd1x, d2fd2x, en, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge

def compute_noether_charge(Field, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h):
    """
    Compute the total Noether charge.

    Parameters
    ----------
    Field : device array
        Device field array.
    en : device array
        Device buffer for per-site charge contributions.
    entmp : device array
        Device scratch buffer used in the reduction.
    gridsum_partial : device array
        Device buffer for partial reduction results.
    p_i_d : device array
        Device integer parameter array.
    p_f_d : device array
        Device float parameter array.
    p_i_h : host array
        Host integer parameter array.

    Returns
    -------
    float
        Total Noether charge over the grid.

    Examples
    --------
    Use ``Q = compute_noether_charge(Field, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h)`` to evaluate the total Noether charge.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(16, 32))
    compute_noether_charge_kernel[grid2d, block2d](Field, en, p_i_d, p_f_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    charge = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return charge