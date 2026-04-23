"""
Time stepping and relaxation kernels for the soliton_solver simulation.

Examples
--------
Use ``make_do_gradient_step_kernel`` to build a theory specific CUDA gradient kernel.
Use ``make_do_rk4_kernel`` to build an RK4 update kernel with optional constraint projection.
Use ``do_arrested_newton_flow`` to advance the field and velocity by one arrested Newton flow step.
"""

from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds, launch_2d, set_field_zero_kernel, compute_max_field, compute_sum
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second

def make_do_gradient_step_kernel(do_gradient_step_point):
    """
    Create a CUDA kernel that computes spatial derivatives and applies a per site gradient update.

    Parameters
    ----------
    do_gradient_step_point : device function
        CUDA device function that applies the gradient update at a single lattice site.

    Returns
    -------
    function
        CUDA kernel that computes derivatives and updates the energy gradient.

    Raises
    ------
    None.

    Examples
    --------
    Use ``gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)`` to create the gradient kernel.
    """
    @cuda.jit
    def _do_gradient_step_kernel(Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, p_i, p_f):
        x, y = cuda.grid(2)
        if not in_bounds(x, y, p_i):
            return
        number_total_fields = p_i[4]
        for a in range(number_total_fields):
            compute_derivative_first(d1fd1x, Field, a, x, y, p_i, p_f)
            compute_derivative_second(d2fd2x, Field, a, x, y, p_i, p_f)
        do_gradient_step_point(Velocity, Field, EnergyGradient, d1fd1x, d2fd2x, x, y, p_i, p_f)

    return _do_gradient_step_kernel

@cuda.jit
def do_rk4_step_kernel(k_out, Velocity, l_in, Temp, Field, k_prev, factor, p_i, p_f):
    """
    Compute one RK4 position slope and the corresponding intermediate field.

    Parameters
    ----------
    k_out : device array
        Output buffer for the RK4 position slope.
    Velocity : device array
        Current velocity field.
    l_in : device array
        Input RK4 velocity slope.
    Temp : device array
        Output buffer for the intermediate field.
    Field : device array
        Current field configuration.
    k_prev : device array
        Previous RK4 position slope.
    factor : float
        RK4 stage factor.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array containing the time step.

    Returns
    -------
    None
        The slope and intermediate field are written in place.

    Raises
    ------
    None.

    Examples
    --------
    Launch ``do_rk4_step_kernel[grid2d, block2d](k_out, Velocity, l_in, Temp, Field, k_prev, factor, p_i, p_f)`` to compute one RK4 stage.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    dt = p_f[5]
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        k_out[idx_field(a, x, y, p_i)] = dt * (Velocity[idx_field(a, x, y, p_i)] + factor * l_in[idx_field(a, x, y, p_i)])
        Temp[idx_field(a, x, y, p_i)] = Field[idx_field(a, x, y, p_i)] + factor * k_prev[idx_field(a, x, y, p_i)]

@cuda.jit
def do_rk4_kernel_no_constraint(Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f):
    """
    Finalize an RK4 update without constraint projection.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Field configuration updated in place.
    k1 : device array
        First RK4 position slope.
    k2 : device array
        Second RK4 position slope.
    k3 : device array
        Third RK4 position slope.
    k4 : device array
        Fourth RK4 position slope.
    l1 : device array
        First RK4 velocity slope.
    l2 : device array
        Second RK4 velocity slope.
    l3 : device array
        Third RK4 velocity slope.
    l4 : device array
        Fourth RK4 velocity slope.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The field and velocity are updated in place.

    Raises
    ------
    None.

    Examples
    --------
    Launch ``do_rk4_kernel_no_constraint[grid2d, block2d](Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f)`` to apply the RK4 update.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] += (1.0 / 6.0) * (l1[idx_field(a, x, y, p_i)] + 2.0 * l2[idx_field(a, x, y, p_i)] + 2.0 * l3[idx_field(a, x, y, p_i)] + l4[idx_field(a, x, y, p_i)])
        Field[idx_field(a, x, y, p_i)] += (1.0 / 6.0) * (k1[idx_field(a, x, y, p_i)] + 2.0 * k2[idx_field(a, x, y, p_i)] + 2.0 * k3[idx_field(a, x, y, p_i)] + k4[idx_field(a, x, y, p_i)])

def make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization):
    """
    Create an RK4 finalize kernel with optional unit magnetization constraint enforcement.

    Parameters
    ----------
    compute_norm_magnetization : device function
        CUDA device function that computes the magnetization norm at a lattice site.
    project_orthogonal_magnetization : device function
        CUDA device function that projects the velocity to satisfy the constraint.

    Returns
    -------
    function
        CUDA kernel that finalizes the RK4 update and applies the optional constraint.

    Raises
    ------
    None.

    Examples
    --------
    Use ``rk4_kernel = make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization)`` to create the constrained RK4 kernel.
    """
    @cuda.jit
    def _do_rk4_kernel(Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f):
        x, y = cuda.grid(2)
        if not in_bounds(x, y, p_i):
            return
        number_total_fields = p_i[4]
        for a in range(number_total_fields):
            Velocity[idx_field(a, x, y, p_i)] += (1.0 / 6.0) * (l1[idx_field(a, x, y, p_i)] + 2.0 * l2[idx_field(a, x, y, p_i)] + 2.0 * l3[idx_field(a, x, y, p_i)] + l4[idx_field(a, x, y, p_i)])
            Field[idx_field(a, x, y, p_i)] += (1.0 / 6.0) * (k1[idx_field(a, x, y, p_i)] + 2.0 * k2[idx_field(a, x, y, p_i)] + 2.0 * k3[idx_field(a, x, y, p_i)] + k4[idx_field(a, x, y, p_i)])
        unit_magnetization = p_i[9]
        if unit_magnetization:
            compute_norm_magnetization(Field, x, y, p_i, p_f)
            project_orthogonal_magnetization(Velocity, Field, x, y, p_i, p_f)

    return _do_rk4_kernel

@cuda.jit(device=True)
def arresting_criteria_point(Velocity, EnergyGradient, p_i, x, y):
    force = 0.0
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        force += Velocity[idx_field(a, x, y, p_i)] * EnergyGradient[idx_field(a, x, y, p_i)]
    return force

@cuda.jit
def arresting_criteria_kernel(Velocity, EnergyGradient, en, p_i_d):
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i_d):
        return
    en[idx_field(0, x, y, p_i_d)] = arresting_criteria_point(Velocity, EnergyGradient, p_i_d, x, y)

def arresting_criteria(Velocity, EnergyGradient, en, entmp, gridsum_partial, p_i_d, p_i_h):
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))
    arresting_criteria_kernel[grid2d, block2d](Velocity, EnergyGradient, en, p_i_d)
    cuda.synchronize()
    entmp.copy_to_device(en)
    dim_grid = p_i_h[5]
    force = compute_sum(entmp, gridsum_partial, int(dim_grid))
    entmp[:] = 0.0
    en[:] = 0.0
    return force

def do_arrested_newton_flow(Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, k1, k2, k3, k4, l1, l2, l3, l4, Temp, en, entmp, gridsum_partial, max_partial, p_i_d, p_f_d, p_i_h, p_f_h, prev_energy, compute_energy, gradient_step_kernel, rk4_kernel, compute_norm=None, do_norm_kernel=None):
    """
    Perform one arrested Newton flow step using RK4 integration.

    Parameters
    ----------
    Velocity : device array
        Velocity field updated in place.
    Field : device array
        Field configuration updated in place.
    d1fd1x : device array
        Buffer for first derivatives.
    d2fd2x : device array
        Buffer for second derivatives.
    EnergyGradient : device array
        Buffer for the energy gradient.
    k1 : device array
        First RK4 position slope.
    k2 : device array
        Second RK4 position slope.
    k3 : device array
        Third RK4 position slope.
    k4 : device array
        Fourth RK4 position slope.
    l1 : device array
        First RK4 velocity slope.
    l2 : device array
        Second RK4 velocity slope.
    l3 : device array
        Third RK4 velocity slope.
    l4 : device array
        Fourth RK4 velocity slope.
    Temp : device array
        Intermediate field buffer.
    en : device array
        Energy buffer.
    entmp : device array
        Temporary energy buffer.
    gridsum_partial : device array
        Partial reduction buffer for energy sums.
    max_partial : device array
        Partial reduction buffer for maximum norms.
    p_i_d : device array
        Integer parameter array on the device.
    p_f_d : device array
        Float parameter array on the device.
    p_i_h : host array
        Integer parameter array on the host.
    p_f_h : host array
        Float parameter array on the host.
    prev_energy : float
        Energy from the previous step.
    compute_energy : function
        Host function that computes the scalar energy.
    gradient_step_kernel : function
        CUDA kernel that computes the gradient step.
    rk4_kernel : function
        CUDA kernel that finalizes the RK4 update.
    compute_norm : function, optional
        Host function that computes a normalization factor.
    do_norm_kernel : function, optional
        CUDA kernel that applies the normalization.

    Returns
    -------
    tuple
        Pair ``(new_energy, err)`` containing the updated energy and the maximum norm of the energy gradient.

    Raises
    ------
    None.

    Examples
    --------
    Use ``new_energy, err = do_arrested_newton_flow(...)`` to advance the simulation by one minimization step.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(8, 8))

    def maybe_normalize(target_field):
        """
        Apply the optional field normalization.

        Parameters
        ----------
        target_field : device array
            Field buffer to normalize.

        Returns
        -------
        None
            The field is normalized in place when normalization is enabled.

        Raises
        ------
        None.

        Examples
        --------
        Use ``maybe_normalize(Temp)`` to normalize an intermediate field buffer when a normalization routine is provided.
        """
        if (compute_norm is None) or (do_norm_kernel is None):
            return
        norm = compute_norm(target_field, en, entmp, gridsum_partial, p_i_d, p_i_h, p_f_d)
        do_norm_kernel[grid2d, block2d](target_field, norm, p_i_d)
        cuda.synchronize()

    do_rk4_step_kernel[grid2d, block2d](k1, Velocity, Velocity, Temp, Field, Field, 0.0, p_i_d, p_f_d)
    maybe_normalize(Temp)
    gradient_step_kernel[grid2d, block2d](l1, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    do_rk4_step_kernel[grid2d, block2d](k2, Velocity, l1, Temp, Field, k1, 0.5, p_i_d, p_f_d)
    maybe_normalize(Temp)
    gradient_step_kernel[grid2d, block2d](l2, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    do_rk4_step_kernel[grid2d, block2d](k3, Velocity, l2, Temp, Field, k2, 0.5, p_i_d, p_f_d)
    maybe_normalize(Temp)
    gradient_step_kernel[grid2d, block2d](l3, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    do_rk4_step_kernel[grid2d, block2d](k4, Velocity, l3, Temp, Field, k3, 1.0, p_i_d, p_f_d)
    maybe_normalize(Temp)
    gradient_step_kernel[grid2d, block2d](l4, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    rk4_kernel[grid2d, block2d](Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i_d, p_f_d)
    cuda.synchronize()
    maybe_normalize(Field)
    new_energy = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)
    killkinen = p_i_h[7]
    # arrest = arresting_criteria(Velocity, EnergyGradient, en, entmp, gridsum_partial, p_i_d, p_i_h)
    if killkinen and (new_energy > prev_energy):
        set_field_zero_kernel[grid2d, block2d](Velocity, p_i_d)
        cuda.synchronize()
    # if killkinen and (arrest < 0):
    #     set_field_zero_kernel[grid2d, block2d](Velocity, p_i_d)
    #     cuda.synchronize()
    err = compute_max_field(EnergyGradient, max_partial, p_i_h)
    return new_energy, err