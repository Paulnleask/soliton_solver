# =====================================================================================
# soliton_solver/core/integrator.py
# =====================================================================================
"""
Time stepping / relaxation kernels for the soliton_solver simulation (CUDA + host driver).

This module is focused on minimization:
- CUDA kernels for computing energy gradients and Runge-Kutta (RK4) updates
- A host-side "arrested Newton flow" routine that:
    1) builds RK4 slopes by repeatedly evaluating the energy gradient,
    2) advances (Field, Velocity),
    3) optionally "arrests" the flow by zeroing Velocity if energy increased,
    4) reports the new energy and a max-norm error estimate.

Typical usage (high level):
- Allocate device arrays: Field, Velocity, derivative buffers (d1fd1x, d2fd2x), RK buffers (k1..k4, l1..l4),
  temporary field Temp, and an EnergyGradient array.
- Call do_arrested_newton_flow(...) in a loop until `err` is small.

Important conventions:
- All CUDA kernels iterate over (x, y) threads and then loop over field components `a`.
- `idx_field(a, x, y, p_i)` maps component+site to a flat index.
- `p_i_*` and `p_f_*` are parameter arrays (int/float) with specific slots used here:
    p_i[4] : number_total_fields
    p_i[7] : killkinen flag (host-side checked)
    p_i[9] : unit_magnetization constraint flag (device-side checked)
    p_f[5] : dt (time/flow step)
"""
# ---------------- Imports ----------------
from numba import cuda
from soliton_solver.core.utils import idx_field, in_bounds, launch_2d, set_field_zero_kernel, compute_max_field
from soliton_solver.core.derivatives import compute_derivative_first, compute_derivative_second

# ---------------- Dependency injection of gradient function ----------------
def make_do_gradient_step_kernel(do_gradient_step_point):
    """
    Create a CUDA kernel that computes spatial derivatives and applies a
    theory-specific gradient step at each lattice site.

    Usage:
        gradient_step_kernel = make_do_gradient_step_kernel(do_gradient_step_point)
        gradient_step_kernel[grid2d, block2d](Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, p_i, p_f)

    Parameters:
        do_gradient_step_point:
            CUDA device function implementing the per-site gradient update.

    Outputs:
        Returns a CUDA kernel. When launched, it:
            - Computes first and second spatial derivatives of Field.
            - Updates Velocity according to the energy gradient.
            - Writes the local energy gradient into EnergyGradient.
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

# ---------------- Update 4th order Runge-Kutta slopes ----------------
@cuda.jit
def do_rk4_step_kernel(k_out, Velocity, l_in, Temp, Field, k_prev, factor, p_i, p_f):
    """
    Compute one RK4 position slope and corresponding intermediate field.

    Usage:
        do_rk4_step_kernel[grid2d, block2d](...)

    Parameters:
        k_out: Device array to store the RK4 position slope.
        Velocity: Device array of current velocities.
        l_in: Device array of acceleration slopes.
        Temp: Device array for intermediate field values.
        Field: Device array of current field values.
        k_prev: Previous RK4 position slope.
        factor: RK stage factor (0.0, 0.5, 1.0).
        p_i, p_f: Device parameter arrays.

    Outputs:
        - Writes the RK4 slope into k_out.
        - Writes the intermediate field configuration into Temp.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    dt = p_f[5]
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        k_out[idx_field(a, x, y, p_i)] = dt * (Velocity[idx_field(a, x, y, p_i)] + factor * l_in[idx_field(a, x, y, p_i)])
        Temp[idx_field(a, x, y, p_i)] = Field[idx_field(a, x, y, p_i)] + factor * k_prev[idx_field(a, x, y, p_i)]

# ---------------- Do 4th order Runge-Kutta ----------------
@cuda.jit
def do_rk4_kernel_no_constraint(Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i, p_f):
    """
    Finalize an RK4 update without constraint projection.

    Usage:
        do_rk4_kernel_no_constraint[grid2d, block2d](...)

    Parameters:
        Velocity: Device array of velocities.
        Field: Device array of fields.
        k1..k4: RK4 position slopes.
        l1..l4: RK4 velocity slopes.
        p_i, p_f: Device parameter arrays.

    Outputs:
        - Updates Velocity in-place using RK4 combination of l1..l4.
        - Updates Field in-place using RK4 combination of k1..k4.
    """
    x, y = cuda.grid(2)
    if not in_bounds(x, y, p_i):
        return
    number_total_fields = p_i[4]
    for a in range(number_total_fields):
        Velocity[idx_field(a, x, y, p_i)] += (1.0 / 6.0) * (l1[idx_field(a, x, y, p_i)] + 2.0 * l2[idx_field(a, x, y, p_i)] + 2.0 * l3[idx_field(a, x, y, p_i)] + l4[idx_field(a, x, y, p_i)])
        Field[idx_field(a, x, y, p_i)] += (1.0 / 6.0) * (k1[idx_field(a, x, y, p_i)] + 2.0 * k2[idx_field(a, x, y, p_i)] + 2.0 * k3[idx_field(a, x, y, p_i)] + k4[idx_field(a, x, y, p_i)])

# ---------------- Do 4th order Runge-Kutta with unitary constraint ----------------
def make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization):
    """
    Create an RK4 finalize kernel with optional unit-magnetization constraint.

    Usage:
        rk4_kernel = make_do_rk4_kernel(compute_norm_magnetization, project_orthogonal_magnetization)
        rk4_kernel[grid2d, block2d](...)

    Parameters:
        compute_norm_magnetization:
            CUDA device function computing per-site magnetization norm.
        project_orthogonal_magnetization:
            CUDA device function projecting Velocity to satisfy the constraint.

    Outputs:
        Returns a CUDA kernel. When launched, it:
            - Updates Velocity and Field via RK4.
            - Enforces unit-magnetization constraint if p_i[9] is set.
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

# ---------------- Do arrested Newton flow ----------------
def do_arrested_newton_flow(Velocity, Field, d1fd1x, d2fd2x, EnergyGradient, k1, k2, k3, k4, l1, l2, l3, l4, Temp, en, entmp, gridsum_partial, max_partial, p_i_d, p_f_d, p_i_h, p_f_h, prev_energy, compute_energy, gradient_step_kernel, rk4_kernel):
    """
    Perform one arrested Newton flow step using RK4 integration.

    Usage:
        new_energy, err = do_arrested_newton_flow(...)

    Parameters:
        Velocity, Field: Device arrays updated in-place.
        d1fd1x, d2fd2x: Derivative buffers.
        EnergyGradient: Gradient buffer.
        k1..k4, l1..l4: RK4 slope buffers.
        Temp: Intermediate field buffer.
        en, entmp, gridsum_partial: Energy reduction buffers.
        max_partial: Max-reduction buffer.
        p_i_d, p_f_d: Device parameter arrays.
        p_i_h, p_f_h: Host parameter arrays.
        prev_energy: Previous scalar energy.
        compute_energy: Host energy evaluation function.
        gradient_step_kernel: CUDA gradient kernel.
        rk4_kernel: CUDA RK4 finalize kernel.

    Outputs:
        - Updates Field and Velocity by one RK4 step.
        - Zeros Velocity if energy increased and killkinen flag is set.
        - Computes and returns:
            new_energy: scalar energy after the step.
            err: max-norm of EnergyGradient.
    """
    grid2d, block2d = launch_2d(p_i_h, threads=(16, 16))
    do_rk4_step_kernel[grid2d, block2d](k1, Velocity, Velocity, Temp, Field, Field, 0.0, p_i_d, p_f_d)
    gradient_step_kernel[grid2d, block2d](l1, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    do_rk4_step_kernel[grid2d, block2d](k2, Velocity, l1, Temp, Field, k1, 0.5, p_i_d, p_f_d)
    gradient_step_kernel[grid2d, block2d](l2, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    do_rk4_step_kernel[grid2d, block2d](k3, Velocity, l2, Temp, Field, k2, 0.5, p_i_d, p_f_d)
    gradient_step_kernel[grid2d, block2d](l3, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    do_rk4_step_kernel[grid2d, block2d](k4, Velocity, l3, Temp, Field, k3, 1.0, p_i_d, p_f_d)
    gradient_step_kernel[grid2d, block2d](l4, Temp, d1fd1x, d2fd2x, EnergyGradient, p_i_d, p_f_d)
    rk4_kernel[grid2d, block2d](Velocity, Field, k1, k2, k3, k4, l1, l2, l3, l4, p_i_d, p_f_d)
    cuda.synchronize()
    new_energy = compute_energy(Field, d1fd1x, en, entmp, gridsum_partial, p_i_d, p_f_d, p_i_h, p_f_h)
    killkinen = p_i_h[7]
    if killkinen and (new_energy > prev_energy):
        set_field_zero_kernel[grid2d, block2d](Velocity, p_i_d)
        cuda.synchronize()
    err = compute_max_field(EnergyGradient, max_partial, p_i_h)
    return new_energy, err