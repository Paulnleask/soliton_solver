# =====================================================================================
# soliton_solver/core/simulation.py
# =====================================================================================
"""
High-level Simulation driver for soliton_solver.

Purpose:
    Owns CUDA device buffers and orchestrates initialization, stepping, observable
    evaluation, and output writing for a given theory.

Usage:
    sim = Simulation(params, theory)
    sim.initialize()
    result = sim.minimize(tol=1e-4)
    sim.save_output("output")

Outputs:
    - Allocates and manages core device arrays (Field, Velocity, derivatives, RK buffers).
    - Runs theory-provided kernels for grid creation, gradients, and observables.
    - Produces minimization history and writes output bundles via theory I/O.
"""
# ---------------- Imports ----------------
from __future__ import annotations
import numpy as np
from numba import cuda
from soliton_solver.core.params import Params
from soliton_solver.core.utils import launch_2d, set_field_zero_kernel
from soliton_solver.core.integrator import do_arrested_newton_flow, do_rk4_kernel_no_constraint
import warnings
from numba.core.errors import NumbaPerformanceWarning

# ---------------- Switch off performance warnings ----------------
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ---------------- Main simulation class ----------------
class Simulation:
    """
    High-level simulation wrapper binding a theory to shared CUDA workflows.

    Usage:
        sim = Simulation(params, theory)
        sim.initialize()
        obs = sim.observables()
        result = sim.minimize(tol=1e-4)
        sim.save_output("output")

    Parameters:
        params: Params or ResolvedParams-like object used to configure grid and solver.
        theory: Theory module/object providing kernels, params packing, initialization, observables, and I/O.

    Outputs:
        - Allocates and stores CUDA device buffers and parameter arrays.
        - Provides convenience methods for initialization, stepping, diagnostics, and outputs.
    """

    def __init__(self, params, theory):
        """
        Construct a Simulation and allocate core device buffers.

        Usage:
            sim = Simulation(params, theory)

        Parameters:
            params: Params or ResolvedParams-like object used to configure grid and solver.
            theory: Object providing:
                - kernels (CUDA kernels, including do_gradient_step_kernel and create_grid_kernel)
                - params.pack_device_params(...)
                - initial_config.initialize(...)
                - observables (compute_energy and optional topological charges)
                - io.output_data_bundle(...)

        Outputs:
            - Binds theory modules (kernels, initial_config, observables, io).
            - Selects gradient kernel (required) and RK4 finalize kernel (optional; falls back to no-constraint RK4).
            - Packs host/device parameter arrays (p_i, p_f).
            - Allocates device arrays for:
                Field, Velocity, EnergyGradient, RK buffers (k1..k4, l1..l4), Temp,
                grid, derivative buffers (d1fd1x, d2fd2x), and reduction buffers.
            - Allocates optional theory-dependent buffers (MagneticFluxDensity, Supercurrent) if gauge fields are present.
        """
        self.theory = theory
        self.kernels = theory.kernels
        self.initial_config = theory.initial_config
        self.observables_mod = theory.observables
        self.threads2d = getattr(self.theory, "threads2d", (16, 32))

        self.gradient_step_kernel = getattr(self.kernels, "do_gradient_step_kernel", None)
        if self.gradient_step_kernel is None:
            raise AttributeError("Theory kernels must provide do_gradient_step_kernel (typically built via soliton_solver.core.integrator.make_do_gradient_step_kernel).")

        # RK4 finalize kernel is theory-provided if constraints are theory-owned
        self.rk4_kernel = getattr(self.kernels, "do_rk4_kernel", None)
        if self.rk4_kernel is None:
            self.rk4_kernel = do_rk4_kernel_no_constraint

        self.set_params(params)
        self.rp = params if not isinstance(params, Params) else params.resolved()

        p_i_h, p_f_h = self.theory.params.pack_device_params(self.rp)
        self.p_i_h = p_i_h
        self.p_f_h = p_f_h
        self.p_i_d = cuda.to_device(p_i_h)
        self.p_f_d = cuda.to_device(p_f_h)

        self.Field = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.Velocity = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.EnergyGradient = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.k1 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.k2 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.k3 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.k4 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.l1 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.l2 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.l3 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.l4 = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.Temp = cuda.device_array(self.rp.dim_fields, dtype=np.float64)
        self.grid = cuda.device_array(self.rp.number_coordinates * self.rp.dim_grid, dtype=np.float64)
        self.d1fd1x = cuda.device_array(self.rp.number_coordinates * self.rp.number_total_fields * self.rp.dim_grid, dtype=np.float64)
        self.d2fd2x = cuda.device_array(self.rp.number_coordinates * self.rp.number_coordinates * self.rp.number_total_fields * self.rp.dim_grid, dtype=np.float64)
        self.en = cuda.device_array(self.rp.dim_grid, dtype=np.float64)
        self.entmp = cuda.device_array(self.rp.dim_grid, dtype=np.float64)

        tpb = 1024
        self.gridsum_partial = cuda.device_array((self.rp.dim_grid + tpb - 1) // tpb, dtype=np.float64)
        self.max_partial = cuda.device_array((self.rp.dim_fields + tpb - 1) // tpb, dtype=np.float64)

        ng = int(getattr(self.rp, "number_gauge_fields", 0))
        self.MagneticFluxDensity = cuda.device_array(ng * self.rp.dim_grid, dtype=np.float64) if ng > 0 else cuda.device_array(1, dtype=np.float64)
        self.Supercurrent = cuda.device_array(ng * self.rp.dim_grid, dtype=np.float64) if ng > 0 else cuda.device_array(1, dtype=np.float64)

    def initialize(self, init_config: dict | None = None):
        """
        Initialize grid, zero core fields, and run theory-provided initial condition setup.

        Usage:
            sim.initialize()
            sim.initialize(init_config={"seed": 123, ...})

        Parameters:
            init_config: Optional dict passed to theory.initial_config.initialize(...).

        Outputs:
            - Writes the coordinate grid via theory.create_grid_kernel.
            - Zeros Field and Velocity on device.
            - Runs theory initial_config.initialize(...) to populate Field/Velocity and any staging buffers.
        """
        cfg = init_config or {}
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.create_grid_kernel[grid2d, block2d](self.grid, self.p_i_d, self.p_f_d)
        set_field_zero_kernel[grid2d, block2d](self.Field, self.p_i_d)
        set_field_zero_kernel[grid2d, block2d](self.Velocity, self.p_i_d)
        self.initial_config.initialize(Velocity=self.Velocity, Field=self.Field, grid=self.grid, k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4, l1=self.l1, l2=self.l2, l3=self.l3, l4=self.l4, Temp=self.Temp, p_i_d=self.p_i_d, p_f_d=self.p_f_d, p_i_h=self.p_i_h, p_f_h=self.p_f_h, grid2d=grid2d, block2d=block2d, config=cfg)
        cuda.synchronize()

    def set_params(self, p: Params):
        """
        Update parameters and refresh packed host/device parameter arrays.

        Usage:
            sim.set_params(Params(xlen=512, ylen=512, number_total_fields=6))

        Parameters:
            p: Params instance (or compatible object) describing grid/solver configuration.

        Outputs:
            - Stores p and updates resolved parameters (rp).
            - Recomputes packed parameter arrays (p_i_h, p_f_h) using theory params packing.
            - Uploads updated p_i and p_f to device (p_i_d, p_f_d).
        """
        self.p = p
        self.rp = p if not isinstance(p, Params) else p.resolved()
        self.p_i_h, self.p_f_h = self.theory.params.pack_device_params(self.rp)
        self.p_i_d = cuda.to_device(self.p_i_h)
        self.p_f_d = cuda.to_device(self.p_f_h)

    def observables(self):
        """
        Compute scalar observables using theory-provided host routines.

        Usage:
            obs = sim.observables()
            energy = obs["energy"]

        Parameters:
            None

        Outputs:
            - Computes and returns a dict containing:
                - "energy" (always)
                - Optional keys if present on theory.observables:
                    "skyrmion_number", "vortex_number", "magnetic_charge", "electric_charge"
        """
        obs = {}
        obs["energy"] = self.observables_mod.compute_energy(self.Field, self.d1fd1x, self.en, self.entmp, self.gridsum_partial, self.p_i_d, self.p_f_d, self.p_i_h, self.p_f_h)
        if hasattr(self.observables_mod, "compute_skyrmion_number"):
            obs["skyrmion_number"] = self.observables_mod.compute_skyrmion_number(self.Field, self.d1fd1x, self.en, self.entmp, self.gridsum_partial, self.p_i_d, self.p_f_d, self.p_i_h)
        if hasattr(self.observables_mod, "compute_vortex_number"):
            obs["vortex_number"] = self.observables_mod.compute_vortex_number(self.Field, self.d1fd1x, self.en, self.entmp, self.gridsum_partial, 2, self.p_i_d, self.p_f_d, self.p_i_h)
        if hasattr(self.observables_mod, "compute_magnetic_charge"):
            obs["magnetic_charge"] = self.observables_mod.compute_magnetic_charge(self.Field, self.d1fd1x, self.en, self.entmp, self.gridsum_partial, self.p_i_d, self.p_f_d, self.p_i_h)
        if hasattr(self.observables_mod, "compute_electric_charge"):
            obs["electric_charge"] = self.observables_mod.compute_electric_charge(self.Field, self.d1fd1x, self.d2fd2x, self.en, self.entmp, self.gridsum_partial, self.p_i_d, self.p_f_d, self.p_i_h)
        return obs

    def compute_energy_density(self):
        """
        Compute per-site energy density into the shared buffer `en`.

        Usage:
            sim.compute_energy_density()
            h_en = sim.en.copy_to_host()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_energy_kernel to write energy density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_energy_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def compute_higgs_density(self):
        """
        Compute per-site Higgs norm density into the shared buffer `en`.

        Usage:
            sim.compute_higgs_density()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_norm_higgs_kernel to write density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_norm_higgs_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_higgs1_density(self):
        """
        Compute per-site Higgs-1 norm density into the shared buffer `en`.

        Usage:
            sim.compute_higgs1_density()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_norm_higgs1_kernel to write density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_norm_higgs1_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_higgs2_density(self):
        """
        Compute per-site Higgs-2 norm density into the shared buffer `en`.

        Usage:
            sim.compute_higgs2_density()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_norm_higgs2_kernel to write density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_norm_higgs2_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_magnetic_flux_density(self, which):
        """
        Compute a vortex/flux-related per-site quantity into the shared buffer `en`.

        Usage:
            sim.compute_magnetic_flux_density(which=0)

        Parameters:
            which: Theory-defined selector for which flux/vortex channel to compute.

        Outputs:
            - Launches theory.compute_vortex_number_kernel to write the selected density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_vortex_number_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, which, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def magnetic_field(self):
        """
        Compute magnetic flux density field into `MagneticFluxDensity`.

        Usage:
            B = sim.magnetic_field()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_magnetic_field_kernel to write `MagneticFluxDensity`.
            - Returns the device array `MagneticFluxDensity`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_magnetic_field_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.MagneticFluxDensity, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        return self.MagneticFluxDensity
    
    def compute_magnetic_charge_density(self):
        """
        Compute per-site magnetic charge density into the shared buffer `en`.

        Usage:
            sim.compute_magnetic_charge_density()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_magnetic_charge_kernel to write density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_magnetic_charge_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.en, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def compute_electric_charge_density(self):
        """
        Compute per-site electric charge density into the shared buffer `en`.

        Usage:
            sim.compute_electric_charge_density()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_electric_charge_kernel to write density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_electric_charge_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.d2fd2x, self.en, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def compute_noether_charge_density(self):
        """
        Compute per-site Noether charge density into the shared buffer `en`.

        Usage:
            sim.compute_noether_charge_density()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_noether_charge_kernel to write density into `en`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_noether_charge_kernel[grid2d, block2d](self.Field, self.en, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def supercurrent(self):
        """
        Compute supercurrent field into `Supercurrent`.

        Usage:
            J = sim.supercurrent()

        Parameters:
            None

        Outputs:
            - Launches theory.compute_supercurrent_kernel to write `Supercurrent`.
            - Returns the device array `Supercurrent`.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_supercurrent_kernel[grid2d, block2d](self.Field, self.d2fd2x, self.Supercurrent, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        return self.Supercurrent

    def step(self, prev_energy: float) -> tuple[float, float]:
        """
        Advance the simulation by one arrested Newton-flow RK4 step.

        Usage:
            energy, err = sim.step(prev_energy=energy)

        Parameters:
            prev_energy: Previous scalar energy used for energy-increase arrest logic.

        Outputs:
            - Updates Field and Velocity in-place on device via do_arrested_newton_flow.
            - Returns:
                new_energy: scalar energy after the step.
                err: max-norm error estimate from the energy gradient.
        """
        new_energy, err = do_arrested_newton_flow(self.Velocity, self.Field, self.d1fd1x, self.d2fd2x, self.EnergyGradient, self.k1, self.k2, self.k3, self.k4, self.l1, self.l2, self.l3, self.l4, self.Temp, self.en, self.entmp, self.gridsum_partial, self.max_partial, self.p_i_d, self.p_f_d, self.p_i_h, self.p_f_h, prev_energy, self.observables_mod.compute_energy, self.gradient_step_kernel, self.rk4_kernel)
        return new_energy, err

    def minimize(self, tol: float = 1e-4, max_steps: int = 2000, log_every: int = 100, verbose: bool = True) -> dict:
        """
        Run arrested Newton-flow minimization until tolerance or step limit is reached.

        Usage:
            result = sim.minimize(tol=1e-4, max_steps=2000, log_every=100, verbose=True)

        Parameters:
            tol: Stopping threshold on the max-norm error estimate.
            max_steps: Maximum number of minimization iterations.
            log_every: Record/print observables every this many steps.
            verbose: If True, prints progress logs.

        Outputs:
            - Iteratively advances the solver using step(...).
            - Periodically computes observables and appends to history.
            - Returns a dict with final values and history:
                "final_energy", "final_skyrmion", "final_vortex", "final_err", "steps", "history".
        """
        def _fmt(x):
            return "None" if x is None else f"{float(x):.6f}"

        obs0 = self.observables()
        energy = obs0["energy"]
        sk = obs0.get("skyrmion_number")
        vort = obs0.get("vortex_number")
        if verbose:
            print(f"Initial: energy={energy:.6f}, skyrmion={_fmt(sk)}, vortex={_fmt(vort)}")

        history = []
        err = float("inf")

        for step in range(max_steps):
            energy, err = self.step(energy)
            if (step % log_every) == 0:
                obs = self.observables()
                e = obs["energy"]
                s = obs.get("skyrmion_number")
                v = obs.get("vortex_number")
                history.append((step, e, s, v, err))
                if verbose:
                    t = step * float(self.p_f_h[5])
                    print(f"t={t:.3f} ({step}): err={err:.6e}, energy={e:.6f}, sk={_fmt(s)}, vort={_fmt(v)}")
            if err <= tol:
                break

        obsf = self.observables()
        e = obsf["energy"]
        s = obsf.get("skyrmion_number")
        v = obsf.get("vortex_number")
        if verbose:
            print(f"Final: energy={e:.6f}, skyrmion={_fmt(s)}, vortex={_fmt(v)}")

        return {"final_energy": e, "final_skyrmion": s, "final_vortex": v, "final_err": err, "steps": step + 1, "history": history}

    def save_output(self, output_dir: str = "output", precision: int = 32):
        """
        Compute selected densities/fields, copy buffers to host, and write theory output bundle.

        Usage:
            sim.save_output(output_dir="output", precision=32)

        Parameters:
            output_dir: Output directory passed to theory I/O.
            precision: Significant digits for formatting in text outputs.

        Outputs:
            - Optionally computes energy density and skyrmion density if kernels exist.
            - Attempts to compute magnetic field and supercurrent if available.
            - Copies Field, grid, and any computed densities/fields back to host.
            - Calls theory.io.output_data_bundle(...) to write the output files.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        if hasattr(self.kernels, "compute_energy_kernel"):
            self.kernels.compute_energy_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, self.p_i_d, self.p_f_d)
        if hasattr(self.kernels, "compute_skyrmion_number_kernel"):
            self.kernels.compute_skyrmion_number_kernel[grid2d, block2d](self.entmp, self.Field, self.d1fd1x, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        if hasattr(self, "magnetic_field") and callable(getattr(self, "magnetic_field")):
            try:
                self.magnetic_field()
            except Exception:
                pass
        if hasattr(self, "supercurrent") and callable(getattr(self, "supercurrent")):
            try:
                self.supercurrent()
            except Exception:
                pass
        h_Field = self.Field.copy_to_host()
        h_grid = self.grid.copy_to_host()
        h_EnergyDensity = self.en.copy_to_host() if hasattr(self, "en") else None
        h_BaryonDensity = self.entmp.copy_to_host() if hasattr(self, "entmp") else None
        h_MagneticFluxDensity = self.MagneticFluxDensity.copy_to_host() if hasattr(self, "MagneticFluxDensity") else None
        h_Supercurrent = self.Supercurrent.copy_to_host() if hasattr(self, "Supercurrent") else None
        self.theory.io.output_data_bundle(output_dir=output_dir, h_Field=h_Field, h_EnergyDensity=h_EnergyDensity, h_MagneticFluxDensity=h_MagneticFluxDensity, h_BaryonDensity=h_BaryonDensity, h_Supercurrent=h_Supercurrent, h_grid=h_grid, xlen=int(self.p_i_h[0]), ylen=int(self.p_i_h[1]), number_coordinates=int(self.p_i_h[3]), number_total_fields=int(self.p_i_h[4]), precision=precision)