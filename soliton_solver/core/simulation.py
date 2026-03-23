"""
High level Simulation driver for soliton_solver.

Examples
--------
Use ``Simulation(params, theory)`` to construct a simulation for a given theory.
Use ``initialize`` to build the grid and apply the initial configuration.
Use ``minimize`` to run arrested Newton flow minimization.
Use ``save_output`` to write simulation results to disk.
"""

from __future__ import annotations
import numpy as np
from numba import cuda
from pathlib import Path
import inspect
from soliton_solver.core.params import Params
from soliton_solver.core.utils import launch_2d, set_field_zero_kernel
from soliton_solver.core.integrator import do_arrested_newton_flow, do_rk4_kernel_no_constraint
import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

class Simulation:
    """
    High level simulation wrapper that binds a theory to the shared CUDA workflow.

    Parameters
    ----------
    params : Params or compatible object
        Parameter set used to configure the grid and solver.
    theory : object
        Theory object providing kernels, parameter packing, initialization, observables, and output routines.

    Examples
    --------
    Use ``sim = Simulation(params, theory)`` to construct the simulation wrapper.
    Use ``sim.initialize()`` to set up the initial state.
    Use ``result = sim.minimize(tol=1e-4)`` to minimize the configuration.
    Use ``sim.save_output()`` to write output files.
    """

    def __init__(self, params, theory):
        """
        Construct the simulation and allocate core device buffers.

        Parameters
        ----------
        params : Params or compatible object
            Parameter set used to configure the grid and solver.
        theory : object
            Theory object providing kernels, parameter packing, initialization, observables, and output routines.

        Returns
        -------
        None
            The simulation object is initialized with device buffers and parameter arrays.

        Raises
        ------
        AttributeError
            Raised if the theory does not provide ``do_gradient_step_kernel``.

        Examples
        --------
        Use ``sim = Simulation(params, theory)`` to create a simulation instance.
        """
        self.theory = theory
        self.kernels = theory.kernels
        self.initial_config = theory.initial_config
        self.observables_mod = theory.observables
        self.threads2d = getattr(self.theory, "threads2d", (16, 32))

        self.gradient_step_kernel = getattr(self.kernels, "do_gradient_step_kernel", None)
        if self.gradient_step_kernel is None:
            raise AttributeError("Theory kernels must provide do_gradient_step_kernel (typically built via soliton_solver.core.integrator.make_do_gradient_step_kernel).")

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
        Initialize the grid, zero the core fields, and apply the theory specific initial configuration.

        Parameters
        ----------
        init_config : dict or None, optional
            Optional configuration dictionary passed to the theory initialization routine.

        Returns
        -------
        None
            The grid, field, and velocity buffers are initialized in place.

        Examples
        --------
        Use ``sim.initialize()`` to initialize with default settings.
        Use ``sim.initialize(init_config={"seed": 123})`` to pass configuration to the theory initializer.
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
        Update the parameters and refresh the packed host and device parameter arrays.

        Parameters
        ----------
        p : Params or compatible object
            Parameter set describing the grid and solver configuration.

        Returns
        -------
        None
            The stored parameters and device parameter arrays are updated.

        Examples
        --------
        Use ``sim.set_params(Params(xlen=512, ylen=512, number_total_fields=6))`` to update the simulation parameters.
        """
        self.p = p
        self.rp = p if not isinstance(p, Params) else p.resolved()
        self.p_i_h, self.p_f_h = self.theory.params.pack_device_params(self.rp)
        self.p_i_d = cuda.to_device(self.p_i_h)
        self.p_f_d = cuda.to_device(self.p_f_h)

    def observables(self):
        """
        Compute scalar observables using theory provided host routines.

        Returns
        -------
        dict
            Dictionary containing the energy and any additional observables provided by the theory.

        Examples
        --------
        Use ``obs = sim.observables()`` to compute the current observables.
        Use ``energy = obs["energy"]`` to access the energy.
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
        Compute the per site energy density into the shared ``en`` buffer.

        Returns
        -------
        None
            The energy density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_energy_density()`` before copying ``sim.en`` to the host.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_energy_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def compute_higgs_density(self):
        """
        Compute the per site Higgs norm density into the shared ``en`` buffer.

        Returns
        -------
        None
            The Higgs density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_higgs_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_norm_higgs_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_higgs1_density(self):
        """
        Compute the per site Higgs 1 norm density into the shared ``en`` buffer.

        Returns
        -------
        None
            The Higgs 1 density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_higgs1_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_norm_higgs1_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_higgs2_density(self):
        """
        Compute the per site Higgs 2 norm density into the shared ``en`` buffer.

        Returns
        -------
        None
            The Higgs 2 density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_higgs2_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_norm_higgs2_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_phase_difference_density(self):
        """
        Compute the per site phase difference into the shared ``en`` buffer.

        Returns
        -------
        None
            The local phase difference is written into ``en``.

        Examples
        --------
        Use ``sim.compute_phase_difference_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_phase_difference_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        cuda.synchronize()

    def compute_magnetic_flux_density(self, which):
        """
        Compute a theory defined flux related density into the shared ``en`` buffer.

        Parameters
        ----------
        which : int
            Selector identifying which flux or vortex channel to compute.

        Returns
        -------
        None
            The selected density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_magnetic_flux_density(which=0)`` to compute a selected flux density.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_vortex_number_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, which, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def magnetic_field(self):
        """
        Compute the magnetic flux density field.

        Returns
        -------
        device array
            Device array containing the magnetic flux density.

        Examples
        --------
        Use ``B = sim.magnetic_field()`` to compute and retrieve the magnetic field buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_magnetic_field_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.MagneticFluxDensity, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        return self.MagneticFluxDensity
    
    def compute_magnetic_charge_density(self):
        """
        Compute the per site magnetic charge density into the shared ``en`` buffer.

        Returns
        -------
        None
            The magnetic charge density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_magnetic_charge_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_magnetic_charge_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.en, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def compute_electric_charge_density(self):
        """
        Compute the per site electric charge density into the shared ``en`` buffer.

        Returns
        -------
        None
            The electric charge density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_electric_charge_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_electric_charge_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.d2fd2x, self.en, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def compute_noether_charge_density(self):
        """
        Compute the per site Noether charge density into the shared ``en`` buffer.

        Returns
        -------
        None
            The Noether charge density is written into ``en``.

        Examples
        --------
        Use ``sim.compute_noether_charge_density()`` to populate the shared density buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_noether_charge_kernel[grid2d, block2d](self.Field, self.en, self.p_i_d, self.p_f_d)
        cuda.synchronize()

    def supercurrent(self):
        """
        Compute the supercurrent field.

        Returns
        -------
        device array
            Device array containing the supercurrent field.

        Examples
        --------
        Use ``J = sim.supercurrent()`` to compute and retrieve the supercurrent buffer.
        """
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        self.kernels.compute_supercurrent_kernel[grid2d, block2d](self.Field, self.d2fd2x, self.Supercurrent, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        return self.Supercurrent

    def step(self, prev_energy: float) -> tuple[float, float]:
        """
        Advance the simulation by one arrested Newton flow RK4 step.

        Parameters
        ----------
        prev_energy : float
            Energy from the previous step used by the arrest logic.

        Returns
        -------
        tuple of float
            Pair ``(new_energy, err)`` containing the updated energy and the maximum norm error estimate.

        Examples
        --------
        Use ``energy, err = sim.step(prev_energy=energy)`` to advance the simulation by one step.
        """
        compute_norm = getattr(self.observables_mod, "compute_norm", None)
        do_norm_kernel = getattr(self.kernels, "do_norm_kernel", None)
        new_energy, err = do_arrested_newton_flow(self.Velocity, self.Field, self.d1fd1x, self.d2fd2x, self.EnergyGradient, self.k1, self.k2, self.k3, self.k4, self.l1, self.l2, self.l3, self.l4, self.Temp, self.en, self.entmp, self.gridsum_partial, self.max_partial, self.p_i_d, self.p_f_d, self.p_i_h, self.p_f_h, prev_energy, self.observables_mod.compute_energy, self.gradient_step_kernel, self.rk4_kernel, compute_norm=compute_norm, do_norm_kernel=do_norm_kernel)
        return new_energy, err

    def save_output(self, output_dir: str | None = None, precision: int = 32):
        """
        Compute selected outputs, copy buffers to the host, and write the theory output bundle.

        Parameters
        ----------
        output_dir : str or None, optional
            Directory used for the written output files.
        precision : int, optional
            Number of significant digits used for text output formatting.

        Returns
        -------
        None
            The selected outputs are written to disk through the theory I/O routine.

        Examples
        --------
        Use ``sim.save_output(output_dir="output", precision=32)`` to write simulation output files.
        """
        if output_dir is None:
            theory_dir = Path(self.theory.io.__file__).resolve().parent
            output_dir = str(theory_dir / "results")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        h_Field = self.Field.copy_to_host()
        h_grid = self.grid.copy_to_host()
        
        grid2d, block2d = launch_2d(self.p_i_h, threads=self.threads2d)
        
        if hasattr(self.kernels, "compute_energy_kernel"):
            self.kernels.compute_energy_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, self.p_i_d, self.p_f_d)
        if hasattr(self.kernels, "compute_skyrmion_number_kernel"):
            self.kernels.compute_skyrmion_number_kernel[grid2d, block2d](self.entmp, self.Field, self.d1fd1x, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        h_EnergyDensity = self.en.copy_to_host() if hasattr(self, "en") else None
        h_BaryonDensity = self.entmp.copy_to_host() if hasattr(self, "entmp") else None

        if hasattr(self.kernels, "compute_norm_higgs_kernel"):
            self.kernels.compute_norm_higgs_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        if hasattr(self.kernels, "compute_magnetic_charge_kernel"):
            self.kernels.compute_magnetic_charge_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.entmp, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        h_HiggsDensity = self.en.copy_to_host() if hasattr(self, "en") else None
        h_MagneticChargeDensity = self.entmp.copy_to_host() if hasattr(self, "entmp") else None

        if hasattr(self.kernels, "compute_electric_charge_kernel"):
            self.kernels.compute_electric_charge_kernel[grid2d, block2d](self.Field, self.d1fd1x, self.d2fd2x, self.en, self.p_i_d, self.p_f_d)
        if hasattr(self.kernels, "compute_noether_charge_kernel"):
            self.kernels.compute_noether_charge_kernel[grid2d, block2d](self.Field, self.entmp, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        h_ElectricChargeDensity = self.en.copy_to_host() if hasattr(self, "en") else None
        h_NoetherChargeDensity = self.entmp.copy_to_host() if hasattr(self, "entmp") else None

        if hasattr(self.kernels, "compute_norm_higgs1_kernel"):
            self.kernels.compute_norm_higgs1_kernel[grid2d, block2d](self.en, self.Field, self.p_i_d)
        if hasattr(self.kernels, "compute_norm_higgs2_kernel"):
            self.kernels.compute_norm_higgs2_kernel[grid2d, block2d](self.entmp, self.Field, self.p_i_d)
        cuda.synchronize()
        h_Higgs1Density = self.en.copy_to_host() if hasattr(self, "en") else None
        h_Higgs2Density = self.entmp.copy_to_host() if hasattr(self, "entmp") else None

        if hasattr(self.kernels, "compute_vortex_number_kernel"):
            self.kernels.compute_vortex_number_kernel[grid2d, block2d](self.en, self.Field, self.d1fd1x, 2, self.p_i_d, self.p_f_d)
        cuda.synchronize()
        h_VortexDensity = self.en.copy_to_host() if hasattr(self, "en") else None

        if hasattr(self, "magnetic_field") and callable(getattr(self, "magnetic_field")):
            try:
                self.magnetic_field()
            except Exception:
                pass
        h_MagneticFluxDensity = self.MagneticFluxDensity.copy_to_host() if hasattr(self, "MagneticFluxDensity") else None

        if hasattr(self, "supercurrent") and callable(getattr(self, "supercurrent")):
            try:
                self.supercurrent()
            except Exception:
                pass
        h_Supercurrent = self.Supercurrent.copy_to_host() if hasattr(self, "Supercurrent") else None

        kwargs = dict(output_dir=output_dir,
                      h_Field=h_Field,
                      h_EnergyDensity=h_EnergyDensity,
                      h_MagneticFluxDensity=h_MagneticFluxDensity,
                      h_BaryonDensity=h_BaryonDensity,
                      h_VortexDensity=h_VortexDensity,
                      h_Supercurrent=h_Supercurrent,
                      h_HiggsDensity=h_HiggsDensity,
                      h_MagneticChargeDensity=h_MagneticChargeDensity,
                      h_ElectricChargeDensity=h_ElectricChargeDensity,
                      h_NoetherChargeDensity=h_NoetherChargeDensity,
                      h_Higgs1Density=h_Higgs1Density,
                      h_Higgs2Density=h_Higgs2Density,
                      h_grid=h_grid,
                      xlen=int(self.p_i_h[0]),
                      ylen=int(self.p_i_h[1]),
                      number_coordinates=int(self.p_i_h[3]),
                      number_total_fields=int(self.p_i_h[4]),
                      precision=precision)

        sig = inspect.signature(self.theory.io.output_data_bundle)
        accepted = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        self.theory.io.output_data_bundle(**filtered_kwargs)

    def print_instructions(self):
        """
        Print usage instructions for the loaded theory.

        Returns
        -------
        None
            Theory specific instructions are printed when available.

        Examples
        --------
        Use ``sim.print_instructions()`` to display theory specific guidance.
        """
        if hasattr(self.theory, "print_instructions"):
            self.theory.print_instructions()
        else:
            print("No specific instructions available for this theory.")