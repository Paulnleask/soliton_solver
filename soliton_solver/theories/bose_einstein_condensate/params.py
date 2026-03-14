"""
Theory parameters for the Bose-Einstein condensate model.

Examples
--------
Use ``default_params`` to create a parameter set and ``pack_device_params`` to build the device arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soliton_solver.core.params import Params as CoreParams
from soliton_solver.core.params import ResolvedParams as CoreResolvedParams
from soliton_solver.core.params import pack_device_params as pack_core_device_params
import math

@dataclass(frozen=True)
class Params(CoreParams):
    """
    User-facing parameters for the Bose-Einstein condensate model.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of field components.
    number_higgs_fields : int, optional
        Number of Higgs field components.
    N : float, optional
        Total particle number.
    a_0 : float, optional
        Bohr radius.
    a_s : float, optional
        Scattering length.
    m : float, optional
        Particle mass.
    omega : float, optional
        Trap frequency.
    omega_rot : float, optional
        Rotation frequency.
    hbar : float, optional
        Reduced Planck constant.
    beta : float or None, optional
        Interaction parameter.
    tf_radius : float or None, optional
        Thomas-Fermi radius.
    vortex_number : float, optional
        Vortex winding number.

    Returns
    -------
    None
        The dataclass stores the model parameters.

    Examples
    --------
    Use ``p = Params(vortex_number=2.0)`` to create a parameter set.
    Use ``rp = p.resolved()`` to derive the resolved parameters.
    """
    
    number_total_fields: int = 2
    number_higgs_fields: int = 2
    N: float = 1000.0
    a_0: float = 0.529e-10
    a_s: float = 109.0 * a_0
    m: float = 1.45e-25
    omega: float = 674.2 / (2.0 * math.pi)
    omega_rot: float = 0.99
    hbar: float = 1.054571817e-34
    beta: float | None = None
    tf_radius: float | None = None
    
    vortex_number: float = 1.0

    def resolved(self) -> "ResolvedParams":
        """
        Resolve the theory parameters.

        Returns
        -------
        ResolvedParams
            Resolved parameter object for the Bose-Einstein condensate model.

        Examples
        --------
        Use ``rp = p.resolved()`` to construct the resolved parameters.
        """
        return ResolvedParams.from_params(self)

@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Resolved parameters for the Bose-Einstein condensate model.

    Parameters
    ----------
    number_total_fields : int
        Total number of field components.
    number_higgs_fields : int
        Number of Higgs field components.
    N : float
        Total particle number.
    a_0 : float
        Bohr radius.
    a_s : float
        Scattering length.
    m : float
        Particle mass.
    omega : float
        Trap frequency.
    hbar : float
        Reduced Planck constant.
    omega_rot : float
        Rotation frequency.
    beta : float
        Interaction parameter.
    tf_radius : float
        Thomas-Fermi radius.
    vortex_number : float, optional
        Vortex winding number.

    Returns
    -------
    None
        The dataclass stores the resolved model parameters.

    Examples
    --------
    Use ``rp = ResolvedParams.from_params(p)`` to build the resolved parameter set.
    """

    number_total_fields: int
    number_higgs_fields: int
    N: float
    a_0: float
    a_s: float
    m: float
    omega: float
    hbar: float
    omega_rot: float
    beta: float
    tf_radius: float

    vortex_number: float = 1.0

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved parameters from a user-facing parameter set.

        Parameters
        ----------
        p : Params
            User-facing parameter set.

        Returns
        -------
        ResolvedParams
            Fully resolved parameter set.

        Examples
        --------
        Use ``rp = ResolvedParams.from_params(p)`` to derive the resolved parameters.
        """
        core = CoreResolvedParams.from_params(p)
        beta = p.beta if p.beta is not None else (4.0 * math.pi * p.N * p.a_s * math.sqrt(p.m * p.omega / p.hbar))
        tf_radius = p.tf_radius if p.tf_radius is not None else (math.sqrt(2.0 * math.sqrt(beta / math.pi)))

        return ResolvedParams(xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields, dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization, xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step, number_higgs_fields=p.number_higgs_fields, tf_radius=tf_radius, beta=beta, omega_rot=float(p.omega_rot), vortex_number=float(p.vortex_number), N=float(p.N),
        a_0=float(p.a_0),
        a_s=float(p.a_s),
        m=float(p.m),
        omega=float(p.omega),
        hbar=float(p.hbar),
)

def default_params(**overrides) -> Params:
    """
    Create a default parameter set with optional overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments passed to ``Params.with_``.

    Returns
    -------
    Params
        Parameter set with the requested overrides.

    Examples
    --------
    Use ``p = default_params(vortex_number=2.0)`` to create a modified default parameter set.
    """
    return Params().with_(**overrides)

def pack_device_params(rp: ResolvedParams):
    """
    Pack the resolved parameters into device arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Resolved parameter set.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer and floating-point device parameter arrays.

    Examples
    --------
    Use ``p_i, p_f = pack_device_params(rp)`` to build the device arrays.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([rp.number_higgs_fields], dtype=np.int32)

    p_f_theory = np.array([
        rp.beta,
        rp.tf_radius,
        rp.omega_rot,
        rp.vortex_number,
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f

def describe() -> None:
    """
    Print a summary of the Bose-Einstein condensate parameters.

    Returns
    -------
    None
        The parameter summary is printed to the terminal.

    Examples
    --------
    Use ``describe()`` to print the parameter summary.
    """
    print("Field content:")
    print("  The Bose-Einstein condensate model uses 1 complex condensate field by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Model parameters:")
    print("  N : total particle number.")
    print("  a_0 : Bohr radius.")
    print("  a_s : scattering length.")
    print("  m : particle mass.")
    print("  omega : trap frequency.")
    print("  omega_rot : rotation frequency.")
    print("  hbar : reduced Planck constant.")
    print()

    print("Derived parameters:")
    print("  beta : interaction parameter.")
    print("  tf_radius : Thomas-Fermi radius.")
    print("  vortex_number : winding number used by the initial condition.")
    print()