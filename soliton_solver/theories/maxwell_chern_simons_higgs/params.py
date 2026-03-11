"""
Anyon superconductor theory-specific parameters, parameter resolution, device packing, and terminal parameter documentation.

This module extends the core Params and ResolvedParams classes with the theory-specific parameters required by the Maxwell-Chern-Simons-Higgs model.
It preserves the existing CUDA ABI layout by appending theory-specific entries to the core integer and floating-point device parameter arrays.
The module also provides a describe() function so that the anyon superconductor theory can print human-readable parameter information through theory.describe().

Core prefix
-----------
From soliton_solver.core.params.pack_device_params:
- p_i[0..9]
- p_f[0..5]

Maxwell-Chern-Simons-Higgs appended entries
-------------------------------------------
- p_i[10]    number_higgs_fields
- p_i[11]    number_gauge_fields

- p_f[6]     q
- p_f[7]     Lambda
- p_f[8]     kappa
- p_f[9]     u1
- p_f[10]    vortex_number
- p_f[11]    ainf

Examples
--------
>>> from soliton_solver.theories.maxwell_chern_simons_higgs.params import Params, default_params
>>> p = default_params(q=1.0, Lambda=1.0, kappa=0.5, vortex_number=2.0)
>>> rp = p.resolved()
>>> p_i, p_f = pack_device_params(rp)
>>> describe()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soliton_solver.core.params import Params as CoreParams
from soliton_solver.core.params import ResolvedParams as CoreResolvedParams
from soliton_solver.core.params import pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    User-facing Maxwell-Chern-Simons-Higgs parameters.

    This class extends the core solver parameters with Maxwell-Chern-Simons-Higgs matter and gauge-field parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields stored by the solver.
        For the Maxwell-Chern-Simons-Higgs model this defaults to 5.
    number_higgs_fields : int, optional
        Number of Higgs field components.
        This defaults to 2.
    number_gauge_fields : int, optional
        Number of gauge-field components.
        This defaults to 2.
    q : float, optional
        Gauge coupling.
    Lambda : float, optional
        Higgs self-coupling or potential strength parameter.
    kappa : float, optional
        Chern-Simons coupling.
    u1 : float, optional
        Higgs vacuum amplitude scale.
    vortex_number : float, optional
        Winding number used by the vortex initial condition.
    ainf : float | None, optional
        Asymptotic gauge-field value.
        If None, it is derived from vortex_number and q.

    Examples
    --------
    >>> p = Params()
    >>> p = Params(q=1.0, Lambda=2.0, kappa=0.5, u1=1.0, vortex_number=2.0)
    >>> rp = p.resolved()
    """

    number_total_fields: int = 5

    number_higgs_fields: int = 2
    number_gauge_fields: int = 2
    q: float = 1.0
    Lambda: float = 1.0
    kappa: float = 0.0
    u1: float = 1.0

    vortex_number: float = 1.0
    ainf: float | None = None

    def resolved(self) -> "ResolvedParams":
        """
        Convert user-facing parameters into fully resolved Maxwell-Chern-Simons-Higgs parameters.

        Returns
        -------
        ResolvedParams
            Resolved parameter object containing both core derived quantities and Maxwell-Chern-Simons-Higgs theory-specific derived quantities.

        Examples
        --------
        >>> p = Params(q=1.0, vortex_number=2.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved Maxwell-Chern-Simons-Higgs parameters.

    This class contains the full set of core resolved parameters together with Maxwell-Chern-Simons-Higgs theory-specific coefficients needed by the CPU and GPU solver code.

    Parameters
    ----------
    number_higgs_fields : int
        Number of Higgs field components.
    number_gauge_fields : int
        Number of gauge-field components.
    q : float
        Gauge coupling.
    Lambda : float
        Higgs self-coupling or potential strength parameter.
    kappa : float
        Chern-Simons coupling.
    u1 : float
        Higgs vacuum amplitude scale.
    vortex_number : float
        Winding number used in the vortex sector.
    ainf : float
        Asymptotic gauge-field value.

    Examples
    --------
    >>> p = Params(q=1.0, Lambda=1.0, kappa=0.5, vortex_number=1.0)
    >>> rp = ResolvedParams.from_params(p)
    """

    number_higgs_fields: int
    number_gauge_fields: int

    q: float
    Lambda: float
    kappa: float
    u1: float
    vortex_number: float
    ainf: float

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved Maxwell-Chern-Simons-Higgs parameters from user-facing parameters.

        This method computes the asymptotic gauge value required by the rest of the code when it is not supplied explicitly.

        Parameters
        ----------
        p : Params
            User-facing Maxwell-Chern-Simons-Higgs parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved Maxwell-Chern-Simons-Higgs parameters.

        Examples
        --------
        >>> p = Params(q=1.0, Lambda=1.0, kappa=0.5, vortex_number=2.0)
        >>> rp = ResolvedParams.from_params(p)
        """
        core = CoreResolvedParams.from_params(p)
        ainf = p.ainf if p.ainf is not None else (p.vortex_number / p.q)

        return ResolvedParams(xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields, dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization, xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step, number_higgs_fields=p.number_higgs_fields, number_gauge_fields=p.number_gauge_fields, q=float(p.q), Lambda=float(p.Lambda), kappa=float(p.kappa), u1=float(p.u1), vortex_number=float(p.vortex_number), ainf=float(ainf))


def default_params(**overrides) -> Params:
    """
    Construct Maxwell-Chern-Simons-Higgs parameters using defaults plus user overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to Params.with_().

    Returns
    -------
    Params
        Parameter object with the requested overrides applied.

    Examples
    --------
    >>> p = default_params(q=1.0, Lambda=2.0, kappa=0.5, vortex_number=2.0)
    """
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved Maxwell-Chern-Simons-Higgs parameters into device ABI arrays.

    The core integer and floating-point parameter arrays are created first and the Maxwell-Chern-Simons-Higgs theory-specific entries are then appended.
    This preserves the ABI indices expected by the Maxwell-Chern-Simons-Higgs kernels.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved Maxwell-Chern-Simons-Higgs parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple (p_i, p_f) containing the integer and floating-point device parameter arrays.

    Examples
    --------
    >>> rp = default_params().resolved()
    >>> p_i, p_f = pack_device_params(rp)
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([rp.number_higgs_fields, rp.number_gauge_fields], dtype=np.int32)

    p_f_theory = np.array([
        rp.q,
        rp.Lambda,
        rp.kappa,
        rp.u1,
        rp.vortex_number,
        rp.ainf,
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f


def describe() -> None:
    """
    Print a human-readable description of the Maxwell-Chern-Simons-Higgs parameter set.

    The printed output is intended for interactive terminal use through theory.describe().
    It summarizes the field content, theory-specific model parameters, derived gauge quantities, and the meaning of the packed device arrays.

    Returns
    -------
    None
        This function prints parameter information to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories.maxwell_chern_simons_higgs import params
    >>> params.describe()
    """
    print("Field content:")
    print("  The Maxwell-Chern-Simons-Higgs model uses 1 complex superconducting field, 2 spatial gauge fields and 1 temporal gauge potential by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Model couplings:")
    print("  q : gauge coupling.")
    print("  Lambda : Higgs self-coupling or potential strength parameter.")
    print("  kappa : Chern-Simons coupling.")
    print("  u1 : Higgs vacuum amplitude scale.")
    print()

    print("Vortex and gauge controls:")
    print("  vortex_number : winding number used by the vortex initial condition.")
    print("  ainf : asymptotic gauge-field value.")
    print("  If ainf is left as None, it is derived as vortex_number / q.")
    print()