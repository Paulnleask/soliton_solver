"""
Parameters for the Maxwell-Chern-Simons-Higgs theory.

Examples
--------
>>> p = default_params(q=1.0, Lambda=1.0, kappa=0.5, vortex_number=2.0)
>>> rp = p.resolved()
>>> p_i, p_f = pack_device_params(rp)
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
    User-facing parameters for the Maxwell-Chern-Simons-Higgs theory.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of stored fields.
    number_higgs_fields : int, optional
        Number of Higgs field components.
    number_gauge_fields : int, optional
        Number of gauge field components.
    q : float, optional
        Gauge coupling.
    Lambda : float, optional
        Higgs self-coupling.
    kappa : float, optional
        Chern-Simons coupling.
    u1 : float, optional
        Higgs vacuum amplitude.
    vortex_number : float, optional
        Vortex winding number.
    ainf : float or None, optional
        Asymptotic gauge-field value.

    Returns
    -------
    None
        The dataclass stores the theory parameters.

    Examples
    --------
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
        Resolve the theory parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved core and theory-specific parameters.

        Examples
        --------
        >>> p = Params(q=1.0, vortex_number=2.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)

@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved parameters for the Maxwell-Chern-Simons-Higgs theory.

    Parameters
    ----------
    number_higgs_fields : int
        Number of Higgs field components.
    number_gauge_fields : int
        Number of gauge field components.
    q : float
        Gauge coupling.
    Lambda : float
        Higgs self-coupling.
    kappa : float
        Chern-Simons coupling.
    u1 : float
        Higgs vacuum amplitude.
    vortex_number : float
        Vortex winding number.
    ainf : float
        Asymptotic gauge-field value.

    Returns
    -------
    None
        The dataclass stores the resolved theory parameters.

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
        Build resolved parameters from user-facing parameters.

        Parameters
        ----------
        p : Params
            User-facing theory parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved theory parameters.

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
    Create default theory parameters with optional overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments passed to ``Params.with_``.

    Returns
    -------
    Params
        Parameter object with the requested overrides.

    Examples
    --------
    >>> p = default_params(q=1.0, Lambda=2.0, kappa=0.5, vortex_number=2.0)
    """
    return Params().with_(**overrides)

def pack_device_params(rp: ResolvedParams):
    """
    Pack the resolved parameters into device arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved theory parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer and floating-point device parameter arrays.

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
    Print the theory parameter description.

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