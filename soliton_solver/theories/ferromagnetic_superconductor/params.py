"""
Ferromagnetic superconductor parameters and device packing.

Examples
--------
>>> p = default_params(q=1.0, vortex_number=2.0, ansatz="bloch")
>>> rp = p.resolved()
>>> p_i, p_f = pack_device_params(rp)
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from soliton_solver.core.params import Params as CoreParams
from soliton_solver.core.params import ResolvedParams as CoreResolvedParams
from soliton_solver.core.params import pack_device_params as pack_core_device_params

@dataclass(frozen=True)
class Params(CoreParams):
    """
    User-facing ferromagnetic superconductor parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of field components.
    number_magnetization_fields : int, optional
        Number of magnetization field components.
    number_higgs_fields : int, optional
        Number of Higgs field components.
    number_gauge_fields : int, optional
        Number of gauge field components.
    q : float, optional
        Gauge coupling.
    alpha : float, optional
        Quadratic Higgs-sector coefficient.
    beta : float, optional
        Quartic Higgs-sector coefficient.
    gamma : float, optional
        Magnetic-sector coupling coefficient.
    ha : float, optional
        Quadratic magnetic-sector coefficient.
    hb : float, optional
        Quartic magnetic-sector coefficient.
    eta1 : float, optional
        Primary coupling between the superconducting and magnetic sectors.
    eta2 : float or None, optional
        Secondary coupling between the superconducting and magnetic sectors.
    M0 : float or None, optional
        Magnetic vacuum scale.
    u1 : float or None, optional
        Superconducting vacuum scale.
    vortex_number : float, optional
        Vortex winding number.
    ainf : float or None, optional
        Asymptotic gauge-field value.
    skyrmion_number : float, optional
        Skyrmion number used in the initial ansatz.
    skyrmion_rotation : float, optional
        Rotation angle used in the initial ansatz.
    ansatz : str, optional
        Initial ansatz type.

    Returns
    -------
    None
        The dataclass stores the theory parameters.

    Examples
    --------
    >>> p = Params(q=1.0, vortex_number=2.0, ansatz="neel")
    >>> rp = p.resolved()
    """

    number_total_fields: int = 8

    number_magnetization_fields: int = 3
    number_higgs_fields: int = 2
    number_gauge_fields: int = 3
    q: float = 1.0
    alpha: float = -1.0
    beta: float = 1.0
    gamma: float = 1.0
    ha: float = -4.0
    hb: float = 1.0
    eta1: float = 0.0
    eta2: float | None = None

    M0: float | None = None
    u1: float | None = None
    vortex_number: float = 1.0
    ainf: float | None = None

    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"

    def resolved(self) -> "ResolvedParams":
        """
        Resolve the theory parameters.

        Returns
        -------
        ResolvedParams
            Resolved ferromagnetic superconductor parameters.

        Examples
        --------
        >>> p = Params(q=1.0, vortex_number=2.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)

@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Resolved ferromagnetic superconductor parameters.

    Parameters
    ----------
    number_magnetization_fields : int
        Number of magnetization field components.
    number_higgs_fields : int
        Number of Higgs field components.
    number_gauge_fields : int
        Number of gauge field components.
    q : float
        Gauge coupling.
    alpha : float
        Quadratic Higgs-sector coefficient.
    beta : float
        Quartic Higgs-sector coefficient.
    gamma : float
        Magnetic-sector coupling coefficient.
    skyrmion_number : float
        Skyrmion number used in the initial ansatz.
    ha : float
        Quadratic magnetic-sector coefficient.
    hb : float
        Quartic magnetic-sector coefficient.
    eta1 : float
        Primary coupling between the superconducting and magnetic sectors.
    eta2 : float
        Secondary coupling between the superconducting and magnetic sectors.
    M0 : float
        Magnetic vacuum scale.
    u1 : float
        Superconducting vacuum scale.
    vortex_number : float
        Vortex winding number.
    ainf : float
        Asymptotic gauge-field value.
    skyrmion_rotation : float
        Rotation angle used in the initial ansatz.
    ansatz_bloch : bool
        Whether the Bloch ansatz is selected.
    ansatz_neel : bool
        Whether the Néel ansatz is selected.
    ansatz_anti : bool
        Whether the anti-skyrmion ansatz is selected.
    ansatz_uniform : bool
        Whether the uniform ansatz is selected.

    Returns
    -------
    None
        The dataclass stores the resolved theory parameters.

    Examples
    --------
    >>> p = Params(q=1.0, eta1=0.1, vortex_number=1.0)
    >>> rp = ResolvedParams.from_params(p)
    """

    number_magnetization_fields: int
    number_higgs_fields: int
    number_gauge_fields: int

    q: float
    alpha: float
    beta: float
    gamma: float
    skyrmion_number: float
    ha: float
    hb: float
    eta1: float
    eta2: float
    M0: float
    u1: float
    vortex_number: float
    ainf: float

    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved parameters from user-facing parameters.

        Parameters
        ----------
        p : Params
            User-facing ferromagnetic superconductor parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved ferromagnetic superconductor parameters.

        Examples
        --------
        >>> p = Params(q=1.0, eta1=0.1, vortex_number=2.0, ansatz="bloch")
        >>> rp = ResolvedParams.from_params(p)
        """
        core = CoreResolvedParams.from_params(p)
        eta2 = p.eta2 if p.eta2 is not None else (1.0 / (-2.0 * p.ha)) * p.eta1
        denom = (p.hb * p.beta - 4.0 * p.eta1 * p.eta1)

        if p.M0 is None or p.u1 is None:
            if denom != 0.0:
                M0 = math.sqrt((2.0 * p.ha * p.eta1 - p.alpha * p.hb) / denom)
                u1 = math.sqrt((2.0 * p.alpha * p.eta1 - p.ha * p.beta) / denom)
            else:
                M0 = 1.0
                u1 = 1.0
        else:
            M0 = float(p.M0)
            u1 = float(p.u1)

        ainf = p.ainf if p.ainf is not None else (p.vortex_number / p.q)
        ans = (p.ansatz or "bloch").lower()
        ansatz_bloch = ans == "bloch"
        ansatz_neel = ans == "neel"
        ansatz_anti = ans == "anti"
        ansatz_uniform = ans == "uniform"

        return ResolvedParams(xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields, dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization, xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step, number_magnetization_fields=p.number_magnetization_fields, number_higgs_fields=p.number_higgs_fields, number_gauge_fields=p.number_gauge_fields, q=float(p.q), alpha=float(p.alpha), beta=float(p.beta), gamma=float(p.gamma), skyrmion_number=float(p.skyrmion_number), ha=float(p.ha), hb=float(p.hb), eta1=float(p.eta1), eta2=float(eta2), M0=float(M0), u1=float(u1), vortex_number=float(p.vortex_number), ainf=float(ainf), skyrmion_rotation=float(p.skyrmion_rotation), ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform)

def default_params(**overrides) -> Params:
    """
    Construct default theory parameters with overrides.

    Parameters
    ----------
    **overrides
        Keyword overrides passed to ``Params.with_``.

    Returns
    -------
    Params
        Parameter object with the requested overrides.

    Examples
    --------
    >>> p = default_params(q=1.0, vortex_number=2.0, ansatz="neel")
    """
    return Params().with_(**overrides)

def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved parameters into device arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved ferromagnetic superconductor parameters.

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

    p_i_theory = np.array([rp.number_magnetization_fields, rp.number_higgs_fields, rp.number_gauge_fields], dtype=np.int32)

    p_f_theory = np.array([
        rp.q,
        rp.ha,
        rp.hb,
        rp.eta1,
        rp.eta2,
        rp.u1,
        rp.vortex_number,
        rp.ainf,
        rp.alpha,
        rp.beta,
        rp.gamma,
        rp.M0,
        rp.skyrmion_number,
        rp.skyrmion_rotation,
        1.0 if rp.ansatz_bloch else 0.0,
        1.0 if rp.ansatz_neel else 0.0,
        1.0 if rp.ansatz_anti else 0.0,
        1.0 if rp.ansatz_uniform else 0.0
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f

def describe() -> None:
    """
    Print the theory parameter description.

    Returns
    -------
    None
        The parameter description is printed to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories.ferromagnetic_superconductor import params
    >>> params.describe()
    """
    print("Field content:")
    print("  The ferromagnetic superconductor model consists of 3 magnetization fields, 3 gauge fields and 1 complex superconducting field by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Model couplings:")
    print("  q : gauge coupling.")
    print("  alpha : quadratic Higgs-sector coefficient.")
    print("  beta : quartic Higgs-sector coefficient.")
    print("  gamma : magnetic-sector coupling coefficient.")
    print("  ha : quadratic magnetic-sector coefficient.")
    print("  hb : quartic magnetic-sector coefficient.")
    print("  eta1 : primary coupling between the superconducting and magnetic sectors.")
    print("  eta2 : secondary coupling between the superconducting and magnetic sectors.")
    print("  If eta2 is left as None, it is derived as eta1 / (-2 ha).")
    print()

    print("Derived vacuum scales and gauge quantities:")
    print("  M0 : magnetic vacuum scale.")
    print("  u1 : superconducting vacuum scale.")
    print("  vortex_number : winding number used in the gauge-field sector.")
    print("  ainf : asymptotic gauge-field value.")
    print("  If M0 or u1 is left as None, they are derived from alpha, beta, ha, hb, and eta1 when the denominator hb * beta - 4 eta1^2 is non-zero.")
    print("  If that denominator is zero and M0 or u1 are not supplied explicitly, they default to 1.0.")
    print("  If ainf is left as None, it is derived as vortex_number / q.")
    print()

    print("Initial condition controls:")
    print("  skyrmion_number : topological charge used by the initial-condition ansatz.")
    print("  skyrmion_rotation : in-plane rotation angle used in the ansatz.")
    print("  ansatz : initial-condition type.")
    print("  Supported ansatz values:")
    print("    bloch")
    print("    neel")
    print("    anti")
    print("    uniform")
    print()