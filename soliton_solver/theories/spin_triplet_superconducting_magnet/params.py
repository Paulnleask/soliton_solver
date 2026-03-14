"""
Parameters for the spin triplet superconducting ferromagnet theory.

Examples
--------
>>> p = default_params(q=1.0, vortex1_number=1.0, vortex2_number=2.0, ansatz="bloch")
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
    User-facing parameters for the spin triplet superconducting ferromagnet theory.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields stored by the solver.
    number_magnetization_fields : int, optional
        Number of magnetization field components.
    number_higgs_fields : int, optional
        Number of superconducting Higgs field components.
    number_gauge_fields : int, optional
        Number of gauge field components.
    q : float, optional
        Gauge coupling.
    alpha : float, optional
        Quadratic magnetic coefficient.
    beta : float, optional
        Quartic magnetic coefficient.
    gamma : float, optional
        Magnetic gradient coefficient.
    ha : float, optional
        Quadratic superconducting coefficient.
    hb1 : float, optional
        First quartic superconducting coefficient.
    hb2 : float, optional
        Second quartic superconducting coefficient.
    hc : float, optional
        Coupling between the superconducting components.
    M0 : float or None, optional
        Magnetic vacuum scale.
    u1 : float or None, optional
        Vacuum amplitude of the first superconducting component.
    u2 : float or None, optional
        Vacuum amplitude of the second superconducting component.
    vortex_number : float, optional
        Effective total vortex number.
    vortex1_number : float, optional
        Winding number of the first superconducting component.
    vortex2_number : float, optional
        Winding number of the second superconducting component.
    ainf : float or None, optional
        Asymptotic gauge field value.
    skyrmion_number : float, optional
        Skyrmion number used in the initial ansatz.
    skyrmion_rotation : float, optional
        Rotation angle used in the initial ansatz.
    ansatz : str, optional
        Initial ansatz type.

    Examples
    --------
    >>> p = Params(q=1.0, hb1=3.0, hb2=0.5, hc=-0.5, vortex1_number=1.0, vortex2_number=2.0, ansatz="neel")
    >>> rp = p.resolved()
    """
    number_total_fields: int = 10

    number_magnetization_fields: int = 3
    number_higgs_fields: int = 4
    number_gauge_fields: int = 3
    q: float = 1.0
    alpha: float = -1.0
    beta: float = 1.0
    gamma: float = 1.0
    ha: float = -4.0
    hb1: float = 3.0
    hb2: float = 0.5
    hc: float = -0.5

    M0: float | None = None
    u1: float | None = None
    u2: float | None = None
    vortex_number: float = 1.0
    vortex1_number: float = 1.0
    vortex2_number: float = 1.0
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
            Fully resolved parameter set.

        Examples
        --------
        >>> p = Params(q=1.0, vortex1_number=1.0, vortex2_number=2.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)

@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved parameters for the spin triplet superconducting ferromagnet theory.

    Parameters
    ----------
    number_magnetization_fields : int
        Number of magnetization field components.
    number_higgs_fields : int
        Number of superconducting Higgs field components.
    number_gauge_fields : int
        Number of gauge field components.
    q : float
        Gauge coupling.
    alpha : float
        Quadratic magnetic coefficient.
    beta : float
        Quartic magnetic coefficient.
    gamma : float
        Magnetic gradient coefficient.
    skyrmion_number : float
        Skyrmion number used in the initial ansatz.
    ha : float
        Quadratic superconducting coefficient.
    hb1 : float
        First quartic superconducting coefficient.
    hb2 : float
        Second quartic superconducting coefficient.
    hc : float
        Coupling between the superconducting components.
    M0 : float
        Magnetic vacuum scale.
    u1 : float
        Vacuum amplitude of the first superconducting component.
    u2 : float
        Vacuum amplitude of the second superconducting component.
    vortex_number : float
        Effective total vortex number.
    vortex1_number : float
        Winding number of the first superconducting component.
    vortex2_number : float
        Winding number of the second superconducting component.
    ainf : float
        Asymptotic gauge field value.
    skyrmion_rotation : float
        Rotation angle used in the initial ansatz.
    ansatz_bloch : bool
        Whether the Bloch ansatz is enabled.
    ansatz_neel : bool
        Whether the Neel ansatz is enabled.
    ansatz_anti : bool
        Whether the anti-skyrmion ansatz is enabled.
    ansatz_uniform : bool
        Whether the uniform ansatz is enabled.

    Examples
    --------
    >>> p = Params(q=1.0, vortex1_number=1.0, vortex2_number=2.0)
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
    hb1: float
    hb2: float
    hc: float
    M0: float
    u1: float
    u2: float
    vortex_number: float
    vortex1_number: float
    vortex2_number: float
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
            User-facing parameter set.

        Returns
        -------
        ResolvedParams
            Fully resolved parameter set.

        Examples
        --------
        >>> p = Params(q=1.0, vortex1_number=1.0, vortex2_number=2.0, ansatz="bloch")
        >>> rp = ResolvedParams.from_params(p)
        """
        core = CoreResolvedParams.from_params(p)
        denom = (p.hb1 + 2.0 * p.hb2)

        if p.M0 is None or p.u1 is None or p.u2 is None:
            if denom != 0.0:
                M0 = math.sqrt(-1.0 * p.alpha / p.beta)
                u1 = math.sqrt(-1.0 * (p.ha + 2.0 * p.hc) / denom)
                u2 = math.sqrt(-1.0 * (p.ha + 2.0 * p.hc) / denom)
            else:
                M0 = 1.0
                u1 = 1.0
                u2 = 1.0
        else:
            M0 = float(p.M0)
            u1 = float(p.u1)
            u2 = float(p.u2)

        vortex_number = (u1 * u1 * p.vortex1_number + u2 * u2 * p.vortex2_number) / (u1 * u1 + u2 * u2)
        ainf = p.ainf if p.ainf is not None else (vortex_number / p.q)

        ans = (p.ansatz or "bloch").lower()
        ansatz_bloch = ans == "bloch"
        ansatz_neel = ans == "neel"
        ansatz_anti = ans == "anti"
        ansatz_uniform = ans == "uniform"

        return ResolvedParams(
            xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization,
            xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step,
            number_magnetization_fields=p.number_magnetization_fields, number_higgs_fields=p.number_higgs_fields, number_gauge_fields=p.number_gauge_fields,
            q=float(p.q), alpha=float(p.alpha), beta=float(p.beta), gamma=float(p.gamma), skyrmion_number=float(p.skyrmion_number),
            ha=float(p.ha), hb1=float(p.hb1), hb2=float(p.hb2), hc=float(p.hc), M0=float(M0), u1=float(u1), u2=float(u2),
            vortex_number=float(vortex_number), vortex1_number=float(p.vortex1_number), vortex2_number=float(p.vortex2_number), ainf=float(ainf),
            skyrmion_rotation=float(p.skyrmion_rotation), ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform
        )

def default_params(**overrides) -> Params:
    """
    Construct default theory parameters with optional overrides.

    Parameters
    ----------
    **overrides
        Keyword overrides passed to ``Params.with_``.

    Returns
    -------
    Params
        Parameter set with overrides applied.

    Examples
    --------
    >>> p = default_params(q=1.0, vortex1_number=1.0, vortex2_number=2.0, ansatz="neel")
    """
    return Params().with_(**overrides)

def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved parameters into device arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved parameter set.

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
        rp.hb1,
        rp.hb2,
        rp.hc,
        rp.u1,
        rp.u2,
        rp.vortex1_number,
        rp.vortex2_number,
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
    Print parameter information for the theory.

    Examples
    --------
    >>> from soliton_solver.theories.spin_triplet_superconducting_magnet import params
    >>> params.describe()
    """
    print("Field content:")
    print("  The spin triplet superconducting ferromagnet model uses 3 magnetization fields, 3 gauge fields and 2 complex superconducting fields by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Model couplings:")
    print("  q : gauge coupling.")
    print("  alpha : quadratic magnetic-sector coefficient.")
    print("  beta : quartic magnetic-sector coefficient.")
    print("  gamma : magnetic-sector coupling coefficient.")
    print("  ha : quadratic superconducting-sector coefficient.")
    print("  hb1 : first quartic superconducting-sector coefficient.")
    print("  hb2 : second quartic superconducting-sector coefficient.")
    print("  hc : coupling between the two superconducting components.")
    print()

    print("Derived vacuum scales:")
    print("  M0 : magnetic vacuum scale.")
    print("  u1 : vacuum amplitude of the first superconducting component.")
    print("  u2 : vacuum amplitude of the second superconducting component.")
    print("  If M0, u1, or u2 are left as None, they are derived from alpha, beta, ha, hb1, hb2, and hc when possible.")
    print("  If hb1 + 2 hb2 is zero and these quantities are not supplied explicitly, they default to 1.0.")
    print()

    print("Vortex and gauge controls:")
    print("  vortex1_number : winding number associated with the first superconducting component.")
    print("  vortex2_number : winding number associated with the second superconducting component.")
    print("  vortex_number : effective total vortex number computed from u1, u2, vortex1_number, and vortex2_number.")
    print("  ainf : asymptotic gauge-field value.")
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