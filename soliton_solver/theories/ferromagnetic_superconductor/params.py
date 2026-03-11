"""
Ferromagnetic superconductor theory-specific parameters, parameter resolution, device packing, and terminal parameter documentation.

This module extends the core Params and ResolvedParams classes with the theory-specific parameters required by the ferromagnetic superconductor model.
It preserves the existing CUDA ABI layout by appending theory-specific entries to the core integer and floating-point device parameter arrays.
The module also provides a describe() function so that the ferromagnetic superconductor theory can print readable parameter information through theory.describe().

Core prefix
-----------
From soliton_solver.core.params.pack_device_params:
- p_i[0..9]
- p_f[0..5]

Ferromagnetic superconductor appended entries
---------------------------------------------
- p_i[10]    number_magnetization_fields
- p_i[11]    number_higgs_fields
- p_i[12]    number_gauge_fields

- p_f[6]     q
- p_f[7]     ha
- p_f[8]     hb
- p_f[9]     eta1
- p_f[10]    eta2
- p_f[11]    u1
- p_f[12]    vortex_number
- p_f[13]    ainf
- p_f[14]    alpha
- p_f[15]    beta
- p_f[16]    gamma
- p_f[17]    M0
- p_f[18]    skyrmion_number
- p_f[19]    skyrmion_rotation
- p_f[20]    ansatz_bloch
- p_f[21]    ansatz_neel
- p_f[22]    ansatz_anti
- p_f[23]    ansatz_uniform

Examples
--------
>>> from soliton_solver.theories.ferromagnetic_superconductor.params import Params, default_params
>>> p = default_params(q=1.0, vortex_number=2.0, ansatz="bloch")
>>> rp = p.resolved()
>>> p_i, p_f = pack_device_params(rp)
>>> describe()
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

    This class extends the core solver parameters with ferromagnetic superconductor couplings, vacuum scales, gauge-field controls, and initial-condition controls.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields stored by the solver.
        For the ferromagnetic superconductor model this defaults to 8.
    number_magnetization_fields : int, optional
        Number of magnetization field components.
        This defaults to 3.
    number_higgs_fields : int, optional
        Number of Higgs field components.
        This defaults to 2.
    number_gauge_fields : int, optional
        Number of gauge-field components.
        This defaults to 3.
    q : float, optional
        Gauge coupling.
    alpha : float, optional
        Quadratic Higgs-sector coefficient.
    beta : float, optional
        Quartic Higgs-sector coefficient.
    gamma : float, optional
        Coupling associated with the magnetic sector.
    ha : float, optional
        Quadratic magnetic-sector coefficient.
    hb : float, optional
        Quartic magnetic-sector coefficient.
    eta1 : float, optional
        Coupling between the superconducting and magnetic sectors.
    eta2 : float | None, optional
        Secondary inter-sector coupling.
        If None, it is derived from eta1 and ha.
    M0 : float | None, optional
        Derived magnetic vacuum scale.
        If None, it is computed from the model coefficients when possible.
    u1 : float | None, optional
        Derived superconducting vacuum scale.
        If None, it is computed from the model coefficients when possible.
    vortex_number : float, optional
        Winding number used by the vortex or gauge-field initial condition.
    ainf : float | None, optional
        Asymptotic gauge-field value.
        If None, it is derived from vortex_number and q.
    skyrmion_number : float, optional
        Topological charge used by the initial-condition ansatz.
    skyrmion_rotation : float, optional
        Rotation angle applied in the initial-condition ansatz.
    ansatz : str, optional
        Initial-condition ansatz.
        Supported values are "bloch", "neel", "anti", and "uniform".

    Examples
    --------
    >>> p = Params()
    >>> p = Params(q=1.0, alpha=-1.0, beta=1.0, gamma=1.0, vortex_number=2.0, ansatz="neel")
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
        Convert user-facing parameters into fully resolved ferromagnetic superconductor parameters.

        Returns
        -------
        ResolvedParams
            Resolved parameter object containing both core derived quantities and ferromagnetic superconductor theory-specific derived quantities.

        Examples
        --------
        >>> p = Params(q=1.0, vortex_number=2.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved ferromagnetic superconductor parameters.

    This class contains the full set of core resolved parameters together with ferromagnetic superconductor theory-specific flags and coefficients needed by the CPU and GPU solver code.

    Parameters
    ----------
    number_magnetization_fields : int
        Number of magnetization field components.
    number_higgs_fields : int
        Number of Higgs field components.
    number_gauge_fields : int
        Number of gauge-field components.
    q : float
        Gauge coupling.
    alpha : float
        Quadratic Higgs-sector coefficient.
    beta : float
        Quartic Higgs-sector coefficient.
    gamma : float
        Coupling associated with the magnetic sector.
    skyrmion_number : float
        Topological charge used in the initial ansatz.
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
        Winding number used in the gauge-field sector.
    ainf : float
        Asymptotic gauge-field value.
    skyrmion_rotation : float
        Rotation angle used in the initial ansatz.
    ansatz_bloch : bool
        Whether the Bloch ansatz is enabled.
    ansatz_neel : bool
        Whether the Neel ansatz is enabled.
    ansatz_anti : bool
        Whether the anti-skyrmion ansatz is enabled.
    ansatz_uniform : bool
        Whether the uniform initial configuration is enabled.

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
        Build resolved ferromagnetic superconductor parameters from user-facing parameters.

        This method computes derived inter-sector couplings, vacuum scales, asymptotic gauge values, and ansatz flags required by the rest of the code.

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
    Construct ferromagnetic superconductor parameters using defaults plus user overrides.

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
    >>> p = default_params(q=1.0, vortex_number=2.0, ansatz="neel")
    """
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved ferromagnetic superconductor parameters into device ABI arrays.

    The core integer and floating-point parameter arrays are created first and the ferromagnetic superconductor theory-specific entries are then appended.
    This preserves the ABI indices expected by the ferromagnetic superconductor kernels.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved ferromagnetic superconductor parameters.

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
    Print a readable description of the ferromagnetic superconductor parameter set.

    The printed output is intended for interactive terminal use through theory.describe().
    It summarizes the field content, theory-specific model parameters, derived scales, initial-condition controls, and the meaning of the packed device arrays.

    Returns
    -------
    None
        This function prints parameter information to the terminal.

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