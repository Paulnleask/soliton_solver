"""
Parameters for the anisotropic s+id superconductor theory.

Examples
--------
>>> p = default_params(Q=1.0, N1=1.0, N2=2.0, vortex_type=0)
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
    User-facing parameters for the anisotropic s+id superconductor theory.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields stored by the solver.
    number_magnetization_fields : int, optional
        Number of magnetization field components.
    number_higgs_fields : int, optional
        Number of superconducting real field components.
    number_gauge_fields : int, optional
        Number of gauge field components.
    vortex_type : int, optional
        Initial vortex ansatz type.
    N1 : float, optional
        Winding number of the first condensate.
    N2 : float, optional
        Winding number of the second condensate.
    alpha1 : float, optional
        Quadratic coefficient for the first condensate.
    alpha2 : float, optional
        Quadratic coefficient for the second condensate.
    beta1 : float, optional
        Quartic self-coupling of the first condensate.
    beta2 : float, optional
        Quartic self-coupling of the second condensate.
    beta3 : float, optional
        Density-density coupling.
    beta4 : float or None, optional
        Phase-sensitive quartic coupling.
    Q : float, optional
        Gauge coupling.
    kappa : float, optional
        Magnetic coupling.
    gamma1 : float, optional
        Anisotropy coefficient.
    gamma2 : float, optional
        Anisotropy coefficient.
    gamma12 : float, optional
        Mixed anisotropy coefficient.
    theta_diff : float, optional
        Relative condensate phase in the vacuum.
    beta3tilde : float or None, optional
        Effective mixed quartic coupling.
    u1 : float or None, optional
        Vacuum amplitude of the first condensate.
    u2 : float or None, optional
        Vacuum amplitude of the second condensate.
    vortex_number : float or None, optional
        Effective total vortex number.
    ainf : float or None, optional
        Asymptotic gauge field value.

    Examples
    --------
    >>> p = Params(Q=1.0, N1=1.0, N2=2.0, vortex_type=0)
    >>> rp = p.resolved()
    """
    number_total_fields: int = 6
    number_magnetization_fields: int = 0
    number_higgs_fields: int = 4
    number_gauge_fields: int = 2

    vortex_type: int = 0
    N1: float = 1.0
    N2: float = 1.0

    alpha1: float = -1.0
    alpha2: float = -1.24503
    beta1: float = 1.0
    beta2: float = 1.4959
    beta3: float = 4.02498
    beta4: float | None = None

    Q: float = 1.0
    kappa: float = 0.673274

    gamma1: float = 1.0
    gamma2: float = 0.877868
    gamma12: float = 0.534484

    theta_diff: float = math.pi / 2.0
    beta3tilde: float | None = None

    u1: float | None = None
    u2: float | None = None
    vortex_number: float | None = None
    ainf: float | None = None

    def resolved(self) -> "ResolvedParams":
        """
        Resolve the theory parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved parameter set.

        Examples
        --------
        >>> p = Params(Q=1.0, N1=1.0, N2=2.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved parameters for the anisotropic s+id superconductor theory.

    Examples
    --------
    >>> p = Params(Q=1.0, N1=1.0, N2=2.0)
    >>> rp = ResolvedParams.from_params(p)
    """
    number_magnetization_fields: int
    number_higgs_fields: int
    number_gauge_fields: int

    vortex_type: int
    N1: float
    N2: float

    alpha1: float
    alpha2: float
    beta1: float
    beta2: float
    beta3: float
    beta4: float

    Q: float
    kappa: float

    gamma1: float
    gamma2: float
    gamma12: float

    theta_diff: float
    beta3tilde: float

    u1: float
    u2: float
    vortex_number: float
    ainf: float

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
        >>> p = Params(Q=1.0, N1=1.0, N2=2.0, vortex_type=0)
        >>> rp = ResolvedParams.from_params(p)
        """
        core = CoreResolvedParams.from_params(p)

        beta4 = float(p.beta4) if p.beta4 is not None else float(p.beta3) / 4.0
        if beta4 >= 0.0:
            theta_diff =  math.pi / 2.0
        else:
            theta_diff = 0.0
        beta3tilde = float(p.beta3tilde) if p.beta3tilde is not None else float(p.beta3 + 2.0 * math.cos(2.0 * theta_diff) * beta4)

        if p.u1 is None or p.u2 is None:
            denom = beta3tilde * beta3tilde - 4.0 * p.beta1 * p.beta2
            num1 = 2.0 * p.beta2 * p.alpha1 - beta3tilde * p.alpha2
            num2 = 2.0 * p.beta1 * p.alpha2 - beta3tilde * p.alpha1

            if denom != 0.0 and num1 / denom > 0.0:
                u1 = math.sqrt(num1 / denom)
            else:
                u1 = 0.0
            if denom != 0.0 and num2 / denom > 0.0:
                u2 = math.sqrt(num2 / denom)
            else:
                u2 = 0.0
        else:
            u1 = float(p.u1)
            u2 = float(p.u2)

        if p.vortex_number is None:
            denom_v = u1 * u1 + u2 * u2
            if denom_v != 0.0:
                vortex_number = ((u1 * u1) * p.N1 + (u2 * u2) * p.N2) / denom_v
            else:
                vortex_number = 0.0
        else:
            vortex_number = float(p.vortex_number)

        if p.ainf is None:
            ainf = -vortex_number / p.Q if p.Q != 0.0 else 0.0
        else:
            ainf = float(p.ainf)

        return ResolvedParams(
            xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization,
            xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step,
            number_magnetization_fields=int(p.number_magnetization_fields), number_higgs_fields=int(p.number_higgs_fields), number_gauge_fields=int(p.number_gauge_fields),
            vortex_type=int(p.vortex_type), N1=float(p.N1), N2=float(p.N2),
            alpha1=float(p.alpha1), alpha2=float(p.alpha2), beta1=float(p.beta1), beta2=float(p.beta2), beta3=float(p.beta3), beta4=float(beta4),
            Q=float(p.Q), kappa=float(p.kappa),
            gamma1=float(p.gamma1), gamma2=float(p.gamma2), gamma12=float(p.gamma12),
            theta_diff=float(theta_diff), beta3tilde=float(beta3tilde),
            u1=float(u1), u2=float(u2), vortex_number=float(vortex_number), ainf=float(ainf)
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
    >>> p = default_params(Q=1.0, N1=1.0, N2=2.0, vortex_type=0)
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

    p_i_theory = np.array([rp.number_magnetization_fields, rp.number_higgs_fields, rp.number_gauge_fields, rp.vortex_type], dtype=np.int32)

    p_f_theory = np.array([
        rp.kappa,
        rp.Q,
        rp.alpha1,
        rp.alpha2,
        rp.beta1,
        rp.beta2,
        rp.beta3,
        rp.beta4,
        rp.gamma1,
        rp.gamma2,
        rp.gamma12,
        rp.u1,
        rp.u2,
        rp.theta_diff,
        rp.beta3tilde,
        rp.vortex_number,
        rp.ainf,
        rp.N1,
        rp.N2
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f


def describe() -> None:
    """
    Print parameter information for the theory.

    Examples
    --------
    >>> from solitons.theories.anisotropic_s_id import params
    >>> params.describe()
    """
    print("Field content:")
    print("  The anisotropic s+id model uses 2 gauge fields and 2 complex condensates by default.")
    print("  The default real-field ordering is:")
    print("    A_x, A_y, psi1_r, psi1_i, psi2_r, psi2_i")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Model couplings:")
    print("  alpha1, alpha2 : quadratic coefficients")
    print("  beta1, beta2 : quartic self-couplings")
    print("  beta3 : density-density coupling")
    print("  beta4 : phase-sensitive quartic coupling")
    print("  Q : gauge coupling")
    print("  kappa : magnetic coupling")
    print("  gamma1, gamma2, gamma12 : anisotropy coefficients")
    print()

    print("Derived vacuum parameters:")
    print("  theta_diff : relative vacuum phase")
    print("  beta3tilde = beta3 + 2 cos(2 theta_diff) beta4")
    print("  u1, u2 : vacuum amplitudes")
    print("  If u1 or u2 are left as None, they are derived from the C++ formulas when possible.")
    print()

    print("Vortex controls:")
    print("  vortex_type : initial vortex ansatz type")
    print("  N1, N2 : condensate winding numbers")
    print("  vortex_number : effective total vortex number")
    print("  ainf : asymptotic gauge field value")
    print("  If vortex_number is left as None, it is derived from u1, u2, N1, and N2.")
    print("  If ainf is left as None, it is derived as vortex_number / Q.")
    print()