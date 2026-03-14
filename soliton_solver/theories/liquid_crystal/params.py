"""
Liquid crystal theory parameters and device packing.

Examples
--------
>>> p = default_params(deformation="twist", ansatz="bloch")
>>> rp = p.resolved()
>>> p_i, p_f = pack_device_params(rp)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable
import math

import numpy as np

from soliton_solver.core.params import Params as CoreParams
from soliton_solver.core.params import ResolvedParams as CoreResolvedParams
from soliton_solver.core.params import pack_device_params as pack_core_device_params

@dataclass(frozen=True)
class Params(CoreParams):
    """
    User facing liquid crystal parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields stored by the solver.
    number_magnetization_fields : int, optional
        Number of director or magnetization-like field components.
    K : float, optional
        Elastic constant in the one-constant approximation.
    P : float, optional
        Cholesteric pitch.
    d : float, optional
        Sample thickness.
    e1 : float, optional
        Flexoelectric splay coefficient.
    e3 : float, optional
        Flexoelectric bend coefficient.
    w0 : float, optional
        Homeotropic anchoring strength.
    eps0 : float, optional
        Vacuum permittivity.
    voltage : float, optional
        Applied voltage across the sample.
    delta_eps : float, optional
        Dielectric anisotropy.
    E : float | None, optional
        Electric field magnitude.
    q0 : float | None, optional
        Chiral wave number.
    coup_eps : float | None, optional
        Dimensionless dielectric or flexoelectric coupling.
    coup_Pot : float | None, optional
        Dimensionless potential prefactor.
    dmi_dresselhaus : bool, optional
        Enable the twist-favoured deformation sector.
    dmi_rashba : bool, optional
        Enable the splay-bend-favoured deformation sector.
    depol : bool, optional
        Enable flexoelectric depolarization effects.
    deformation : str | Iterable[str] | None, optional
        Convenience selector for the deformation sector.
    skyrmion_number : float, optional
        Topological charge used by the initial ansatz.
    skyrmion_rotation : float, optional
        Rotation angle used by the initial ansatz.
    ansatz : str, optional
        Initial-condition ansatz.

    Examples
    --------
    >>> p = Params(K=10e-12, P=7e-6, voltage=4.0, deformation="twist", ansatz="neel")
    >>> rp = p.resolved()
    """

    number_total_fields: int = 4

    number_magnetization_fields: int = 3
    K: float = 10e-12
    P: float = 7.0e-6
    d: float = 4e-6
    e1: float = 2e-12
    e3: float = 4e-12
    w0: float = 1.0
    eps0: float = 8.854e-12
    voltage: float = 4.0
    delta_eps: float = 3.7

    E: float | None = None
    q0: float | None = None
    coup_eps: float | None = None
    coup_Pot: float | None = None

    dmi_dresselhaus: bool = True
    dmi_rashba: bool = False

    depol: bool = False

    deformation: str | Iterable[str] | None = None

    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"

    def with_(self, **kwargs) -> "Params":
        """
        Return a copy with selected fields replaced.

        Parameters
        ----------
        **kwargs
            Dataclass field values to replace.

        Returns
        -------
        Params
            Updated parameter object.

        Examples
        --------
        >>> p = Params()
        >>> p2 = p.with_(voltage=5.0, ansatz="neel")
        """
        return replace(self, **kwargs)

    def resolved(self) -> "ResolvedParams":
        """
        Resolve the liquid crystal parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved parameter object.

        Examples
        --------
        >>> p = Params(deformation="twist", voltage=4.0)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)

@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved liquid crystal parameters.

    Parameters
    ----------
    number_magnetization_fields : int
        Number of director or magnetization-like field components.
    K : float
        Elastic constant in the one-constant approximation.
    P : float
        Cholesteric pitch.
    d : float
        Sample thickness.
    e1 : float
        Flexoelectric splay coefficient.
    e3 : float
        Flexoelectric bend coefficient.
    w0 : float
        Homeotropic anchoring strength.
    eps0 : float
        Vacuum permittivity.
    voltage : float
        Applied voltage across the sample.
    delta_eps : float
        Dielectric anisotropy.
    E : float
        Electric field magnitude.
    q0 : float
        Chiral wave number.
    coup_eps : float
        Dimensionless dielectric or flexoelectric coupling.
    coup_Pot : float
        Dimensionless potential prefactor.
    coup_PotE : float
        Electric-field contribution to the potential prefactor.
    coup_Potw0 : float
        Anchoring contribution to the potential prefactor.
    skyrmion_number : float
        Topological charge used in the initial ansatz.
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
    dmi_dresselhaus : bool
        Whether the twist-favoured sector is enabled.
    dmi_rashba : bool
        Whether the splay-bend-favoured sector is enabled.
    depol : bool
        Whether depolarization effects are enabled.

    Examples
    --------
    >>> p = Params(deformation="splay-bend", depol=True)
    >>> rp = ResolvedParams.from_params(p)
    """

    number_magnetization_fields: int

    K: float
    P: float
    d: float
    e1: float
    e3: float
    w0: float
    eps0: float
    voltage: float
    delta_eps: float
    E: float
    q0: float
    coup_eps: float
    coup_Pot: float
    coup_PotE: float
    coup_Potw0: float

    skyrmion_number: float
    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    dmi_dresselhaus: bool
    dmi_rashba: bool
    depol: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved parameters from user parameters.

        Parameters
        ----------
        p : Params
            User-facing liquid crystal parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved liquid crystal parameters.

        Raises
        ------
        ValueError
            Raised if the deformation selector contains an unknown name.
        ValueError
            Raised if the deformation selector is provided but enables no deformation sector.

        Examples
        --------
        >>> p = Params(deformation="twist", depol=True, voltage=4.0)
        >>> rp = ResolvedParams.from_params(p)
        """
        core = CoreResolvedParams.from_params(p)

        ans = (p.ansatz or "bloch").lower()
        ansatz_bloch = ans == "bloch"
        ansatz_neel = ans == "neel"
        ansatz_anti = ans == "anti"
        ansatz_uniform = ans == "uniform"

        E = float(p.voltage / p.d) if p.E is None else float(p.E)
        q0 = float(2.0 * math.pi / p.P) if p.q0 is None else float(p.q0)

        if p.coup_eps is None:
            coup_eps = float((p.K * p.eps0) / (p.e1 * p.e1)) if p.e1 != 0.0 else 0.0
        else:
            coup_eps = float(p.coup_eps)

        if p.coup_Pot is None:
            coup_Pot = float(1.0 / (q0 * q0 * p.K)) if (q0 != 0.0 and p.K != 0.0) else 0.0
        else:
            coup_Pot = float(p.coup_Pot)

        coup_PotE = float(coup_Pot * p.eps0 * p.delta_eps * E * E)
        coup_Potw0 = float(coup_Pot * p.w0)

        dresselhaus = bool(p.dmi_dresselhaus)
        rashba = bool(p.dmi_rashba)

        if p.deformation is not None:
            dresselhaus = False
            rashba = False

            def _norm(s: str) -> str:
                """
                Normalize a deformation name for matching.

                Parameters
                ----------
                s : str
                    Raw deformation name.

                Returns
                -------
                str
                    Normalized deformation name.

                Examples
                --------
                >>> _norm("splay-bend")
                'splay bend'
                """
                return (s or "").strip().lower().replace("_", " ").replace("-", " ")

            items = [p.deformation] if isinstance(p.deformation, str) else list(p.deformation)

            for item in items:
                key = _norm(str(item))

                if key in ("dresselhaus", "twist"):
                    dresselhaus = True
                elif key in ("rashba", "splay bend", "splaybend", "splay-bend"):
                    rashba = True
                else:
                    raise ValueError("Unknown deformation '{0}'. Valid: twist, splay-bend".format(item))

            if not (dresselhaus or rashba):
                raise ValueError("p.deformation was provided but no valid DMIs were selected.")

        return ResolvedParams(
            xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization,
            xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step,
            number_magnetization_fields=int(p.number_magnetization_fields),
            K=float(p.K), P=float(p.P), d=float(p.d), e1=float(p.e1), e3=float(p.e3), w0=float(p.w0), eps0=float(p.eps0),
            voltage=float(p.voltage), delta_eps=float(p.delta_eps), E=float(E), q0=float(q0),
            coup_eps=float(coup_eps), coup_Pot=float(coup_Pot), coup_PotE=float(coup_PotE), coup_Potw0=float(coup_Potw0),
            skyrmion_number=float(p.skyrmion_number), skyrmion_rotation=float(p.skyrmion_rotation),
            ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform,
            dmi_dresselhaus=dresselhaus, dmi_rashba=rashba, depol=bool(p.depol),
        )

def default_params(**overrides) -> Params:
    """
    Construct default liquid crystal parameters with overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to ``Params.with_()``.

    Returns
    -------
    Params
        Parameter object with overrides applied.

    Examples
    --------
    >>> p = default_params(voltage=5.0, deformation="twist", depol=True)
    """
    return Params().with_(**overrides)

def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved parameters into device arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved liquid crystal parameters.

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

    p_i_theory = np.array([
        rp.number_magnetization_fields,
        1 if rp.dmi_dresselhaus else 0,
        1 if rp.dmi_rashba else 0,
        1 if rp.depol else 0,
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.coup_PotE,
        rp.coup_Potw0,
        rp.coup_eps,
        rp.e1,
        rp.e3,
        rp.skyrmion_number,
        rp.skyrmion_rotation,
        1.0 if rp.ansatz_bloch else 0.0,
        1.0 if rp.ansatz_neel else 0.0,
        1.0 if rp.ansatz_anti else 0.0,
        1.0 if rp.ansatz_uniform else 0.0,
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f

def describe() -> None:
    """
    Print the liquid crystal parameter description.

    Examples
    --------
    >>> from soliton_solver.theories.liquid_crystal import params
    >>> params.describe()
    """
    print("Field content:")
    print("  The liquid crystal model uses 3 director fields and 1 electrostatic potential by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Material and elastic parameters:")
    print("  K : elastic constant in the one-constant approximation.")
    print("  P : cholesteric pitch.")
    print("  d : sample thickness.")
    print("  e1 : flexoelectric splay coefficient.")
    print("  e3 : flexoelectric bend coefficient.")
    print("  w0 : homeotropic anchoring strength.")
    print("  eps0 : vacuum permittivity.")
    print("  voltage : applied voltage across the sample.")
    print("  delta_eps : dielectric anisotropy.")
    print()

    print("Derived quantities and couplings:")
    print("  E : electric field magnitude.")
    print("  q0 : chiral wave number.")
    print("  coup_eps : dimensionless dielectric or flexoelectric coupling.")
    print("  coup_Pot : dimensionless potential prefactor.")
    print("  coup_PotE : electric-field contribution to the potential prefactor.")
    print("  coup_Potw0 : anchoring contribution to the potential prefactor.")
    print("  If E is left as None, it is derived as voltage / d.")
    print("  If q0 is left as None, it is derived as 2 pi / P.")
    print("  If coup_eps is left as None, it is derived from K, eps0, and e1 when e1 is non-zero.")
    print("  If coup_Pot is left as None, it is derived from q0 and K when both are non-zero.")
    print()

    print("Deformation selection:")
    print("  dmi_dresselhaus : enable the twist-favoured deformation sector.")
    print("  dmi_rashba : enable the splay-bend-favoured deformation sector.")
    print("  deformation : convenience selector that overrides the individual dmi_* flags.")
    print("  Valid convenience names:")
    print("    twist")
    print("    dresselhaus")
    print("    splay-bend")
    print("    rashba")
    print("  The convenience selector may be a single string or an iterable of strings.")
    print()

    print("Depolarization:")
    print("  depol : enables or disables flexoelectric depolarization effects.")
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