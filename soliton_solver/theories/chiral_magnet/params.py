"""
Chiral magnet theory-specific parameters, parameter resolution, device packing,
and terminal parameter documentation.

This module extends the core Params and ResolvedParams classes with the
theory-specific parameters required by the chiral magnet model. It preserves
the existing CUDA ABI layout by appending theory-specific entries to the core
integer and floating-point device parameter arrays.

The module also provides a describe() function so that the chiral magnet theory
can print readable parameter information through theory.describe().

Core prefix
-----------
From soliton_solver.core.params.pack_device_params:
- p_i[0..9]
- p_f[0..5]

Chiral magnet appended entries
------------------------------
- p_i[10]    number_magnetization_fields
- p_i[11]    dmi_dresselhaus
- p_i[12]    dmi_rashba
- p_i[13]    dmi_heusler
- p_i[14]    dmi_hybrid
- p_i[15]    demag

- p_f[6]     coup_K
- p_f[7]     coup_h
- p_f[8]     coup_mu
- p_f[9]     skyrmion_number
- p_f[10]    skyrmion_rotation
- p_f[11]    ansatz_bloch
- p_f[12]    ansatz_neel
- p_f[13]    ansatz_anti
- p_f[14]    ansatz_uniform

Examples
--------
>>> from soliton_solver.theories.chiral_magnet.params import Params, default_params
>>> p = default_params(dmi_term="Dresselhaus", ansatz="bloch")
>>> rp = p.resolved()
>>> p_i, p_f = pack_device_params(rp)
>>> describe()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from soliton_solver.core.params import Params as CoreParams
from soliton_solver.core.params import ResolvedParams as CoreResolvedParams
from soliton_solver.core.params import pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    User-facing chiral magnet parameters.

    This class extends the core solver parameters with chiral magnet material
    constants, Dzyaloshinskii-Moriya interaction controls, demagnetization
    settings, and initial-condition controls.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields stored by the solver. For the chiral magnet
        model this defaults to 4.
    number_magnetization_fields : int, optional
        Number of magnetization field components. This defaults to 3.
    J : float, optional
        Exchange stiffness.
    K : float, optional
        Magnetic anisotropy constant.
    D : float, optional
        Dzyaloshinskii-Moriya interaction strength.
    M : float, optional
        Saturation magnetization.
    B : float, optional
        Applied magnetic field magnitude.
    mu0 : float, optional
        Vacuum permeability.
    coup_K : float | None, optional
        Dimensionless anisotropy coupling. If None, it is derived from J, K,
        and D when possible.
    coup_h : float | None, optional
        Dimensionless Zeeman-field coupling. If None, it is derived from J, M,
        B, and D when possible.
    coup_mu : float | None, optional
        Dimensionless demagnetization coupling. If None, it is derived from J,
        mu0, M, and D when possible.
    dmi_dresselhaus : bool, optional
        Enable the Dresselhaus or bulk DMI term.
    dmi_rashba : bool, optional
        Enable the Rashba or interfacial DMI term.
    dmi_heusler : bool, optional
        Enable the Heusler or B20 DMI term.
    dmi_hybrid : bool, optional
        Enable the hybrid DMI term.
    demag : bool, optional
        Enable the demagnetization term.
    dmi_term : str | Iterable[str] | None, optional
        Convenience selector for DMI choice. If provided, it overrides the
        individual dmi_* booleans. It may be a single string, an iterable of
        strings, or None.
    skyrmion_number : float, optional
        Topological charge used by the initial-condition ansatz.
    skyrmion_rotation : float, optional
        Rotation angle applied in the initial-condition ansatz.
    ansatz : str, optional
        Initial-condition ansatz. Supported values are "bloch", "neel", "anti",
        and "uniform".

    Examples
    --------
    >>> p = Params()
    >>> p = Params(D=3e-3, B=50e-3, dmi_term="Rashba", ansatz="neel")
    >>> rp = p.resolved()
    """

    number_total_fields: int = 4

    number_magnetization_fields: int = 3
    J: float = 40e-12
    K: float = 0.8e+6
    D: float = 4e-3
    M: float = 580e+3
    B: float = 0e-3
    mu0: float = 1.25663706127e-6
    coup_K: float | None = None
    coup_h: float | None = None
    coup_mu: float | None = None

    dmi_dresselhaus: bool = True
    dmi_rashba: bool = False
    dmi_heusler: bool = False
    dmi_hybrid: bool = False

    demag: bool = False

    dmi_term: str | Iterable[str] | None = None

    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"

    def resolved(self) -> "ResolvedParams":
        """
        Convert user-facing parameters into fully resolved chiral magnet parameters.

        Returns
        -------
        ResolvedParams
            Resolved parameter object containing both core derived quantities and
            chiral magnet theory-specific derived quantities.

        Examples
        --------
        >>> p = Params(dmi_term="Dresselhaus", demag=True)
        >>> rp = p.resolved()
        """
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved chiral magnet parameters.

    This class contains the full set of core resolved parameters together with
    chiral magnet theory-specific flags and coefficients needed by the CPU and
    GPU solver code.

    Parameters
    ----------
    number_magnetization_fields : int
        Number of magnetization field components.
    J : float
        Exchange stiffness.
    K : float
        Magnetic anisotropy constant.
    D : float
        Dzyaloshinskii-Moriya interaction strength.
    M : float
        Saturation magnetization.
    B : float
        Applied magnetic field magnitude.
    mu0 : float
        Vacuum permeability.
    coup_K : float
        Dimensionless anisotropy coupling.
    coup_h : float
        Dimensionless Zeeman-field coupling.
    coup_mu : float
        Dimensionless demagnetization coupling.
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
        Whether the uniform initial configuration is enabled.
    dmi_dresselhaus : bool
        Whether the Dresselhaus or bulk DMI term is enabled.
    dmi_rashba : bool
        Whether the Rashba or interfacial DMI term is enabled.
    dmi_heusler : bool
        Whether the Heusler or B20 DMI term is enabled.
    dmi_hybrid : bool
        Whether the hybrid DMI term is enabled.
    demag : bool
        Whether the demagnetization term is enabled.

    Examples
    --------
    >>> p = Params(dmi_term="Hybrid", demag=True)
    >>> rp = ResolvedParams.from_params(p)
    """

    number_magnetization_fields: int

    J: float
    K: float
    D: float
    M: float
    B: float
    mu0: float
    coup_K: float
    coup_h: float
    coup_mu: float
    skyrmion_number: float

    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    dmi_dresselhaus: bool
    dmi_rashba: bool
    dmi_heusler: bool
    dmi_hybrid: bool
    demag: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved chiral magnet parameters from user-facing parameters.

        This method resolves ansatz strings into explicit boolean flags,
        computes dimensionless couplings when they are not provided, and
        converts the convenience DMI selector into the explicit dmi_* boolean
        fields expected by the rest of the code.

        Parameters
        ----------
        p : Params
            User-facing chiral magnet parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved chiral magnet parameters.

        Raises
        ------
        ValueError
            If an unknown DMI name is supplied, or if the DMI selector is
            provided but resolves to no active DMI term.

        Examples
        --------
        >>> p = Params(dmi_term=["Dresselhaus", "Hybrid"], demag=True)
        >>> rp = ResolvedParams.from_params(p)
        """
        core = CoreResolvedParams.from_params(p)

        ans = (p.ansatz or "bloch").lower()
        ansatz_bloch = ans == "bloch"
        ansatz_neel = ans == "neel"
        ansatz_anti = ans == "anti"
        ansatz_uniform = ans == "uniform"

        denom = (p.D * p.D)

        if p.coup_K is None or p.coup_h is None or p.coup_mu is None:
            if denom != 0.0:
                coup_K = (p.K * p.J / denom)
                coup_h = (p.M * p.B * p.J / denom)
                coup_mu = (p.mu0 * p.M * p.M * p.J / denom)
            else:
                coup_K = 1.0
                coup_h = 1.0
                coup_mu = 1.0
        else:
            coup_K = float(p.coup_K)
            coup_h = float(p.coup_h)
            coup_mu = float(p.coup_mu)

        dresselhaus = bool(p.dmi_dresselhaus)
        rashba = bool(p.dmi_rashba)
        heusler = bool(p.dmi_heusler)
        hybrid = bool(p.dmi_hybrid)

        if p.dmi_term is not None:
            dresselhaus = False
            rashba = False
            heusler = False
            hybrid = False

            def _norm(s: str) -> str:
                """
                Normalize a DMI name for case-insensitive matching.

                Parameters
                ----------
                s : str
                    Raw DMI name.

                Returns
                -------
                str
                    Normalized DMI name.
                """
                return (s or "").strip().lower().replace("_", " ").replace("-", " ")

            if isinstance(p.dmi_term, str):
                items = [p.dmi_term]
            else:
                items = list(p.dmi_term)

            for item in items:
                key = _norm(str(item))

                if key in ("dresselhaus", "bulk"):
                    dresselhaus = True
                elif key in ("rashba", "interfacial"):
                    rashba = True
                elif key in ("heusler", "b20"):
                    heusler = True
                elif key in ("hybrid",):
                    hybrid = True
                else:
                    raise ValueError("Unknown DMI '{0}'. Valid: Dresselhaus (bulk), Rashba (interfacial), Heusler (B20), Hybrid".format(item))

            if not (dresselhaus or rashba or heusler or hybrid):
                raise ValueError("p.dmi_term was provided but no valid DMIs were selected.")

        return ResolvedParams(
            xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization,
            xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step,
            number_magnetization_fields=int(p.number_magnetization_fields),
            J=float(p.J), D=float(p.D), K=float(p.K), M=float(p.M), B=float(p.B), mu0=float(p.mu0),
            coup_K=float(coup_K), coup_h=float(coup_h), coup_mu=float(coup_mu),
            skyrmion_number=float(p.skyrmion_number),
            skyrmion_rotation=float(p.skyrmion_rotation),
            ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform,
            dmi_dresselhaus=dresselhaus, dmi_rashba=rashba, dmi_heusler=heusler, dmi_hybrid=hybrid,
            demag=p.demag,
        )


def default_params(**overrides) -> Params:
    """
    Construct chiral magnet parameters using defaults plus user overrides.

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
    >>> p = default_params(B=50e-3, dmi_term="Rashba", demag=True)
    """
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved chiral magnet parameters into device ABI arrays.

    The core integer and floating-point parameter arrays are created first and
    the chiral magnet theory-specific entries are then appended. This preserves
    the ABI indices expected by the chiral magnet kernels.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved chiral magnet parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple (p_i, p_f) containing the integer and floating-point device
        parameter arrays.

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
        1 if rp.dmi_heusler else 0,
        1 if rp.dmi_hybrid else 0,
        1 if rp.demag else 0,
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.coup_K,
        rp.coup_h,
        rp.coup_mu,
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
    Print a readable description of the chiral magnet parameter set.

    The printed output is intended for interactive terminal use through
    theory.describe(). It summarizes the field content, theory-specific model
    parameters, DMI selection controls, demagnetization settings,
    initial-condition controls, and the meaning of the packed device arrays.

    Returns
    -------
    None
        This function prints parameter information to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories.chiral_magnet import params
    >>> params.describe()
    """
    print("Field content:")
    print("  The chiral magnet model uses 3 magnetization components and 1 magnetostatic potential by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Material and model parameters:")
    print("  J : exchange stiffness")
    print("  K : anisotropy constant")
    print("  D : Dzyaloshinskii-Moriya interaction strength")
    print("  M : saturation magnetization")
    print("  B : applied magnetic field magnitude")
    print("  mu0 : vacuum permeability")
    print()

    print("Derived couplings:")
    print("  coup_K : dimensionless anisotropy coupling")
    print("  coup_h : dimensionless Zeeman-field coupling")
    print("  coup_mu : dimensionless demagnetization coupling")
    print("  If coup_K, coup_h, or coup_mu are left as None, they are derived from")
    print("  J, K, D, M, B, and mu0 when D is non-zero.")
    print("  If D is zero and the couplings are not supplied explicitly, they default to 1.0.")
    print()

    print("DMI selection:")
    print("  dmi_dresselhaus : enable the Dresselhaus or bulk DMI term")
    print("  dmi_rashba : enable the Rashba or interfacial DMI term")
    print("  dmi_heusler : enable the Heusler or B20 DMI term")
    print("  dmi_hybrid : enable the hybrid DMI term")
    print("  dmi_term : convenience selector that overrides the individual dmi_* flags")
    print("  Valid convenience names:")
    print("    Dresselhaus")
    print("    Bulk")
    print("    Rashba")
    print("    Interfacial")
    print("    Heusler")
    print("    B20")
    print("    Hybrid")
    print("  The convenience selector may be a single string or an iterable of strings.")
    print()

    print("Demagnetization:")
    print("  demag : enables or disables the demagnetization term")
    print()

    print("Initial condition controls:")
    print("  skyrmion_number : topological charge used by the initial-condition ansatz")
    print("  skyrmion_rotation : in-plane rotation angle used in the ansatz")
    print("  ansatz : initial-condition type")
    print("  Supported ansatz values:")
    print("    bloch")
    print("    neel")
    print("    anti")
    print("    uniform")
    print()