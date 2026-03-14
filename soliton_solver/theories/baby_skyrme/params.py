"""
Baby Skyrme theory specific parameters, parameter resolution, device packing, and terminal parameter documentation.

Examples
--------
Use ``default_params`` to create a Baby Skyrme parameter set with optional overrides.
Use ``Params.resolved()`` to derive the fully resolved theory parameters.
Use ``pack_device_params`` to build the device parameter arrays for the Baby Skyrme kernels.
Use ``describe()`` to print the Baby Skyrme parameter documentation.
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
    User facing Baby Skyrme parameters.

    Parameters
    ----------
    number_total_fields : int, optional
        Total number of fields in the theory.
    number_magnetization_fields : int, optional
        Number of magnetization field components.
    mpi : float, optional
        Coefficient controlling the potential strength.
    kappa : float, optional
        Coefficient controlling the Skyrme term strength.
    N : int, optional
        Integer exponent used by the broken potential.
    Potential_Standard : bool, optional
        Enable the standard potential.
    Potential_Holomorphic : bool, optional
        Enable the holomorphic potential.
    Potential_EasyPlane : bool, optional
        Enable the easy plane potential.
    Potential_Dihedral2 : bool, optional
        Enable the dihedral-2 potential.
    Potential_Aloof : bool, optional
        Enable the aloof potential.
    Potential_Dihedral3 : bool, optional
        Enable the dihedral-3 potential.
    Potential_Broken : bool, optional
        Enable the broken potential.
    Potential_DoubleVacua : bool, optional
        Enable the double-vacua potential.
    potential : str or Iterable[str] or None, optional
        Convenience selector for the potential choice.
    skyrmion_number : float, optional
        Topological charge used by the initial ansatz.
    skyrmion_rotation : float, optional
        Rotation angle used by the initial ansatz.
    ansatz : str, optional
        Initial ansatz type.

    Returns
    -------
    None
        The dataclass stores the Baby Skyrme parameter set.

    Examples
    --------
    Use ``p = Params()`` to create the default Baby Skyrme parameter set.
    Use ``p = Params(mpi=2.0, kappa=0.5, potential="Easy plane", ansatz="neel")`` to set model parameters and the initial ansatz.
    Use ``rp = p.resolved()`` to derive the resolved parameter set.
    """

    number_total_fields: int = 3

    number_magnetization_fields: int = 3
    mpi: float = 1.0
    kappa: float = 1.0
    N: int = 1

    Potential_Standard: bool = True
    Potential_Holomorphic: bool = False
    Potential_EasyPlane: bool = False
    Potential_Dihedral2: bool = False
    Potential_Aloof: bool = False
    Potential_Dihedral3: bool = False
    Potential_Broken: bool = False
    Potential_DoubleVacua: bool = False

    potential: str | Iterable[str] | None = None

    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"

    def resolved(self) -> "ResolvedParams":
        """
        Convert user facing parameters into resolved Baby Skyrme parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved Baby Skyrme parameters.

        Examples
        --------
        Use ``rp = p.resolved()`` to derive the resolved Baby Skyrme parameters.
        """
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Fully resolved Baby Skyrme parameters.

    Parameters
    ----------
    number_magnetization_fields : int
        Number of magnetization field components.
    N : int
        Integer exponent for the broken potential.
    mpi : float
        Potential strength coefficient.
    kappa : float
        Skyrme term coefficient.
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
    Potential_Standard : bool
        Whether the standard potential is enabled.
    Potential_Holomorphic : bool
        Whether the holomorphic potential is enabled.
    Potential_EasyPlane : bool
        Whether the easy plane potential is enabled.
    Potential_Dihedral2 : bool
        Whether the dihedral-2 potential is enabled.
    Potential_Aloof : bool
        Whether the aloof potential is enabled.
    Potential_Dihedral3 : bool
        Whether the dihedral-3 potential is enabled.
    Potential_Broken : bool
        Whether the broken potential is enabled.
    Potential_DoubleVacua : bool
        Whether the double-vacua potential is enabled.

    Returns
    -------
    None
        The dataclass stores the resolved Baby Skyrme parameter set.

    Examples
    --------
    Use ``rp = ResolvedParams.from_params(p)`` to build the resolved Baby Skyrme parameters from ``p``.
    """

    number_magnetization_fields: int
    N: int

    mpi: float
    kappa: float
    skyrmion_number: float

    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    Potential_Standard: bool
    Potential_Holomorphic: bool
    Potential_EasyPlane: bool
    Potential_Dihedral2: bool
    Potential_Aloof: bool
    Potential_Dihedral3: bool
    Potential_Broken: bool
    Potential_DoubleVacua: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Build resolved Baby Skyrme parameters from user facing parameters.

        Parameters
        ----------
        p : Params
            User facing Baby Skyrme parameters.

        Returns
        -------
        ResolvedParams
            Fully resolved Baby Skyrme parameters.

        Raises
        ------
        ValueError
            Raised if an unknown potential name is supplied.
        ValueError
            Raised if ``p.potential`` is provided but no valid potential is selected.

        Examples
        --------
        Use ``rp = ResolvedParams.from_params(p)`` to resolve the Baby Skyrme parameters.
        """
        core = CoreResolvedParams.from_params(p)

        ans = (p.ansatz or "bloch").lower()
        ansatz_bloch = ans == "bloch"
        ansatz_neel = ans == "neel"
        ansatz_anti = ans == "anti"
        ansatz_uniform = ans == "uniform"

        pot_standard = bool(p.Potential_Standard)
        pot_holomorphic = bool(p.Potential_Holomorphic)
        pot_easyplane = bool(p.Potential_EasyPlane)
        pot_dihedral2 = bool(p.Potential_Dihedral2)
        pot_aloof = bool(p.Potential_Aloof)
        pot_dihedral3 = bool(p.Potential_Dihedral3)
        pot_broken = bool(p.Potential_Broken)
        pot_doublevacua = bool(p.Potential_DoubleVacua)

        if p.potential is not None:
            pot_standard = False
            pot_holomorphic = False
            pot_easyplane = False
            pot_dihedral2 = False
            pot_aloof = False
            pot_dihedral3 = False
            pot_broken = False
            pot_doublevacua = False

            def _norm(s: str) -> str:
                """
                Normalize a potential name for case insensitive matching.

                Parameters
                ----------
                s : str
                    Raw potential name.

                Returns
                -------
                str
                    Normalized potential name.

                Examples
                --------
                Use ``key = _norm(name)`` before matching a potential name.
                """
                return (s or "").strip().lower().replace("_", " ").replace("-", " ")

            if isinstance(p.potential, str):
                items = [p.potential]
            else:
                items = list(p.potential)

            for item in items:
                key = _norm(str(item))

                if key in ("standard",):
                    pot_standard = True
                elif key in ("holomorphic", "holomorhpic"):
                    pot_holomorphic = True
                elif key in ("easy plane", "easyplane", "easy-plane"):
                    pot_easyplane = True
                elif key in ("dihedral2", "dihedral 2", "dihedral-2"):
                    pot_dihedral2 = True
                elif key in ("aloof",):
                    pot_aloof = True
                elif key in ("dihedral3", "dihedral 3", "dihedral-3"):
                    pot_dihedral3 = True
                elif key in ("broken",):
                    pot_broken = True
                elif key in ("double vacua", "doublevacua", "double-vacua", "2 vacua", "two vacua"):
                    pot_doublevacua = True
                else:
                    raise ValueError("Unknown potential '{0}'. Valid: Standard, Holomorphic, Easy plane, Dihedral2, Aloof, Dihedral3, Broken, DoubleVacua".format(item))

            if not (pot_standard or pot_holomorphic or pot_easyplane or pot_dihedral2 or pot_aloof or pot_dihedral3 or pot_broken or pot_doublevacua):
                raise ValueError("p.potential was provided but no valid potentials were selected.")

        return ResolvedParams(
            xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization,
            xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step,
            number_magnetization_fields=int(p.number_magnetization_fields),
            mpi=float(p.mpi), kappa=float(p.kappa),
            skyrmion_number=float(p.skyrmion_number),
            skyrmion_rotation=float(p.skyrmion_rotation),
            ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform,
            Potential_Standard=pot_standard, Potential_Holomorphic=pot_holomorphic, Potential_EasyPlane=pot_easyplane, Potential_Dihedral2=pot_dihedral2,
            Potential_Aloof=pot_aloof, Potential_Dihedral3=pot_dihedral3, Potential_Broken=pot_broken, Potential_DoubleVacua=pot_doublevacua,
            N=int(p.N),
        )


def default_params(**overrides) -> Params:
    """
    Construct Baby Skyrme parameters using defaults plus user overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments forwarded to ``Params.with_()``.

    Returns
    -------
    Params
        Parameter object with the requested overrides applied.

    Examples
    --------
    Use ``p = default_params(mpi=2.0, ansatz="neel")`` to build a customized Baby Skyrme parameter set.
    """
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack resolved Baby Skyrme parameters into device ABI arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Fully resolved Baby Skyrme parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple ``(p_i, p_f)`` containing the integer and floating point device parameter arrays.

    Examples
    --------
    Use ``p_i, p_f = pack_device_params(rp)`` to build the Baby Skyrme device parameter arrays.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        rp.number_magnetization_fields,
        1 if rp.Potential_Standard else 0,
        1 if rp.Potential_Holomorphic else 0,
        1 if rp.Potential_EasyPlane else 0,
        1 if rp.Potential_Dihedral2 else 0,
        1 if rp.Potential_Aloof else 0,
        1 if rp.Potential_Dihedral3 else 0,
        1 if rp.Potential_Broken else 0,
        1 if rp.Potential_DoubleVacua else 0,
        int(rp.N),
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.mpi,
        rp.kappa,
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
    Print a readable description of the Baby Skyrme parameter set.

    Returns
    -------
    None
        The parameter information is printed to the terminal.

    Examples
    --------
    Use ``describe()`` to print the Baby Skyrme parameter documentation.
    """
    print("Field content:")
    print("  The Baby Skyrme model uses 3 field components by default.")
    print()

    print("Grid content:")
    print("  xlen : number of grid points in x-direction")
    print("  ylen : number of grid points in y-direction")
    print("  xsize : dimensionless grid size in x-direction")
    print("  ysize : dimensionless grid size in y-direction")
    print()

    print("Model couplings:")
    print("  mpi : coefficient controlling the potential strength")
    print("  kappa : coefficient controlling the Skyrme term strength")
    print("  N : integer exponent used by the broken potential")
    print()

    print("Potential selection:")
    print("  Potential_Standard : enable the standard potential")
    print("  Potential_Holomorphic : enable the holomorphic potential")
    print("  Potential_EasyPlane : enable the easy-plane potential")
    print("  Potential_Dihedral2 : enable the dihedral-2 potential")
    print("  Potential_Aloof : enable the aloof potential")
    print("  Potential_Dihedral3 : enable the dihedral-3 potential")
    print("  Potential_Broken : enable the broken potential")
    print("  Potential_DoubleVacua : enable the double-vacua potential")
    print("  potential : convenience selector that overrides the individual Potential_* flags")
    print("  Valid convenience names:")
    print("    Standard")
    print("    Holomorphic")
    print("    Easy plane")
    print("    Dihedral2")
    print("    Aloof")
    print("    Dihedral3")
    print("    Broken")
    print("    DoubleVacua")
    print("  The convenience selector may be a single string or an iterable of strings.")
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