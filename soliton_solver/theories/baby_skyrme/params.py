# =========================
# soliton_solver/theories/baby_skyrme/params.py
# =========================
"""
Baby skyrmion theory-specific parameters.

This module extends the core Params/ResolvedParams and appends entries to p_i/p_f so that
the existing CUDA ABI indices are unchanged.

Core prefix (from soliton_solver.core.params.pack_device_params):
- p_i[0..9], p_f[0..5]

BabySkyrme appends:
- p_i[10] number_magnetization_fields
- p_i[11..18] potential flags (0/1)
- p_i[19] N (broken potential exponent)

- p_f[6] mpi
- p_f[7] kappa
- p_f[8] skyrmion_number
- p_f[9] skyrmion_rotation
- p_f[10..13] ansatz flags (bloch/neel/anti/uniform) as 0/1
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from soliton_solver.core.params import Params as CoreParams, ResolvedParams as CoreResolvedParams, pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    BabySkyrme user-facing params: core + theory specifics.
    """
    # Override default for this theory
    number_total_fields: int = 3

    # Model params (theory-specific)
    number_magnetization_fields: int = 3
    mpi: float = 1.0
    kappa: float = 1.0
    N: int = 1

    # Potential terms (raw flags, still supported)
    Potential_Standard: bool = True
    Potential_Holomorphic: bool = False
    Potential_EasyPlane: bool = False
    Potential_Dihedral2: bool = False
    Potential_Aloof: bool = False
    Potential_Dihedral3: bool = False
    Potential_Broken: bool = False
    Potential_DoubleVacua: bool = False

    # New convenience selector (theory-only):
    # - None: keep explicit Potential_* booleans as provided
    # - "Standard": enables only that potential
    # - {"Standard","Easy plane"}: enables multiple
    potential: str | Iterable[str] | None = None

    # Initial condition controls (theory-specific)
    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"  # "bloch" | "neel" | "anti" | "uniform"

    def resolved(self) -> "ResolvedParams":
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    BabySkyrme resolved params: core derived + theory derived.
    """
    # ints (appended to p_i)
    number_magnetization_fields: int
    N: int

    # floats (appended to p_f)
    mpi: float
    kappa: float
    skyrmion_number: float

    # initial config (stored in p_f as flags/controls)
    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    # potential terms
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
        core = CoreResolvedParams.from_params(p)

        ans = (p.ansatz or "bloch").lower()
        ansatz_bloch = ans == "bloch"
        ansatz_neel = ans == "neel"
        ansatz_anti = ans == "anti"
        ansatz_uniform = ans == "uniform"

        # Start from explicit booleans
        pot_standard = bool(p.Potential_Standard)
        pot_holomorphic = bool(p.Potential_Holomorphic)
        pot_easyplane = bool(p.Potential_EasyPlane)
        pot_dihedral2 = bool(p.Potential_Dihedral2)
        pot_aloof = bool(p.Potential_Aloof)
        pot_dihedral3 = bool(p.Potential_Dihedral3)
        pot_broken = bool(p.Potential_Broken)
        pot_doublevacua = bool(p.Potential_DoubleVacua)

        # If p.potential is provided, it overrides the individual Potential_* booleans.
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
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack (p_i, p_f) for BabySkyrme by:
      1) building the core prefix arrays
      2) appending theory-specific entries

    This preserves the original ABI indices expected by BabySkyrme kernels.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        rp.number_magnetization_fields,         # 10
        1 if rp.Potential_Standard else 0,      # 11
        1 if rp.Potential_Holomorphic else 0,   # 12
        1 if rp.Potential_EasyPlane else 0,     # 13
        1 if rp.Potential_Dihedral2 else 0,     # 14
        1 if rp.Potential_Aloof else 0,         # 15
        1 if rp.Potential_Dihedral3 else 0,     # 16
        1 if rp.Potential_Broken else 0,        # 17
        1 if rp.Potential_DoubleVacua else 0,   # 18
        int(rp.N),                              # 19
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.mpi,                                 # 6
        rp.kappa,                               # 7
        rp.skyrmion_number,                     # 8
        rp.skyrmion_rotation,                   # 9
        1.0 if rp.ansatz_bloch else 0.0,        # 10
        1.0 if rp.ansatz_neel else 0.0,         # 11
        1.0 if rp.ansatz_anti else 0.0,         # 12
        1.0 if rp.ansatz_uniform else 0.0,      # 13
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f