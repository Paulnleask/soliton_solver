# =========================
# soliton_solver/theories/chiral_magnet/params.py
# =========================
"""
Chiral magnet theory-specific parameters.

This module extends the core Params/ResolvedParams and appends entries to p_i/p_f so that
the existing CUDA ABI indices are unchanged.

Core prefix (from soliton_solver.core.params.pack_device_params):
- p_i[0..9], p_f[0..5]

chiral_magnet appends:
- p_i[10] number_magnetization_fields
- p_i[11,12,13,14] DMI flags (0/1) [dmi_dresselhaus,dmi_rashba,dmi_heusler,dmi_hybrid]
- p_i[15] demag flag (0/1)

- p_f[6,7,8] coup_k, coup_h, coup_mu
- p_f[9] skyrmion_number
- p_f[10] skyrmion_rotation
- p_f[11..14] ansatz flags (bloch/neel/anti/uniform) as 0/1

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from soliton_solver.core.params import Params as CoreParams, ResolvedParams as CoreResolvedParams, pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    chirlamagnet user-facing params: core + theory specifics.
    """
    # Override default for this theory
    number_total_fields: int = 4

    # Model params (theory-specific)
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

    # DMI terms (raw flags, still supported)
    dmi_dresselhaus: bool = True
    dmi_rashba: bool = False
    dmi_heusler: bool = False
    dmi_hybrid: bool = False

    # Demagnetization flag
    demag: bool = False

    # New convenience selector (theory-only):
    # - None: keep explicit dmi_* booleans as provided
    # - "Dresselhaus": enables only that potential
    # - {"Dresselhaus","Rashba"}: enables multiple
    dmi_term: str | Iterable[str] | None = None

    # Initial condition controls (theory-specific)
    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"  # "bloch" | "neel" | "anti" | "uniform"

    def resolved(self) -> "ResolvedParams":
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    chiral_magnet resolved params: core derived + theory derived.
    """
    # ints (appended to p_i)
    number_magnetization_fields: int

    # floats (appended to p_f)
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

    # initial config (stored in p_f as flags/controls)
    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    # potential terms
    dmi_dresselhaus: bool
    dmi_rashba: bool
    dmi_heusler: bool
    dmi_hybrid: bool
    demag: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
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

        # Start from explicit booleans
        dresselhaus = bool(p.dmi_dresselhaus)
        rashba = bool(p.dmi_rashba)
        heusler = bool(p.dmi_heusler)
        hybrid = bool(p.dmi_hybrid)

        # If p.potential is provided, it overrides the individual Potential_* booleans.
        if p.dmi_term is not None:
            dresselhaus = False
            rashba = False
            heusler = False
            hybrid = False

            def _norm(s: str) -> str:
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
                elif key in ("heusler", "B20"):
                    heusler = True
                elif key in ("hybrid"):
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
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack (p_i, p_f) for chiral_magnet by:
      1) building the core prefix arrays
      2) appending theory-specific entries

    This preserves the original ABI indices expected by chiral_magnet kernels.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        rp.number_magnetization_fields,         # 10
        1 if rp.dmi_dresselhaus else 0,         # 11
        1 if rp.dmi_rashba else 0,              # 12
        1 if rp.dmi_heusler else 0,             # 13
        1 if rp.dmi_hybrid else 0,              # 14
        1 if rp.demag else 0,                   # 15
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.coup_K,                              # 6
        rp.coup_h,                              # 7
        rp.coup_mu,                             # 8
        rp.skyrmion_number,                     # 9
        rp.skyrmion_rotation,                   # 10
        1.0 if rp.ansatz_bloch else 0.0,        # 11
        1.0 if rp.ansatz_neel else 0.0,         # 12
        1.0 if rp.ansatz_anti else 0.0,         # 13
        1.0 if rp.ansatz_uniform else 0.0,      # 14
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f