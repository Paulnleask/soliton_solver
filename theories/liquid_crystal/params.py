# =========================
# soliton_solver/theories/liquid_crystal/params.py
# =========================
"""
Liquid crystal / chiral-magnet theory-specific parameters.

This module extends the core Params/ResolvedParams and appends entries to p_i/p_f so that
the existing CUDA ABI indices are unchanged.

Core prefix (from soliton_solver.core.params.pack_device_params):
- p_i[0..12], p_f[0..23]  (whatever your core currently defines)

liquid_crystal appends (example):
- p_i[10] number_magnetization_fields
- p_i[11,12] DMI flags (0/1) [dmi_dresselhaus,dmi_rashba]
- p_i[13] depol flag (0/1)

- p_f[6..] theory scalars (coup_PotE, coup_Potw0, coup_eps, e1, e3, skyrmion controls, ansatz flags)
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Iterable
import numpy as np
import math

from soliton_solver.core.params import Params as CoreParams, ResolvedParams as CoreResolvedParams, pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    Liquid crystal user-facing params: core + theory specifics.
    """
    # Override default for this theory
    number_total_fields: int = 4

    # Model params (theory-specific)
    number_magnetization_fields: int = 3
    K: float = 10e-12      # Elastic constant (one-constant approximation)
    P: float = 7.0e-6      # Cholesteric pitch
    d: float = 4e-6        # Thickness
    e1: float = 2e-12      # Flexoelectric coefficient splay
    e3: float = 4e-12      # Flexoelectric coefficient bend
    w0: float = 1.0        # Homeotropic anchoring
    eps0: float = 8.854e-12
    voltage: float = 4.0
    delta_eps: float = 3.7

    # Optional derived controls
    E: float | None = None
    q0: float | None = None
    coup_eps: float | None = None
    coup_Pot: float | None = None

    # DMI terms (raw flags, still supported)
    dmi_dresselhaus: bool = True
    dmi_rashba: bool = False

    # Depolarization flag
    depol: bool = False

    # Convenience selector (theory-only)
    deformation: str | Iterable[str] | None = None

    # Initial condition controls (theory-specific)
    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"  # "bloch" | "neel" | "anti" | "uniform"

    def with_(self, **kwargs) -> "Params":
        return replace(self, **kwargs)

    def resolved(self) -> "ResolvedParams":
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Liquid crystal resolved params: core derived + theory derived.
    """
    # ints (appended to p_i)
    number_magnetization_fields: int

    # physical/theory floats
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

    # initial config controls
    skyrmion_number: float
    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    # DMI + depol flags
    dmi_dresselhaus: bool
    dmi_rashba: bool
    depol: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
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

        # Start from explicit booleans
        dresselhaus = bool(p.dmi_dresselhaus)
        rashba = bool(p.dmi_rashba)

        # Optional override selector
        if p.deformation is not None:
            dresselhaus = False
            rashba = False

            def _norm(s: str) -> str:
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
            # core fields
            xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields,
            dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization,
            xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step,
            # theory ints
            number_magnetization_fields=int(p.number_magnetization_fields),
            # theory floats (all declared fields provided)
            K=float(p.K), P=float(p.P), d=float(p.d), e1=float(p.e1), e3=float(p.e3), w0=float(p.w0), eps0=float(p.eps0),
            voltage=float(p.voltage), delta_eps=float(p.delta_eps), E=float(E), q0=float(q0),
            coup_eps=float(coup_eps), coup_Pot=float(coup_Pot), coup_PotE=float(coup_PotE), coup_Potw0=float(coup_Potw0),
            # init config
            skyrmion_number=float(p.skyrmion_number), skyrmion_rotation=float(p.skyrmion_rotation),
            ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform,
            # flags
            dmi_dresselhaus=dresselhaus, dmi_rashba=rashba, depol=bool(p.depol),
        )


def default_params(**overrides) -> Params:
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack (p_i, p_f) for liquid_crystal by:
      1) building the core prefix arrays
      2) appending theory-specific entries

    Preserves ABI indices expected by the liquid_crystal kernels.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([
        rp.number_magnetization_fields,         # 10
        1 if rp.dmi_dresselhaus else 0,         # 11
        1 if rp.dmi_rashba else 0,              # 12
        1 if rp.depol else 0,                   # 13
    ], dtype=np.int32)

    p_f_theory = np.array([
        rp.coup_PotE,                           # 6
        rp.coup_Potw0,                          # 7
        rp.coup_eps,                            # 8
        rp.e1,                                  # 9
        rp.e3,                                  # 10
        rp.skyrmion_number,                     # 11
        rp.skyrmion_rotation,                   # 12
        1.0 if rp.ansatz_bloch else 0.0,        # 13
        1.0 if rp.ansatz_neel else 0.0,         # 14
        1.0 if rp.ansatz_anti else 0.0,         # 15
        1.0 if rp.ansatz_uniform else 0.0,      # 16
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f