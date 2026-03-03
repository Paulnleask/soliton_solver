# =========================
# soliton_solver/theories/ferromagnetic_superconductor/params.py
# =========================
"""
Superferro theory-specific parameters.

This module extends the core Params/ResolvedParams and appends entries to p_i/p_f so that
the existing CUDA ABI indices are unchanged.

Core prefix (from soliton_solver.core.params.pack_device_params):
- p_i[0..9], p_f[0..5]

Superferro appends:
- p_i[10] number_magnetization_fields
- p_i[11] number_higgs_fields
- p_i[12] number_gauge_fields

- p_f[6..23] ferromagnetic_superconductor physical constants + initial-condition controls, matching the original layout.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from soliton_solver.core.params import Params as CoreParams, ResolvedParams as CoreResolvedParams, pack_device_params as pack_core_device_params


@dataclass(frozen=True)
class Params(CoreParams):
    """
    Superferro user-facing params: core + theory specifics.
    """
    # Override default for this theory
    number_total_fields: int = 8

    # Model params (theory-specific)
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

    # Derived scales (can be overridden)
    M0: float | None = None
    u1: float | None = None
    vortex_number: float = 1.0
    ainf: float | None = None

    # Initial condition controls (theory-specific)
    skyrmion_number: float = 1.0
    skyrmion_rotation: float = 0.0
    ansatz: str = "bloch"  # "bloch" | "neel" | "anti" | "uniform"

    def resolved(self) -> "ResolvedParams":
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Superferro resolved params: core derived + theory derived.
    """
    # ints (appended to p_i)
    number_magnetization_fields: int
    number_higgs_fields: int
    number_gauge_fields: int

    # floats (appended to p_f)
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

    # initial config (stored in p_f as flags/controls)
    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
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
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack (p_i, p_f) for ferromagnetic_superconductor by:
      1) building the core prefix arrays
      2) appending theory-specific entries

    This preserves the original ABI indices expected by ferromagnetic_superconductor kernels.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([rp.number_magnetization_fields, rp.number_higgs_fields, rp.number_gauge_fields], dtype=np.int32)

    p_f_theory = np.array([
        rp.q,                                   # 6
        rp.ha,                                  # 7
        rp.hb,                                  # 8
        rp.eta1,                                # 9
        rp.eta2,                                # 10
        rp.u1,                                  # 11
        rp.vortex_number,                       # 12
        rp.ainf,                                # 13
        rp.alpha,                               # 14
        rp.beta,                                # 15
        rp.gamma,                               # 16
        rp.M0,                                  # 17
        rp.skyrmion_number,                     # 18
        rp.skyrmion_rotation,                   # 19
        1.0 if rp.ansatz_bloch else 0.0,        # 20
        1.0 if rp.ansatz_neel else 0.0,         # 21
        1.0 if rp.ansatz_anti else 0.0,         # 22
        1.0 if rp.ansatz_uniform else 0.0       # 23
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f