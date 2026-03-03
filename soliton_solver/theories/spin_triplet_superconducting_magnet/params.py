# =========================
# soliton_solver/theories/spin_triplet_superconducting_magnet/params.py
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

- p_f[6..23] superferro physical constants + initial-condition controls, matching the original layout.
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
    number_total_fields: int = 10

    # Model params (theory-specific)
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

    # Derived scales (can be overridden)
    M0: float | None = None
    u1: float | None = None
    u2: float | None = None
    vortex_number: float = 1.0
    vortex1_number: float = 1.0
    vortex2_number: float = 1.0
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

    # initial config (stored in p_f as flags/controls)
    skyrmion_rotation: float
    ansatz_bloch: bool
    ansatz_neel: bool
    ansatz_anti: bool
    ansatz_uniform: bool

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
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
        return ResolvedParams(xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields, dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization, xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step, number_magnetization_fields=p.number_magnetization_fields, number_higgs_fields=p.number_higgs_fields, number_gauge_fields=p.number_gauge_fields, q=float(p.q), alpha=float(p.alpha), beta=float(p.beta), gamma=float(p.gamma), skyrmion_number=float(p.skyrmion_number), ha=float(p.ha), hb1=float(p.hb1), hb2=float(p.hb2), hc=float(p.hc), M0=float(M0), u1=float(u1), u2=float(u2), vortex_number=float(vortex_number), vortex1_number = float(p.vortex1_number), vortex2_number = float(p.vortex2_number), ainf=float(ainf), skyrmion_rotation=float(p.skyrmion_rotation), ansatz_bloch=ansatz_bloch, ansatz_neel=ansatz_neel, ansatz_anti=ansatz_anti, ansatz_uniform=ansatz_uniform)


def default_params(**overrides) -> Params:
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack (p_i, p_f) for superferro by:
      1) building the core prefix arrays
      2) appending theory-specific entries

    This preserves the original ABI indices expected by superferro kernels.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([rp.number_magnetization_fields, rp.number_higgs_fields, rp.number_gauge_fields], dtype=np.int32)

    p_f_theory = np.array([
        rp.q,                                   # 6
        rp.ha,                                  # 7
        rp.hb1,                                 # 8
        rp.hb2,                                 # 9
        rp.hc,                                  # 10
        rp.u1,                                  # 11
        rp.u2,                                  # 12
        rp.vortex1_number,                      # 13
        rp.vortex2_number,                      # 14
        rp.vortex_number,                       # 15
        rp.ainf,                                # 16
        rp.alpha,                               # 17
        rp.beta,                                # 18
        rp.gamma,                               # 19
        rp.M0,                                  # 20
        rp.skyrmion_number,                     # 21
        rp.skyrmion_rotation,                   # 22
        1.0 if rp.ansatz_bloch else 0.0,        # 23
        1.0 if rp.ansatz_neel else 0.0,         # 24
        1.0 if rp.ansatz_anti else 0.0,         # 25
        1.0 if rp.ansatz_uniform else 0.0       # 26
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f