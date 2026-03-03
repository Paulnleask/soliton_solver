# =========================
# soliton_solver/theories/maxwell_chern_simons_higgs/params.py
# =========================
"""
Superferro theory-specific parameters.

This module extends the core Params/ResolvedParams and appends entries to p_i/p_f so that
the existing CUDA ABI indices are unchanged.

Core prefix (from soliton_solver.core.params.pack_device_params):
- p_i[0..9], p_f[0..5]

Superferro appends:
- p_i[10] number_higgs_fields
- p_i[11] number_gauge_fields

- p_f[6..11] maxwell_chern_simons_higgs physical constants + initial-condition controls, matching the original layout.
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
    number_total_fields: int = 5

    # Model params (theory-specific)
    number_higgs_fields: int = 2
    number_gauge_fields: int = 2
    q: float = 1.0
    Lambda: float = 1.0
    kappa: float = 0.0
    u1: float = 1.0

    # Derived scales (can be overridden)
    vortex_number: float = 1.0
    ainf: float | None = None

    def resolved(self) -> "ResolvedParams":
        return ResolvedParams.from_params(self)


@dataclass(frozen=True)
class ResolvedParams(CoreResolvedParams):
    """
    Superferro resolved params: core derived + theory derived.
    """
    # ints (appended to p_i)
    number_higgs_fields: int
    number_gauge_fields: int

    # floats (appended to p_f)
    q: float
    Lambda: float
    kappa: float
    u1: float
    vortex_number: float
    ainf: float

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        core = CoreResolvedParams.from_params(p)
        ainf = p.ainf if p.ainf is not None else (p.vortex_number / p.q)
        return ResolvedParams(xlen=core.xlen, ylen=core.ylen, halo=core.halo, number_coordinates=core.number_coordinates, number_total_fields=core.number_total_fields, dim_grid=core.dim_grid, dim_fields=core.dim_fields, killkinen=core.killkinen, newtonflow=core.newtonflow, unit_magnetization=core.unit_magnetization, xsize=core.xsize, ysize=core.ysize, lsx=core.lsx, lsy=core.lsy, grid_volume=core.grid_volume, time_step=core.time_step, number_higgs_fields=p.number_higgs_fields, number_gauge_fields=p.number_gauge_fields, q=float(p.q), Lambda=float(p.Lambda), kappa=float(p.kappa), u1=float(p.u1), vortex_number=float(p.vortex_number), ainf=float(ainf))


def default_params(**overrides) -> Params:
    return Params().with_(**overrides)


def pack_device_params(rp: ResolvedParams):
    """
    Pack (p_i, p_f) for maxwell_chern_simons_higgs by:
      1) building the core prefix arrays
      2) appending theory-specific entries

    This preserves the original ABI indices expected by maxwell_chern_simons_higgs kernels.
    """
    p_i_core, p_f_core = pack_core_device_params(rp)

    p_i_theory = np.array([rp.number_higgs_fields, rp.number_gauge_fields], dtype=np.int32)

    p_f_theory = np.array([
        rp.q,                                   # 6
        rp.Lambda,                              # 7
        rp.kappa,                               # 8
        rp.u1,                                  # 9
        rp.vortex_number,                       # 10
        rp.ainf,                                # 11
    ], dtype=np.float64)

    p_i = np.concatenate((p_i_core, p_i_theory))
    p_f = np.concatenate((p_f_core, p_f_theory))
    return p_i, p_f