# =====================================================================================
# soliton_solver/core/params.py
# =====================================================================================
"""
Core (theory-agnostic) parameter management for soliton_solver.

Purpose:
    Defines user-facing parameters (Params), derives solver-ready values
    (ResolvedParams), and packs the core ABI prefix arrays (p_i, p_f) for device code.

Usage:
    p = default_params(number_total_fields=..., xlen=..., ylen=..., ...)
    rp = p.resolved()
    p_i_core, p_f_core = pack_device_params(rp)  # theories may append entries

Outputs:
    - Produces a stable core prefix layout for p_i and p_f used by CUDA kernels.
    - Provides resolved grid spacings, volumes, dimensions, dt, and device flags.
"""
# ---------------- Imports ----------------
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

# ---------------- Main parameters ----------------
@dataclass(frozen=True)
class Params:
    """
    User-facing core parameters shared by all theories.

    Usage:
        p = Params(number_total_fields=6)
        p2 = p.with_(xlen=512, ylen=512)
        rp = p2.resolved()

    Parameters:
        xlen, ylen: Grid extents.
        halo: Halo width used by finite-difference stencils.
        number_coordinates: Spatial coordinate count (typically 2).
        number_total_fields: Total field components (theory-defined).
        xsize, ysize: Physical domain sizes.
        lsx, lsy: Grid spacings; if None, derived from size and length.
        courant: CFL-like factor used to derive time_step when unset.
        time_step: Integration/flow step; if None, derived from courant and lsx.
        killkinen: Host-side flag controlling energy-increase arrest behavior.
        newtonflow: Host-side flag selecting Newton-flow style stepping.
        unit_magnetization: Device-side flag enabling unit constraint if supported.

    Outputs:
        - Stores core solver configuration with defaults.
    """
    # Grid
    xlen: int = 256
    ylen: int = 256
    halo: int = 2
    number_coordinates: int = 2
    number_total_fields: int = 0  # theory should set appropriately

    # Physical size
    xsize: float = 80.0
    ysize: float = 80.0

    # Spacings (if None, computed from size/len)
    lsx: float | None = None
    lsy: float | None = None

    # Solver
    courant: float = 0.5
    time_step: float | None = None
    killkinen: bool = True
    newtonflow: bool = True
    unit_magnetization: bool = False  # theory may enable; default false for generic solvers

    # Override parameters
    def with_(self, **kwargs) -> "Params":
        """
        Create a modified Params instance with selected fields overridden.

        Usage:
            p2 = p.with_(xlen=512, time_step=0.01)

        Parameters:
            **kwargs: Dataclass field overrides.

        Outputs:
            - Returns a new Params with the provided overrides applied.
        """
        return replace(self, **kwargs)

    # Resolve derived parameters
    def resolved(self) -> "ResolvedParams":
        """
        Resolve derived solver parameters and normalize flags for device use.

        Usage:
            rp = p.resolved()

        Parameters:
            None

        Outputs:
            - Returns a ResolvedParams instance with derived spacings, volumes, dimensions,
            time_step, and integer device flags.
        """
        return ResolvedParams.from_params(self)

# ---------------- Resolved parameters ----------------
@dataclass(frozen=True)
class ResolvedParams:
    """
    Solver-ready parameter set derived from Params.

    Usage:
        rp = ResolvedParams.from_params(p)

    Parameters:
        (See dataclass fields.)
        Integers include dimensions and 0/1 flags for device code.
        Floats include physical sizes, spacings, cell volume, and time_step.

    Outputs:
        - Stores fully derived values required by core kernels and host drivers.
    """
    # ints
    xlen: int
    ylen: int
    halo: int
    number_coordinates: int
    number_total_fields: int
    dim_grid: int
    dim_fields: int
    killkinen: int
    newtonflow: int
    unit_magnetization: int

    # floats
    xsize: float
    ysize: float
    lsx: float
    lsy: float
    grid_volume: float
    time_step: float

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Construct ResolvedParams from Params by deriving spacings, volumes, dt, and dimensions.

        Usage:
            rp = ResolvedParams.from_params(p)

        Parameters:
            p: Params instance.

        Outputs:
            - Computes lsx/lsy from xsize/ysize and xlen/ylen if not provided.
            - Computes grid_volume = lsx * lsy.
            - Computes time_step from courant * lsx if not provided.
            - Computes dim_grid = xlen * ylen and dim_fields = number_total_fields * dim_grid.
            - Converts boolean flags into 0/1 integers for device-side use.
            - Returns the constructed ResolvedParams instance.
        """
        lsx = p.lsx if p.lsx is not None else p.xsize / (p.xlen - 1)
        lsy = p.lsy if p.lsy is not None else p.ysize / (p.ylen - 1)
        grid_volume = lsx * lsy
        time_step = p.time_step if p.time_step is not None else (p.courant * lsx)
        dim_grid = p.xlen * p.ylen
        dim_fields = int(p.number_total_fields) * dim_grid
        return ResolvedParams(xlen=p.xlen, ylen=p.ylen, halo=p.halo, number_coordinates=p.number_coordinates, number_total_fields=int(p.number_total_fields), dim_grid=dim_grid, dim_fields=dim_fields, killkinen=1 if p.killkinen else 0, newtonflow=1 if p.newtonflow else 0, unit_magnetization=1 if p.unit_magnetization else 0, xsize=float(p.xsize), ysize=float(p.ysize), lsx=float(lsx), lsy=float(lsy), grid_volume=float(grid_volume), time_step=float(time_step))

# ---------------- Default parameters ----------------
def default_params(**overrides) -> Params:
    """
    Create a Params instance with optional overrides.

    Usage:
        p = default_params(number_total_fields=6, xlen=512)

    Parameters:
        **overrides: Dataclass field overrides passed to Params.with_().

    Outputs:
        - Returns a Params instance with overrides applied.
    """
    return Params().with_(**overrides)

# ---------------- Default parameters ----------------
def pack_device_params(rp: ResolvedParams):
    """
    Pack the core ABI prefix arrays (p_i, p_f) for device kernels.

    Usage:
        rp = p.resolved()
        p_i_core, p_f_core = pack_device_params(rp)

    Parameters:
        rp: ResolvedParams instance.

    Outputs:
        - Returns (p_i_core, p_f_core) as numpy arrays:
            p_i_core: np.int32 with fixed core layout (length 10).
            p_f_core: np.float64 with fixed core layout (length 6).
        - Provides only the core prefix; theories may append additional entries.
    """
    p_i = np.array([rp.xlen, rp.ylen, rp.halo, rp.number_coordinates, rp.number_total_fields, rp.dim_grid, rp.dim_fields, rp.killkinen, rp.newtonflow, rp.unit_magnetization], dtype=np.int32)
    p_f = np.array([rp.xsize, rp.ysize, rp.lsx, rp.lsy, rp.grid_volume, rp.time_step], dtype=np.float64)
    return p_i, p_f