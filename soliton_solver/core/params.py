"""
Core theory agnostic parameter management for soliton_solver.

Examples
--------
Use ``default_params`` to create a parameter set with optional overrides.
Use ``Params.resolved`` to derive solver ready parameters.
Use ``pack_device_params`` to build the core device parameter arrays.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class Params:
    """
    User facing core parameters shared by all theories.

    Parameters
    ----------
    xlen : int, optional
        Number of lattice points along the x direction.
    ylen : int, optional
        Number of lattice points along the y direction.
    halo : int, optional
        Halo width used by finite difference stencils.
    number_coordinates : int, optional
        Number of spatial coordinates.
    number_total_fields : int, optional
        Total number of field components.
    xsize : float, optional
        Physical size of the domain along the x direction.
    ysize : float, optional
        Physical size of the domain along the y direction.
    lsx : float or None, optional
        Grid spacing along the x direction.
    lsy : float or None, optional
        Grid spacing along the y direction.
    courant : float, optional
        Factor used to derive the time step when it is not set explicitly.
    time_step : float or None, optional
        Integration or flow step size.
    killkinen : bool, optional
        Flag controlling whether the velocity is zeroed after an energy increase.
    newtonflow : bool, optional
        Flag selecting Newton flow style stepping.
    unit_magnetization : bool, optional
        Flag enabling a unit magnetization constraint.

    Returns
    -------
    None
        The dataclass stores the core solver configuration.

    Examples
    --------
    Use ``p = Params(number_total_fields=6)`` to create a parameter set.
    Use ``p2 = p.with_(xlen=512, ylen=512)`` to create a modified copy.
    Use ``rp = p2.resolved()`` to derive solver ready parameters.
    """
    xlen: int = 256
    ylen: int = 256
    halo: int = 2
    number_coordinates: int = 2
    number_total_fields: int = 0

    xsize: float = 80.0
    ysize: float = 80.0

    lsx: float | None = None
    lsy: float | None = None

    courant: float = 0.5
    time_step: float | None = None
    killkinen: bool = True
    newtonflow: bool = True
    unit_magnetization: bool = False

    def with_(self, **kwargs) -> "Params":
        """
        Return a copy of the parameter set with selected fields overridden.

        Parameters
        ----------
        **kwargs
            Dataclass field overrides.

        Returns
        -------
        Params
            New parameter set with the requested overrides applied.

        Examples
        --------
        Use ``p.with_(xlen=512, time_step=0.01)`` to override selected fields.
        """
        return replace(self, **kwargs)

    def resolved(self) -> "ResolvedParams":
        """
        Resolve derived solver parameters.

        Returns
        -------
        ResolvedParams
            Parameter set containing derived spacings, volumes, dimensions, and device flags.

        Examples
        --------
        Use ``rp = p.resolved()`` to construct the resolved parameter set.
        """
        return ResolvedParams.from_params(self)

@dataclass(frozen=True)
class ResolvedParams:
    """
    Solver ready parameter set derived from ``Params``.

    Parameters
    ----------
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    halo : int
        Halo width used by finite difference stencils.
    number_coordinates : int
        Number of spatial coordinates.
    number_total_fields : int
        Total number of field components.
    dim_grid : int
        Total number of lattice sites.
    dim_fields : int
        Total number of stored field values.
    killkinen : int
        Integer flag for arrested flow behaviour.
    newtonflow : int
        Integer flag selecting Newton flow stepping.
    unit_magnetization : int
        Integer flag enabling a unit magnetization constraint.
    xsize : float
        Physical size of the domain along the x direction.
    ysize : float
        Physical size of the domain along the y direction.
    lsx : float
        Grid spacing along the x direction.
    lsy : float
        Grid spacing along the y direction.
    grid_volume : float
        Area of one lattice cell.
    time_step : float
        Integration or flow step size.

    Returns
    -------
    None
        The dataclass stores fully resolved solver parameters.

    Examples
    --------
    Use ``rp = ResolvedParams.from_params(p)`` to build a resolved parameter set.
    """
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

    xsize: float
    ysize: float
    lsx: float
    lsy: float
    grid_volume: float
    time_step: float

    @staticmethod
    def from_params(p: Params) -> "ResolvedParams":
        """
        Construct resolved parameters from a ``Params`` instance.

        Parameters
        ----------
        p : Params
            Input parameter set.

        Returns
        -------
        ResolvedParams
            Resolved parameter set with derived spacings, dimensions, volume, time step, and integer flags.

        Examples
        --------
        Use ``rp = ResolvedParams.from_params(p)`` to derive solver parameters from ``p``.
        """
        lsx = p.lsx if p.lsx is not None else p.xsize / (p.xlen - 1)
        lsy = p.lsy if p.lsy is not None else p.ysize / (p.ylen - 1)
        grid_volume = lsx * lsy
        time_step = p.time_step if p.time_step is not None else (p.courant * lsx)
        dim_grid = p.xlen * p.ylen
        dim_fields = int(p.number_total_fields) * dim_grid
        return ResolvedParams(xlen=p.xlen, ylen=p.ylen, halo=p.halo, number_coordinates=p.number_coordinates, number_total_fields=int(p.number_total_fields), dim_grid=dim_grid, dim_fields=dim_fields, killkinen=1 if p.killkinen else 0, newtonflow=1 if p.newtonflow else 0, unit_magnetization=1 if p.unit_magnetization else 0, xsize=float(p.xsize), ysize=float(p.ysize), lsx=float(lsx), lsy=float(lsy), grid_volume=float(grid_volume), time_step=float(time_step))

def default_params(**overrides) -> Params:
    """
    Create a default parameter set with optional overrides.

    Parameters
    ----------
    **overrides
        Dataclass field overrides passed to ``Params.with_``.

    Returns
    -------
    Params
        Parameter set with the requested overrides applied.

    Examples
    --------
    Use ``default_params(number_total_fields=6, xlen=512)`` to create a customized default parameter set.
    """
    return Params().with_(**overrides)

def pack_device_params(rp: ResolvedParams):
    """
    Pack the core device parameter arrays.

    Parameters
    ----------
    rp : ResolvedParams
        Resolved parameter set.

    Returns
    -------
    tuple of ndarray
        Pair ``(p_i, p_f)`` containing the integer and floating point device parameter arrays.

    Examples
    --------
    Use ``p_i_core, p_f_core = pack_device_params(rp)`` to build the core device parameter arrays.
    """
    p_i = np.array([rp.xlen, rp.ylen, rp.halo, rp.number_coordinates, rp.number_total_fields, rp.dim_grid, rp.dim_fields, rp.killkinen, rp.newtonflow, rp.unit_magnetization], dtype=np.int32)
    p_f = np.array([rp.xsize, rp.ysize, rp.lsx, rp.lsy, rp.grid_volume, rp.time_step], dtype=np.float64)
    return p_i, p_f