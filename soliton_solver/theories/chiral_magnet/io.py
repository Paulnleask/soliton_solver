"""
Chiral magnet theory output bundle.

Examples
--------
>>> output_data_bundle("out", h_Field, h_EnergyDensity, h_BaryonDensity, h_MagneticChargeDensity, h_grid, xlen, ylen)
"""

from __future__ import annotations
import numpy as np
from soliton_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_BaryonDensity: np.ndarray,
    h_MagneticChargeDensity: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int = 2,
    number_total_fields: int = 4,
    precision: int = 32,
) -> None:
    """
    Write the standard output bundle for the chiral magnet theory.

    Parameters
    ----------
    output_dir : str
        Directory for output files.
    h_Field : ndarray
        Host array containing the field configuration.
    h_EnergyDensity : ndarray
        Host array containing the energy density grid.
    h_BaryonDensity : ndarray
        Host array containing the baryon density grid.
    h_MagneticChargeDensity : ndarray
        Host array containing the magnetic charge density grid.
    h_grid : ndarray
        Host coordinate grid.
    xlen : int
        Grid extent along the x direction.
    ylen : int
        Grid extent along the y direction.
    number_coordinates : int, optional
        Number of coordinate components.
    number_total_fields : int, optional
        Number of field components.
    precision : int, optional
        Floating point output precision.

    Examples
    --------
    >>> output_data_bundle("out", h_Field, h_EnergyDensity, h_BaryonDensity, h_MagneticChargeDensity, h_grid, xlen, ylen)
    """
    arrays = {
        "grid": h_grid,
        "Field": h_Field,
        "EnergyDensity": h_EnergyDensity,
        "BaryonDensity": h_BaryonDensity,
        "MagneticChargeDensity": h_MagneticChargeDensity,
    }

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("Field", 0, "MagnetField1.dat"),
        ("Field", 1, "MagnetField2.dat"),
        ("Field", 2, "MagnetField3.dat"),
        ("BaryonDensity", 0, "SkyrmionChargeDensity.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat"),
        ("MagneticChargeDensity", 0, "MagneticChargeDensity.dat"),
    ]

    output_data_bundle_core(
        output_dir,
        h_Field=h_Field,
        h_grid=h_grid,
        xlen=xlen,
        ylen=ylen,
        number_coordinates=number_coordinates,
        number_total_fields=number_total_fields,
        precision=precision,
        bundle_spec=bundle_spec,
        arrays=arrays,
        lattice_points_name="LatticePoints.dat",
        lattice_vectors_name="LatticeVectors.dat",
        field_dump_name="Field.dat",
    )