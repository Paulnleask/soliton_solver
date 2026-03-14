"""
Write the liquid crystal theory output bundle.

Examples
--------
>>> output_data_bundle(output_dir, h_Field, h_EnergyDensity, h_BaryonDensity, h_ElectricChargeDensity, h_grid, xlen, ylen)
"""
from __future__ import annotations
import numpy as np
from soliton_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_BaryonDensity: np.ndarray,
    h_ElectricChargeDensity: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int = 2,
    number_total_fields: int = 4,
    precision: int = 32,
) -> None:
    """
    Write the output files used by the plotting workflow.

    Parameters
    ----------
    output_dir : str
        Directory where output files are written.
    h_Field : numpy.ndarray
        Host field array.
    h_EnergyDensity : numpy.ndarray
        Host energy density array.
    h_BaryonDensity : numpy.ndarray
        Host skyrmion charge density array.
    h_ElectricChargeDensity : numpy.ndarray
        Host electric charge density array.
    h_grid : numpy.ndarray
        Host coordinate grid array.
    xlen : int
        Number of lattice points in the x direction.
    ylen : int
        Number of lattice points in the y direction.
    number_coordinates : int, optional
        Number of coordinate components.
    number_total_fields : int, optional
        Number of field components.
    precision : int, optional
        Output precision.

    Examples
    --------
    >>> output_data_bundle(output_dir, h_Field, h_EnergyDensity, h_BaryonDensity, h_ElectricChargeDensity, h_grid, xlen, ylen)
    """
    arrays = {
        "grid": h_grid,
        "Field": h_Field,
        "EnergyDensity": h_EnergyDensity,
        "BaryonDensity": h_BaryonDensity,
        "ElectricChargeDensity": h_ElectricChargeDensity
    }

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("Field", 0, "MagnetField1.dat"),
        ("Field", 1, "MagnetField2.dat"),
        ("Field", 2, "MagnetField3.dat"),
        ("BaryonDensity", 0, "SkyrmionChargeDensity.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat"),
        ("ElectricChargeDensity", 0, "ElectricChargeDensity.dat")
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