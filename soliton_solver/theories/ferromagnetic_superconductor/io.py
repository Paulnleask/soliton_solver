"""
Write the ferromagnetic superconductor data bundle using the core output utilities.

Examples
--------
>>> output_data_bundle("output", h_Field, h_EnergyDensity, h_MagneticFluxDensity, h_BaryonDensity, h_Supercurrent, h_grid, 256, 256)
"""

from __future__ import annotations
import numpy as np
from soliton_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_MagneticFluxDensity: np.ndarray,
    h_BaryonDensity: np.ndarray,
    h_Supercurrent: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int = 2,
    number_total_fields: int = 8,
    precision: int = 32,
) -> None:
    """
    Write the ferromagnetic superconductor simulation output bundle.

    Parameters
    ----------
    output_dir : str
        Directory where output files are written.
    h_Field : ndarray
        Host array containing the field configuration.
    h_EnergyDensity : ndarray
        Host array containing the energy density.
    h_MagneticFluxDensity : ndarray
        Host array containing the magnetic flux density.
    h_BaryonDensity : ndarray
        Host array containing the magnetization charge density.
    h_Supercurrent : ndarray
        Host array containing the supercurrent density.
    h_grid : ndarray
        Host array containing the coordinate grid.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    number_coordinates : int, optional
        Number of coordinate components stored in the grid.
    number_total_fields : int, optional
        Number of field components.
    precision : int, optional
        Output precision for text files.

    Returns
    -------
    None
        The output files are written to the specified directory.

    Examples
    --------
    >>> output_data_bundle("output", h_Field, h_EnergyDensity, h_MagneticFluxDensity, h_BaryonDensity, h_Supercurrent, h_grid, 256, 256)
    """
    arrays = {"grid": h_grid, "Field": h_Field, "EnergyDensity": h_EnergyDensity, "MagneticFluxDensity": h_MagneticFluxDensity, "BaryonDensity": h_BaryonDensity, "Supercurrent": h_Supercurrent}

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("Field", 0, "MagnetField1.dat"),
        ("Field", 1, "MagnetField2.dat"),
        ("Field", 2, "MagnetField3.dat"),
        ("Field", 3, "GaugeField1.dat"),
        ("Field", 4, "GaugeField2.dat"),
        ("Field", 5, "GaugeField3.dat"),
        ("Field", 6, "HiggsField1.dat"),
        ("Field", 7, "HiggsField2.dat"),
        ("BaryonDensity", 0, "MagnetChargeDensity.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat"),
        ("MagneticFluxDensity", 0, "ChargeDensityX.dat"),
        ("MagneticFluxDensity", 1, "ChargeDensityY.dat"),
        ("MagneticFluxDensity", 2, "ChargeDensity.dat"),
        ("Supercurrent", 0, "Supercurrent1.dat"),
        ("Supercurrent", 1, "Supercurrent2.dat"),
        ("Supercurrent", 2, "Supercurrent3.dat"),
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