"""
Write the output bundle for the anisotropic s+id superconductor theory.

Examples
--------
>>> output_data_bundle("out", h_Field, h_EnergyDensity, h_MagneticFluxDensity, h_BaryonDensity, h_Supercurrent, h_grid, xlen, ylen)
"""

from __future__ import annotations
import numpy as np
from soliton_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_MagneticFluxDensity: np.ndarray,
    h_Supercurrent: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int = 2,
    number_total_fields: int = 6,
    precision: int = 32,
) -> None:
    """
    Write the theory output bundle using the standard core I/O routine.

    Parameters
    ----------
    output_dir : str
        Output directory path.
    h_Field : numpy.ndarray
        Flattened field array.
    h_EnergyDensity : numpy.ndarray
        Energy density grid.
    h_MagneticFluxDensity : numpy.ndarray
        Magnetic flux density grid.
    h_BaryonDensity : numpy.ndarray
        Magnetization charge density grid.
    h_Supercurrent : numpy.ndarray
        Supercurrent density grid.
    h_grid : numpy.ndarray
        Flattened coordinate grid.
    xlen : int
        Lattice size along the x direction.
    ylen : int
        Lattice size along the y direction.
    number_coordinates : int, optional
        Number of coordinate components.
    number_total_fields : int, optional
        Number of field components.
    precision : int, optional
        Output formatting precision.

    Returns
    -------
    None
        Output files are written to the specified directory.

    Examples
    --------
    >>> output_data_bundle("out", h_Field, h_EnergyDensity, h_MagneticFluxDensity, h_BaryonDensity, h_Supercurrent, h_grid, xlen, ylen)
    """
    arrays = {"grid": h_grid, "Field": h_Field, "EnergyDensity": h_EnergyDensity, "MagneticFluxDensity": h_MagneticFluxDensity, "Supercurrent": h_Supercurrent}

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("Field", 0, "GaugeField1.dat"),
        ("Field", 1, "GaugeField2.dat"),
        ("Field", 2, "HiggsField1.dat"),
        ("Field", 3, "HiggsField2.dat"),
        ("Field", 4, "HiggsField3.dat"),
        ("Field", 5, "HiggsField4.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat"),
        ("MagneticFluxDensity", 0, "ChargeDensity.dat"),
        ("Supercurrent", 0, "Supercurrent1.dat"),
        ("Supercurrent", 1, "Supercurrent2.dat")
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