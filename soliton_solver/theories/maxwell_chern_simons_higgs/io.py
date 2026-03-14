"""
Output bundle for the Ginzburg-Landau superconductor theory.

Examples
--------
>>> output_data_bundle("out", h_Field, h_EnergyDensity, h_VortexDensity, h_Supercurrent, h_NoetherChargeDensity, h_ElectricChargeDensity, h_grid, xlen, ylen)
"""
from __future__ import annotations
import numpy as np
from soliton_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_VortexDensity: np.ndarray,
    h_Supercurrent: np.ndarray,
    h_NoetherChargeDensity: np.ndarray,
    h_ElectricChargeDensity: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int = 2,
    number_total_fields: int = 5,
    precision: int = 32,
) -> None:
    """
    Write the simulation output bundle using the filenames expected by the plotting workflow.

    Parameters
    ----------
    output_dir : str
        Output directory.
    h_Field : ndarray
        Flattened field array.
    h_EnergyDensity : ndarray
        Energy density grid.
    h_VortexDensity : ndarray
        Vortex density grid.
    h_Supercurrent : ndarray
        Supercurrent grid.
    h_NoetherChargeDensity : ndarray
        Noether charge density grid.
    h_ElectricChargeDensity : ndarray
        Electric charge density grid.
    h_grid : ndarray
        Coordinate grid array.
    xlen : int
        Number of lattice points in x.
    ylen : int
        Number of lattice points in y.
    number_coordinates : int, optional
        Number of coordinate components.
    number_total_fields : int, optional
        Number of field components.
    precision : int, optional
        Output precision.

    Returns
    -------
    None

    Examples
    --------
    >>> output_data_bundle("out", h_Field, h_EnergyDensity, h_VortexDensity, h_Supercurrent, h_NoetherChargeDensity, h_ElectricChargeDensity, h_grid, xlen, ylen)
    """
    arrays = {"grid": h_grid, "Field": h_Field, "EnergyDensity": h_EnergyDensity, "VortexDensity": h_VortexDensity, "Supercurrent": h_Supercurrent, "NoetherChargeDensity": h_NoetherChargeDensity, "ElectricChargeDensity": h_ElectricChargeDensity}

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("Field", 0, "GaugeField1.dat"),
        ("Field", 1, "GaugeField2.dat"),
        ("Field", 2, "HiggsField1.dat"),
        ("Field", 3, "HiggsField2.dat"),
        ("Field", 4, "GaugeField0.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat"),
        ("VortexDensity", 0, "VortexDensity.dat"),
        ("Supercurrent", 0, "Supercurrent1.dat"),
        ("Supercurrent", 1, "Supercurrent2.dat"),
        ("NoetherChargeDensity", 0, "NoetherChargeDensity.dat"),
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