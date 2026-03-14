"""
Write Bose-Einstein condensate output files using the theory specific filenames.

Examples
--------
Use ``output_data_bundle`` to write field, grid, and energy density files for plotting.
"""
from __future__ import annotations
import numpy as np
from soliton_solver.core.io import output_data_bundle_core

def output_data_bundle(
    output_dir: str,
    h_Field: np.ndarray,
    h_EnergyDensity: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int = 2,
    number_total_fields: int = 2,
    precision: int = 32,
) -> None:
    """
    Write the standard Bose-Einstein condensate output bundle.

    Parameters
    ----------
    output_dir : str
        Output directory path.
    h_Field : ndarray
        Host field array.
    h_EnergyDensity : ndarray
        Host energy density array.
    h_grid : ndarray
        Host coordinate grid array.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    number_coordinates : int, optional
        Number of coordinate components.
    number_total_fields : int, optional
        Number of field components.
    precision : int, optional
        Number of significant digits used in the output format.

    Returns
    -------
    None
        The output bundle is written to disk.

    Raises
    ------
    OSError
        Raised if an output directory or file cannot be created or written.

    Examples
    --------
    Use ``output_data_bundle(output_dir, h_Field, h_EnergyDensity, h_grid, xlen, ylen)`` to write the output files.
    """
    arrays = {"grid": h_grid, "Field": h_Field, "EnergyDensity": h_EnergyDensity}

    bundle_spec = [
        ("grid", 0, "xGrid.dat"),
        ("grid", 1, "yGrid.dat"),
        ("Field", 0, "HiggsField1.dat"),
        ("Field", 1, "HiggsField2.dat"),
        ("EnergyDensity", 0, "EnergyDensity.dat")
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