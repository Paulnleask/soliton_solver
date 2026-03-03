# =========================
# soliton_solver/theories/liquid_crystal/io.py
# =========================
"""
Superferro theory-specific output bundle.

This preserves the exact filenames expected by the existing plotting.py workflow, while
keeping the core I/O implementation theory-agnostic.
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
    arrays = {"grid": h_grid, "Field": h_Field, "EnergyDensity": h_EnergyDensity, "BaryonDensity": h_BaryonDensity}

    bundle_spec = [
        ("grid", 0, "Higgs_xGrid.dat"),
        ("grid", 1, "Higgs_yGrid.dat"),
        ("Field", 0, "Magnet_Field1.dat"),
        ("Field", 1, "Magnet_Field2.dat"),
        ("Field", 2, "Magnet_Field3.dat"),
        ("BaryonDensity", 0, "Magnet_ChargeDensity.dat"),
        ("EnergyDensity", 0, "Higgs_EnergyDensity.dat"),
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
        lattice_points_name="Higgs_LatticePoints.dat",
        lattice_vectors_name="Higgs_LatticeVectors.dat",
        field_dump_name="d_Field.dat",
    )