# =========================
# soliton_solver/theories/spin_triplet_superconducting_magnet/io.py
# =========================
"""
Spin-triplet superferro theory-specific output bundle.

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
        ("Field", 8, "HiggsField3.dat"),
        ("Field", 9, "HiggsField4.dat"),
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