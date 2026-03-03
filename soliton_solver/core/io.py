# =====================================================================================
# soliton_solver/core/io.py
# =====================================================================================
"""
Host-side I/O utilities for writing simulation outputs as plain-text .dat files.

Purpose:
    Provides theory-agnostic output routines for exporting lattice data,
    full field configurations, and optional per-component grids.

Core functionality:
    - Flattened indexing helper for host arrays.
    - Writers for full multi-component fields.
    - Writers for scalar iteration data.
    - Writers for individual per-component density grids.
    - A bundle writer that produces a consistent output directory structure.

Usage:
    A theory constructs a bundle_spec list:
        [(array_key, component_index, filename), ...]
    and calls output_data_bundle_core(...) with:
        - Host field and grid arrays,
        - Lattice sizes and metadata,
        - Optional additional arrays mapped by name.

Outputs:
    - Creates an output directory (if needed).
    - Writes lattice size and lattice vectors files.
    - Writes full field dump.
    - Writes any additional per-component grids defined in bundle_spec.
"""
# ---------------- Imports ----------------
from __future__ import annotations
import os
import numpy as np

# ---------------- Array flattening ----------------
def _flat(a: int, x: int, y: int, xlen: int, ylen: int) -> int:
    """
    Compute flattened (a, x, y) index for arrays stored as [a][x][y] in a 1D buffer.

    Usage:
        i = _flat(a, x, y, xlen, ylen)
        value = field_flat[i]

    Parameters:
        a: Component index.
        x, y: Lattice indices.
        xlen, ylen: Lattice extents.

    Outputs:
        - Returns the integer flat index into a 1D array storing components and grid points.
    """
    return y + x * ylen + a * xlen * ylen

# ---------------- Outputs field data ----------------
def output_field_dat(field_flat: np.ndarray, path: str, xlen: int, ylen: int, nfields: int, precision: int = 32) -> None:
    """
    Write a full multi-component field dump as a plain-text .dat file.

    Usage:
        output_field_dat(h_Field, "out/d_Field.dat", xlen, ylen, nfields, precision=32)

    Parameters:
        field_flat: Flattened field array containing all components and grid points.
        path: Output file path.
        xlen, ylen: Lattice extents.
        nfields: Number of field components to write.
        precision: Significant digits for formatting (default 32).

    Outputs:
        - Creates/overwrites `path`.
        - Writes each component as an x-by-y tab-separated grid, separated by a blank line between components.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for a in range(nfields):
            for x in range(xlen):
                for y in range(ylen):
                    f.write(fmt.format(float(field_flat[_flat(a, x, y, xlen, ylen)])))
                    f.write("\t")
                f.write("\n")
            f.write("\n")

# ---------------- Outputs iteration data ----------------
def output_iteration_data_dat(values, path: str, precision: int = 32) -> None:
    """
    Write a 1D list of scalar values as a single tab-separated line in a .dat file.

    Usage:
        output_iteration_data_dat([energy, err], "out/IterationData.dat", precision=16)

    Parameters:
        values: Iterable of scalar values.
        path: Output file path.
        precision: Significant digits for formatting (default 32).

    Outputs:
        - Creates/overwrites `path`.
        - Writes all values on one line separated by tabs.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for v in values:
            f.write(fmt.format(float(v)))
            f.write("\t")

# ---------------- Outputs density data ----------------
def output_density_data_dat(density_flat: np.ndarray, a: int, path: str, xlen: int, ylen: int, precision: int = 32) -> None:
    """
    Write a single component (a) of a flattened per-component grid as a plain-text .dat file.

    Usage:
        output_density_data_dat(density_flat, a=0, path="out/Density0.dat", xlen=xlen, ylen=ylen, precision=32)

    Parameters:
        density_flat: Flattened array containing one or more per-component x-by-y grids.
        a: Component index to write.
        path: Output file path.
        xlen, ylen: Lattice extents.
        precision: Significant digits for formatting (default 32).

    Outputs:
        - Creates/overwrites `path`.
        - Writes the selected component as an x-by-y tab-separated grid.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for x in range(xlen):
            for y in range(ylen):
                f.write(fmt.format(float(density_flat[_flat(a, x, y, xlen, ylen)])))
                f.write("\t")
            f.write("\n")

# ---------------- Outputs data bundle for plotting ----------------
def output_data_bundle_core(
    output_dir: str,
    *,
    h_Field: np.ndarray,
    h_grid: np.ndarray,
    xlen: int,
    ylen: int,
    number_coordinates: int,
    number_total_fields: int,
    precision: int,
    bundle_spec: list[tuple[str, int, str]],
    arrays: dict[str, np.ndarray],
    lattice_points_name: str = "LatticePoints.dat",
    lattice_vectors_name: str = "LatticeVectors.dat",
    field_dump_name: str = "d_Field.dat",
) -> None:
    """
    Write a theory-agnostic output bundle (lattice metadata, full field dump, and optional extra grids).

    Usage:
        output_data_bundle_core(
            output_dir,
            h_Field=h_Field,
            h_grid=h_grid,
            xlen=xlen,
            ylen=ylen,
            number_coordinates=2,
            number_total_fields=nfields,
            precision=32,
            bundle_spec=[("Density", 0, "Density0.dat")],
            arrays={"Density": density_flat},
        )

    Parameters:
        output_dir: Directory to create/use for outputs.
        h_Field: Flattened host field array to dump.
        h_grid: Flattened host coordinate grid (components are coordinate axes).
        xlen, ylen: Lattice extents.
        number_coordinates: Number of coordinate components available in h_grid (e.g. 1 or 2).
        number_total_fields: Number of field components in h_Field.
        precision: Significant digits for formatting.
        bundle_spec: List of (array_key, component_index, filename) for optional per-component grid outputs.
        arrays: Mapping from array_key to flattened numpy arrays holding per-component grids.
        lattice_points_name: Filename for lattice size output (default "LatticePoints.dat").
        lattice_vectors_name: Filename for lattice vectors output (default "LatticeVectors.dat").
        field_dump_name: Filename for full field dump (default "d_Field.dat").

    Outputs:
        - Creates `output_dir` if needed.
        - Writes lattice point counts to `lattice_points_name`.
        - Writes lattice vectors (derived from max grid coordinates) to `lattice_vectors_name`.
        - Writes the full field dump to `field_dump_name`.
        - For each (array_key, a, filename) in bundle_spec present in `arrays`, writes component `a` to `filename`.
    """
    os.makedirs(output_dir, exist_ok=True)

    lattice_points = np.array([xlen, ylen], dtype=np.float64)
    x_max = float(h_grid[_flat(0, xlen - 1, ylen - 1, xlen, ylen)]) if number_coordinates >= 1 else 0.0
    y_max = float(h_grid[_flat(1, xlen - 1, ylen - 1, xlen, ylen)]) if number_coordinates >= 2 else 0.0
    lattice_vectors = np.array([x_max, 0.0, 0.0, y_max], dtype=np.float64)

    output_iteration_data_dat(lattice_points[:number_coordinates], os.path.join(output_dir, lattice_points_name), precision=precision)
    output_iteration_data_dat(lattice_vectors[:number_coordinates * number_coordinates], os.path.join(output_dir, lattice_vectors_name), precision=precision)

    output_field_dat(h_Field, os.path.join(output_dir, field_dump_name), xlen, ylen, number_total_fields, precision=precision)

    for array_key, a, filename in bundle_spec:
        if array_key not in arrays:
            continue
        arr = arrays[array_key]
        if arr is None:
            continue
        output_density_data_dat(arr, int(a), os.path.join(output_dir, filename), xlen, ylen, precision=precision)