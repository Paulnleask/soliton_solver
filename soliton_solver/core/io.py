"""
Host side I/O utilities for writing simulation outputs as plain text .dat files.

Examples
--------
Use ``output_field_dat`` to write a full multi component field to disk.
Use ``output_iteration_data_dat`` to write scalar iteration data to disk.
Use ``output_density_data_dat`` to write a single component grid to disk.
Use ``output_data_bundle_core`` to write a complete output bundle for plotting.
"""

from __future__ import annotations
import os
import numpy as np

def _flat(a: int, x: int, y: int, xlen: int, ylen: int) -> int:
    """
    Compute the flattened index for a component and lattice site.

    Parameters
    ----------
    a : int
        Component index.
    x : int
        Lattice index along the x direction.
    y : int
        Lattice index along the y direction.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.

    Returns
    -------
    int
        Flattened index into the one dimensional array.

    Examples
    --------
    Use ``i = _flat(a, x, y, xlen, ylen)`` to access a flattened field or grid array.
    """
    return y + x * ylen + a * xlen * ylen

def output_field_dat(field_flat: np.ndarray, path: str, xlen: int, ylen: int, nfields: int, precision: int = 32) -> None:
    """
    Write a full multi component field to a plain text .dat file.

    Parameters
    ----------
    field_flat : ndarray
        Flattened field array containing all components and lattice sites.
    path : str
        Output file path.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    nfields : int
        Number of field components to write.
    precision : int, optional
        Number of significant digits used in the output format.

    Returns
    -------
    None
        The field data are written to ``path``.

    Raises
    ------
    OSError
        Raised if the output file cannot be opened or written.

    Examples
    --------
    Use ``output_field_dat(h_Field, "out/d_Field.dat", xlen, ylen, nfields, precision=32)`` to write the full field.
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

def output_iteration_data_dat(values, path: str, precision: int = 32) -> None:
    """
    Write scalar values as a single tab separated line in a plain text .dat file.

    Parameters
    ----------
    values : iterable
        Scalar values to write.
    path : str
        Output file path.
    precision : int, optional
        Number of significant digits used in the output format.

    Returns
    -------
    None
        The values are written to ``path``.

    Raises
    ------
    OSError
        Raised if the output file cannot be opened or written.

    Examples
    --------
    Use ``output_iteration_data_dat([energy, err], "out/IterationData.dat", precision=16)`` to write iteration data.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for v in values:
            f.write(fmt.format(float(v)))
            f.write("\t")

def output_density_data_dat(density_flat: np.ndarray, a: int, path: str, xlen: int, ylen: int, precision: int = 32) -> None:
    """
    Write one component of a flattened grid to a plain text .dat file.

    Parameters
    ----------
    density_flat : ndarray
        Flattened array containing one or more component grids.
    a : int
        Component index to write.
    path : str
        Output file path.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    precision : int, optional
        Number of significant digits used in the output format.

    Returns
    -------
    None
        The selected component grid is written to ``path``.

    Raises
    ------
    OSError
        Raised if the output file cannot be opened or written.

    Examples
    --------
    Use ``output_density_data_dat(density_flat, 0, "out/Density0.dat", xlen, ylen, precision=32)`` to write one component grid.
    """
    fmt = f"{{:.{precision}g}}"
    with open(path, "w", newline="\n") as f:
        for x in range(xlen):
            for y in range(ylen):
                f.write(fmt.format(float(density_flat[_flat(a, x, y, xlen, ylen)])))
                f.write("\t")
            f.write("\n")

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
    Write lattice metadata, a full field dump, and optional component grids.

    Parameters
    ----------
    output_dir : str
        Directory used for all output files.
    h_Field : ndarray
        Flattened host field array.
    h_grid : ndarray
        Flattened host coordinate grid.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    number_coordinates : int
        Number of coordinate components stored in ``h_grid``.
    number_total_fields : int
        Number of field components stored in ``h_Field``.
    precision : int
        Number of significant digits used in the output format.
    bundle_spec : list of tuple
        List of ``(array_key, component_index, filename)`` entries describing optional outputs.
    arrays : dict of ndarray
        Mapping from array names to flattened arrays used for optional outputs.
    lattice_points_name : str, optional
        Filename for the lattice point counts.
    lattice_vectors_name : str, optional
        Filename for the lattice vectors.
    field_dump_name : str, optional
        Filename for the full field dump.

    Returns
    -------
    None
        The output directory and requested files are written to disk.

    Raises
    ------
    OSError
        Raised if the output directory or any output file cannot be created or written.

    Examples
    --------
    Use ``output_data_bundle_core(output_dir, h_Field=h_Field, h_grid=h_grid, xlen=xlen, ylen=ylen, number_coordinates=2, number_total_fields=nfields, precision=32, bundle_spec=[("Density", 0, "Density0.dat")], arrays={"Density": density_flat})`` to write a complete output bundle.
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