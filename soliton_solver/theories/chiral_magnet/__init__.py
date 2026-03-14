"""
Chiral magnet theory package registration and terminal description.

Examples
--------
>>> from soliton_solver.theories import load_theory
>>> theory = load_theory("Chiral magnet")
>>> theory.describe()
"""
from __future__ import annotations

from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl
from .instructions import print_instructions

THEORY_SPEC = TheorySpec(
    name="Chiral magnet",
    aliases=("Chiral ferromagnet", "Chiral magnetic skyrmion", "Magnetic skyrmion"),
    import_path="soliton_solver.theories.chiral_magnet",
    description="Skyrmions in chiral ferromagnetic systems, with various DMI choices and optional demagnetization.",
    version="1.1"
)

def _print_section(title: str) -> None:
    """
    Print a section heading.

    Parameters
    ----------
    title : str
        Section title.

    Returns
    -------
    None
        The section heading is printed to the terminal.

    Examples
    --------
    >>> _print_section("Aliases")
    """
    print(title)
    print("-" * len(title))

def _print_metadata() -> None:
    """
    Print the theory metadata.

    Returns
    -------
    None
        The theory metadata are printed to the terminal.

    Examples
    --------
    >>> _print_metadata()
    """
    print("=" * 72)
    print(f"{THEORY_SPEC.name} (version {THEORY_SPEC.version})")
    print("=" * 72)
    print(THEORY_SPEC.description)
    print()
    print(f"Import path: {THEORY_SPEC.import_path}")
    print()

    _print_section("Aliases")

    if THEORY_SPEC.aliases:
        print(", ".join(THEORY_SPEC.aliases))
    else:
        print("None")

    print()

def _print_submodules() -> None:
    """
    Print the main theory submodules.

    Returns
    -------
    None
        The submodule names are printed to the terminal.

    Examples
    --------
    >>> _print_submodules()
    """
    _print_section("Main submodules")
    print("params")
    print("kernels")
    print("initial_config")
    print("observables")
    print("io")
    print("render_gl")
    print()

def _print_parameter_information() -> None:
    """
    Print parameter information for the theory.

    Returns
    -------
    None
        The parameter information is printed to the terminal.

    Examples
    --------
    >>> _print_parameter_information()
    """
    _print_section("Parameter information")

    describe_fn = getattr(params, "describe", None)

    if callable(describe_fn):
        describe_fn()
    else:
        print("No detailed parameter description is defined.")

    print()

def _print_notes() -> None:
    """
    Print a short summary of the theory package.

    Returns
    -------
    None
        The summary notes are printed to the terminal.

    Examples
    --------
    >>> _print_notes()
    """
    _print_section("Notes")
    print("This theory package provides a chiral magnet model for magnetic skyrmions.")
    print("It includes kernels for the field equations, initial condition utilities, observable calculations, I/O helpers, and OpenGL rendering support.")
    print("The model supports multiple DMI choices and can optionally include demagnetization effects.")
    print()

def _print_instructions() -> None:
    """
    Print usage instructions for the theory.

    Returns
    -------
    None
        The usage instructions are printed to the terminal.

    Examples
    --------
    >>> _print_instructions()
    """
    _print_section("Instructions")
    print_instructions()
    print()

def describe() -> None:
    """
    Print a structured terminal description of the theory.

    Returns
    -------
    None
        The theory description is printed to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories import load_theory
    >>> theory = load_theory("Chiral magnet")
    >>> theory.describe()
    """
    _print_metadata()
    _print_submodules()
    _print_notes()
    _print_parameter_information()
    _print_instructions()