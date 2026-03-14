"""
Register the Ginzburg-Landau superconductor theory and print its terminal description.

Examples
--------
>>> from soliton_solver.theories import load_theory
>>> theory = load_theory("Ginzburg-Landau superconductor")
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
    name="Ginzburg-Landau superconductor",
    aliases=("Abelian Higgs", "Ginzburg-Landau", "Abelian Higgs model", "Ginzburg-Landau model"),
    import_path="soliton_solver.theories.ginzburg_landau_superconductor",
    description="Superconducting vortices in the Ginzburg-Landau or Abelian Higgs model.",
    version="1.0"
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
        The main submodules are printed to the terminal.

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
    Print the theory parameter information.

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
    Print summary notes about the theory package.

    Returns
    -------
    None
        The summary notes are printed to the terminal.

    Examples
    --------
    >>> _print_notes()
    """
    _print_section("Notes")
    print("This theory package provides a Ginzburg-Landau or Abelian Higgs model for superconducting vortices.")
    print("It includes kernels for the field equations, initial condition utilities, observable calculations, I/O helpers, and OpenGL rendering support.")
    print("The model describes coupled superconducting and gauge fields in a vortex-forming regime.")
    print()

def _print_instructions() -> None:
    """
    Print the theory usage instructions.

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
    Print a structured description of the theory.

    Returns
    -------
    None
        The theory description is printed to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories import load_theory
    >>> theory = load_theory("Ginzburg-Landau superconductor")
    >>> theory.describe()
    """
    _print_metadata()
    _print_submodules()
    _print_notes()
    _print_parameter_information()
    _print_instructions()