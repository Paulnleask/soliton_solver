"""
Spin triplet superconducting ferromagnet theory package registration and user-facing theory description.

This module defines the registry metadata for the spin triplet superconducting ferromagnet model.
It also provides a user-facing describe() function that prints a structured summary of the theory to the terminal.
The printed summary includes the theory name, version, description, aliases, available submodules, parameter information, and additional usage instructions.

Examples
--------
>>> from soliton_solver.theories import load_theory
>>> theory = load_theory("Spin triplet superconducting ferromagnet")
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
    name="Spin triplet superconducting ferromagnet",
    import_path="soliton_solver.theories.spin_triplet_superconducting_magnet",
    description="Ferromagnetic superconductor with equal-spin triplet pairing and mixed solitons, including skyrmions and multi-component vortices.",
    version="1.0"
)


def _print_section(title: str) -> None:
    """
    Print a section heading for terminal-based theory descriptions.

    Parameters
    ----------
    title : str
        Section title to print.

    Returns
    -------
    None
        This function prints a formatted section heading to the terminal.

    Examples
    --------
    >>> _print_section("Aliases")
    """
    print(title)
    print("-" * len(title))


def _print_metadata() -> None:
    """
    Print the core registry metadata for the spin triplet superconducting ferromagnet theory.

    The printed metadata includes the canonical theory name, version, short description, import path, and aliases.

    Returns
    -------
    None
        This function prints theory metadata to the terminal.

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
    Print the main spin triplet superconducting ferromagnet theory submodules available to the user.

    Returns
    -------
    None
        This function prints the main theory submodules to the terminal.

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
    Print detailed parameter information for the spin triplet superconducting ferromagnet theory.

    If the params module defines a callable describe() function, that function is used to print detailed parameter information.
    Otherwise, a fallback message is printed.

    Returns
    -------
    None
        This function prints parameter information to the terminal.

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
    Print a short high-level summary of the spin triplet superconducting ferromagnet model contents.

    This section is intended to give the user immediate orientation when inspecting the theory from the terminal.

    Returns
    -------
    None
        This function prints explanatory notes to the terminal.

    Examples
    --------
    >>> _print_notes()
    """
    _print_section("Notes")
    print("This theory package provides a ferromagnetic superconductor model with equal-spin triplet pairing.")
    print("It supports coupled magnetic and superconducting degrees of freedom together with mixed soliton configurations.")
    print("The solver includes kernels for the coupled field equations, initial-condition utilities, observable calculations, I/O helpers, and OpenGL rendering support.")
    print()


def _print_instructions() -> None:
    """
    Print additional usage instructions for the spin triplet superconducting ferromagnet theory.

    If print_instructions() is available from the local instructions module, it is called directly.

    Returns
    -------
    None
        This function prints usage instructions to the terminal.

    Examples
    --------
    >>> _print_instructions()
    """
    _print_section("Instructions")
    print_instructions()
    print()


def describe() -> None:
    """
    Print a structured terminal description of the spin triplet superconducting ferromagnet theory.

    The printed output includes registry metadata, aliases, available submodules, notes about the package contents, parameter information, and additional usage instructions.

    Returns
    -------
    None
        This function prints the spin triplet superconducting ferromagnet theory description to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories import load_theory
    >>> theory = load_theory("Spin triplet superconducting ferromagnet")
    >>> theory.describe()
    """
    _print_metadata()
    _print_submodules()
    _print_notes()
    _print_parameter_information()
    _print_instructions()