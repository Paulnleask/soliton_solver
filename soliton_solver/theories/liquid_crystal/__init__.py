"""
Liquid crystal theory package registration and user-facing theory description.

This module defines the registry metadata for the liquid crystal model.
It also provides a user-facing describe() function that prints a structured summary of the theory to the terminal.
The printed summary includes the theory name, version, description, aliases, available submodules, parameter information, and additional usage instructions.

Examples
--------
>>> from soliton_solver.theories import load_theory
>>> theory = load_theory("Liquid crystal")
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
    name="Liquid crystal",
    aliases=("Chiral liquid crystal", "Liquid crystal skyrmion", "Nematic liquid crystal"),
    import_path="soliton_solver.theories.liquid_crystal",
    description="Skyrmions in liquid crystal systems with twist or splay-bend preference and flexoelectric depolarization.",
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
    Print the core registry metadata for the liquid crystal theory.

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
    Print the main liquid crystal theory submodules available to the user.

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
    Print detailed parameter information for the liquid crystal theory.

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
    Print a short high-level summary of the liquid crystal model contents.

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
    print("This theory package provides a liquid crystal model for chiral and nematic textures, including skyrmion-like configurations.")
    print("It includes kernels for the field equations, initial-condition utilities, observable calculations, I/O helpers, and OpenGL rendering support.")
    print("The model supports twist-favoured or splay-bend-favoured regimes together with flexoelectric depolarization effects.")
    print()


def _print_instructions() -> None:
    """
    Print additional usage instructions for the liquid crystal theory.

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
    Print a structured terminal description of the liquid crystal theory.

    The printed output includes registry metadata, aliases, available submodules, notes about the package contents, parameter information, and additional usage instructions.

    Returns
    -------
    None
        This function prints the liquid crystal theory description to the terminal.

    Examples
    --------
    >>> from soliton_solver.theories import load_theory
    >>> theory = load_theory("Liquid crystal")
    >>> theory.describe()
    """
    _print_metadata()
    _print_submodules()
    _print_notes()
    _print_parameter_information()
    _print_instructions()