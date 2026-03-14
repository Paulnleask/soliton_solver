"""
Public user facing interface for theory discovery, loading, and description.

Examples
--------
Use ``theories.list()`` to return the canonical names of the registered theories.
Use ``theories.print_table()`` to print a table of the available theories.
Use ``theories.print_description("Baby Skyrme")`` to print a theory description.
"""

from __future__ import annotations

from .registry import TheorySpec, register_theory, discover_theories, load_theory, list_theories, get_theory_spec, describe_theory, print_theory_table


_discovered = False

if not _discovered:
    discover_theories()
    _discovered = True


def list() -> tuple[str, ...]:
    """
    Return the canonical names of all registered theories.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of canonical theory names.

    Examples
    --------
    Use ``theories.list()`` to list the available theories.
    """
    return list_theories()


def print_description(name: str) -> None:
    """
    Print a description of a registered theory.

    Parameters
    ----------
    name : str
        Canonical theory name or alias.

    Returns
    -------
    None
        The theory description is printed to the terminal.

    Examples
    --------
    Use ``theories.print_description("Baby Skyrme")`` to print a theory description.
    """
    describe_theory(name)


def print_table() -> None:
    """
    Print a table of the available registered theories.

    Returns
    -------
    None
        The table is printed to the terminal.

    Examples
    --------
    Use ``theories.print_table()`` to print the theory table.
    """
    print_theory_table()