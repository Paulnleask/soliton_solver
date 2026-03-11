"""
Public user-facing interface for theory discovery, loading, and description.

This module re-exports the main theory registry functionality through a simpler
API intended for user code. It performs one-time automatic discovery of theory
subpackages when imported, so that drop-in theories become available without
manual registration.

The module also provides friendly wrapper functions for listing available
theories, printing a theory description, and printing a CLI-style theory table.

Examples
--------
>>> from soliton_solver import theories
>>> theories.list()
('Baby Skyrme model',)
>>> theories.print_table()
Theory             Version  Description                                      Aliases
------------------------------------------------------------------------------------
Baby Skyrme model  1.0      Solver for the baby Skyrme model, with nume...  Baby Skyrme, Baby skyrmion, Planar skyrmion
>>> theories.print_description("Baby Skyrme")
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

    This is a user-facing wrapper around list_theories().

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of canonical theory names.

    Examples
    --------
    >>> from soliton_solver import theories
    >>> theories.list()
    ('Baby Skyrme model',)
    """
    return list_theories()


def print_description(name: str) -> None:
    """
    Print a description of a registered theory to the terminal.

    This is a user-facing wrapper around describe_theory(). The theory may be
    specified using either its canonical name or any registered alias.

    Parameters
    ----------
    name : str
        Canonical theory name or alias.

    Examples
    --------
    >>> from soliton_solver import theories
    >>> theories.print_description("Baby Skyrme")
    """
    describe_theory(name)


def print_table() -> None:
    """
    Print a CLI-style table of available registered theories.

    The printed table shows canonical theory names only. Aliases are grouped
    into a separate aliases column.

    Examples
    --------
    >>> from soliton_solver import theories
    >>> theories.print_table()
    """
    print_theory_table()