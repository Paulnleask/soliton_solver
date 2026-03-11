"""
Registry utilities for theory discovery, registration, loading, and description.

This module implements the central registry used to manage available theories
within the soliton_solver framework. It allows theory packages to register
metadata describing the theory, supports dynamic importing of theory packages,
validates that required theory submodules exist, and enables automatic
discovery of theory packages placed inside the soliton_solver.theories
namespace.

The registry supports both canonical theory names and aliases. Canonical names
are used for listing and display, while aliases remain available for lookup.

Examples
--------
>>> from soliton_solver.theories.registry import list_theories, print_theory_table
>>> list_theories()
('Baby Skyrme model',)
>>> print_theory_table()
Theory             Version  Description                                         Aliases
---------------------------------------------------------------------------------------
Baby Skyrme model  1.0      Solver for the baby Skyrme model, with numerous...  Baby Skyrme, Baby skyrmion, Planar skyrmion
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Dict
import importlib
import pkgutil


@dataclass(frozen=True)
class TheorySpec:
    """
    Immutable metadata describing a theory implementation.

    This structure defines the information required for the registry to locate
    and validate a theory package.

    Parameters
    ----------
    name : str
        Canonical name of the theory.
    import_path : str
        Full Python import path for the theory package.
    description : str, optional
        Short human-readable description of the theory.
    version : str, optional
        Version identifier for the theory implementation.
    aliases : tuple[str, ...], optional
        Alternative names that may also refer to the same theory.
    required_submodules : tuple[str, ...], optional
        Names of submodules that must exist inside the theory package.

    Examples
    --------
    >>> spec = TheorySpec(
    ...     name="Baby Skyrme model",
    ...     import_path="soliton_solver.theories.baby_skyrme",
    ...     description="Solver for the baby Skyrme model",
    ... )
    """

    name: str
    import_path: str
    description: str = ""
    version: str = "0.0"
    aliases: tuple[str, ...] = ()
    required_submodules: tuple[str, ...] = ("kernels", "initial_config", "observables")


_REGISTRY: Dict[str, TheorySpec] = {}


def register_theory(spec: TheorySpec) -> None:
    """
    Register a theory specification and all of its aliases.

    The theory is registered using its canonical name and each alias.
    All keys are normalized to lowercase.

    Parameters
    ----------
    spec : TheorySpec
        Theory specification object describing the theory.

    Raises
    ------
    ValueError
        If a name or alias is empty or already registered.

    Examples
    --------
    >>> register_theory(spec)
    """
    keys = [spec.name, *spec.aliases]

    for k in keys:
        key = (k or "").strip().lower()

        if not key:
            raise ValueError("TheorySpec names must be non-empty")

        if key in _REGISTRY:
            raise ValueError(f"Theory '{key}' is already registered")

        _REGISTRY[key] = spec


def list_theories() -> tuple[str, ...]:
    """
    Return the canonical names of all registered theories.

    Aliases are not included in this listing. This function is intended for
    user-facing display of the available theories.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of canonical theory names.

    Examples
    --------
    >>> list_theories()
    ('Baby Skyrme model',)
    """
    canonical_names = {spec.name for spec in _REGISTRY.values()}
    return tuple(sorted(canonical_names))


def get_theory_spec(name: str) -> TheorySpec:
    """
    Retrieve the metadata specification for a registered theory.

    The lookup may be performed using either the canonical theory name or any
    registered alias.

    Parameters
    ----------
    name : str
        Canonical name or alias of the theory.

    Returns
    -------
    TheorySpec
        Metadata describing the requested theory.

    Raises
    ------
    KeyError
        If the theory name is not registered.

    Examples
    --------
    >>> spec = get_theory_spec("Baby Skyrme")
    >>> spec.name
    'Baby Skyrme model'
    """
    key = (name or "").strip().lower()

    if key not in _REGISTRY:
        raise KeyError(f"Unknown theory '{name}'. Available: {list_theories()}")

    return _REGISTRY[key]


def _validate_theory_package(spec: TheorySpec) -> None:
    """
    Validate that the required submodules exist in a theory package.

    Parameters
    ----------
    spec : TheorySpec
        Specification describing the theory.

    Raises
    ------
    ImportError
        If any required submodule cannot be imported.

    Examples
    --------
    >>> spec = get_theory_spec("Baby Skyrme model")
    >>> _validate_theory_package(spec)
    """
    pkg = spec.import_path

    for sub in spec.required_submodules:
        importlib.import_module(f"{pkg}.{sub}")


def load_theory(name: str) -> ModuleType:
    """
    Import and return a theory package from the registry.

    The theory package is imported dynamically and validated to ensure that
    required submodules exist.

    Parameters
    ----------
    name : str
        Canonical name or alias of the theory.

    Returns
    -------
    ModuleType
        Imported theory package module.

    Raises
    ------
    KeyError
        If the theory name is not registered.
    ImportError
        If the theory package or required submodules fail to import.

    Examples
    --------
    >>> theory = load_theory("Baby Skyrme")
    >>> theory.describe()
    """
    spec = get_theory_spec(name)
    mod = importlib.import_module(spec.import_path)
    _validate_theory_package(spec)
    return mod


def describe_theory(name: str) -> None:
    """
    Print a description of a theory to the terminal.

    If the theory package defines a callable describe() function, that
    function is used. Otherwise the registry metadata is printed.

    Parameters
    ----------
    name : str
        Canonical name or alias of the theory.

    Examples
    --------
    >>> describe_theory("Baby Skyrme model")
    """
    spec = get_theory_spec(name)

    print(f"Theory: {spec.name}")
    print(f"Import path: {spec.import_path}")
    print(f"Version: {spec.version}")

    if spec.aliases:
        print("Aliases:", ", ".join(spec.aliases))

    if spec.description:
        print(f"Description: {spec.description}")


def print_theory_table() -> None:
    """
    Print a CLI-style table of registered theories.

    The table displays canonical theory names only, with aliases grouped into a
    separate column. This makes it easier to distinguish the primary theory
    names from alternative lookup names.

    Examples
    --------
    >>> print_theory_table()
    """
    specs = sorted({spec for spec in _REGISTRY.values()}, key=lambda spec: spec.name.lower())

    if not specs:
        print("No theories registered.")
        return

    theory_header = "Theory"
    version_header = "Version"
    description_header = "Description"

    theory_width = max(len(theory_header), max(len(spec.name) for spec in specs))
    version_width = max(len(version_header), max(len(spec.version) for spec in specs))
    description_width = max(len(description_header), max(len(spec.description) for spec in specs))

    header = (
        f"{theory_header:<{theory_width}}  "
        f"{version_header:<{version_width}}  "
        f"{description_header:<{description_width}}  "
    )

    separator = "-" * len(header)

    print(header)
    print(separator)

    for spec in specs:
        print(
            f"{spec.name:<{theory_width}}  "
            f"{spec.version:<{version_width}}  "
            f"{spec.description:<{description_width}}  "
        )


def discover_theories(package: str = "soliton_solver.theories") -> None:
    """
    Discover and register theory subpackages automatically.

    A subpackage is registered if it defines a THEORY_SPEC object that is an
    instance of TheorySpec.

    Parameters
    ----------
    package : str, optional
        Import path of the parent package containing theory subpackages.

    Examples
    --------
    >>> discover_theories()
    """
    pkg = importlib.import_module(package)

    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.ispkg:
            continue

        if m.name in ("__pycache__",):
            continue

        import_path = f"{package}.{m.name}"

        try:
            subpkg = importlib.import_module(import_path)
        except Exception:
            continue

        spec = getattr(subpkg, "THEORY_SPEC", None)

        if isinstance(spec, TheorySpec):
            keys = [spec.name, *spec.aliases]

            for k in keys:
                key = (k or "").strip().lower()

                if not key:
                    continue

                if key not in _REGISTRY:
                    _REGISTRY[key] = spec