"""
Registry utilities for theory discovery, registration, loading, and description.

Examples
--------
Use ``list_theories`` to return the canonical names of all registered theories.
Use ``load_theory`` to import a theory package from the registry.
Use ``print_theory_table`` to print a table of registered theories.
Use ``discover_theories`` to find and register theory subpackages automatically.
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

    Parameters
    ----------
    name : str
        Canonical name of the theory.
    import_path : str
        Full Python import path for the theory package.
    description : str, optional
        Short description of the theory.
    version : str, optional
        Version identifier for the theory implementation.
    aliases : tuple[str, ...], optional
        Alternative names for the same theory.
    required_submodules : tuple[str, ...], optional
        Names of submodules that must exist inside the theory package.

    Returns
    -------
    None
        The dataclass stores the theory metadata.

    Examples
    --------
    Use ``TheorySpec(name="Baby Skyrme model", import_path="soliton_solver.theories.baby_skyrme", description="Solver for the baby Skyrme model")`` to define a theory specification.
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
    Register a theory specification and its aliases.

    Parameters
    ----------
    spec : TheorySpec
        Theory specification to register.

    Returns
    -------
    None
        The theory specification is added to the registry.

    Raises
    ------
    ValueError
        Raised if a name or alias is empty or already registered.

    Examples
    --------
    Use ``register_theory(spec)`` to add a theory specification to the registry.
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

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of canonical theory names.

    Examples
    --------
    Use ``list_theories()`` to list the registered theories.
    """
    canonical_names = {spec.name for spec in _REGISTRY.values()}
    return tuple(sorted(canonical_names))


def get_theory_spec(name: str) -> TheorySpec:
    """
    Retrieve the metadata specification for a registered theory.

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
        Raised if the theory name is not registered.

    Examples
    --------
    Use ``spec = get_theory_spec("Baby Skyrme")`` to retrieve a registered theory specification.
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

    Returns
    -------
    None
        The required submodules are imported for validation.

    Raises
    ------
    ImportError
        Raised if a required submodule cannot be imported.

    Examples
    --------
    Use ``_validate_theory_package(spec)`` to check that a theory package provides its required submodules.
    """
    pkg = spec.import_path

    for sub in spec.required_submodules:
        importlib.import_module(f"{pkg}.{sub}")


def load_theory(name: str) -> ModuleType:
    """
    Import and return a theory package from the registry.

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
        Raised if the theory name is not registered.
    ImportError
        Raised if the theory package or a required submodule cannot be imported.

    Examples
    --------
    Use ``theory = load_theory("Baby Skyrme")`` to import a registered theory package.
    """
    spec = get_theory_spec(name)
    mod = importlib.import_module(spec.import_path)
    _validate_theory_package(spec)
    return mod


def describe_theory(name: str) -> None:
    """
    Print a description of a registered theory.

    Parameters
    ----------
    name : str
        Canonical name or alias of the theory.

    Returns
    -------
    None
        The theory description is printed to the terminal.

    Examples
    --------
    Use ``describe_theory("Baby Skyrme model")`` to print the theory description.
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
    Print a table of registered theories.

    Returns
    -------
    None
        The table is printed to the terminal.

    Examples
    --------
    Use ``print_theory_table()`` to print a table of registered theories.
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

    Parameters
    ----------
    package : str, optional
        Import path of the parent package containing theory subpackages.

    Returns
    -------
    None
        Any discovered theory specifications are added to the registry.

    Examples
    --------
    Use ``discover_theories()`` to scan the default theory package and register available theories.
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