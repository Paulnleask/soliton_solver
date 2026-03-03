# =========================
# soliton_solver/theories/registry.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Dict, Optional
import importlib
import pkgutil


@dataclass(frozen=True)
class TheorySpec:
    name: str
    import_path: str
    description: str = ""
    version: str = "0.0"
    aliases: tuple[str, ...] = ()
    required_submodules: tuple[str, ...] = ("kernels", "initial_config", "observables")


_REGISTRY: Dict[str, TheorySpec] = {}


def register_theory(spec: TheorySpec) -> None:
    keys = [spec.name, *spec.aliases]
    for k in keys:
        key = (k or "").strip().lower()
        if not key:
            raise ValueError("TheorySpec names must be non-empty")
        if key in _REGISTRY:
            raise ValueError(f"Theory '{key}' is already registered")
        _REGISTRY[key] = spec


def list_theories() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


def get_theory_spec(name: str) -> TheorySpec:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown theory '{name}'. Available: {list_theories()}")
    return _REGISTRY[key]


def _validate_theory_package(mod: ModuleType, spec: TheorySpec) -> None:
    # Ensure required submodules can be imported (keeps failure local + clear)
    pkg = spec.import_path
    for sub in spec.required_submodules:
        importlib.import_module(f"{pkg}.{sub}")


def load_theory(name: str) -> ModuleType:
    """
    Returns the imported theory *package* module.
    Example: load_theory("superferro") -> soliton_solver.theories.superferro
    """
    spec = get_theory_spec(name)
    mod = importlib.import_module(spec.import_path)
    _validate_theory_package(mod, spec)
    return mod


def discover_theories(package: str = "soliton_solver.theories") -> None:
    """
    Auto-register any subpackage that defines THEORY_SPEC: TheorySpec.
    This allows plugin-like drop-in theories without touching central code.
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
            # Register canonical name + aliases, but do not crash on duplicates during discovery
            keys = [spec.name, *spec.aliases]
            for k in keys:
                key = (k or "").strip().lower()
                if not key:
                    continue
                if key not in _REGISTRY:
                    _REGISTRY[key] = spec