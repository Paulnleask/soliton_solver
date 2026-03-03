# =========================
# soliton_solver/theories/__init__.py
# =========================
from __future__ import annotations
from .registry import TheorySpec, register_theory, discover_theories, load_theory, list_theories, get_theory_spec

# Run plugin discovery once on import, so drop-in theories auto-appear.
# If you want to disable auto-discovery for faster startup, comment this out and use explicit registration.
_discovered = False
if not _discovered:
    discover_theories()
    _discovered = True

# Friendly aliases for the user-facing API
def list() -> tuple[str, ...]:
    return list_theories()

def describe(name: str) -> TheorySpec:
    return get_theory_spec(name)