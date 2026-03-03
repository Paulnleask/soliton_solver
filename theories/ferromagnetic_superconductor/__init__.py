# =========================
# soliton_solver/theories/ferromagnetic_superconductor/__init__.py
# =========================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Ferromagnetic superconductor",
    aliases=("Superconducting ferromagnet", "Superconducting magnet", "Magnetic skyrmions superconducting vortices", "skyrmion vortex pairs"),
    import_path="soliton_solver.theories.ferromagnetic_superconductor",
    description="Solver for ferromagnetic superconductors with mixed solitons (skyrmions + vortices).",
    version="1.0"
    )