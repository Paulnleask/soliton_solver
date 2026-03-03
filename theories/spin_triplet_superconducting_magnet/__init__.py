# =========================
# soliton_solver/theories/spin_triplet_superconducting_magnet/__init__.py
# =========================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Spin triplet superconducting ferromagnet",
    import_path="soliton_solver.theories.spin_triplet_superconducting_magnet",
    description="Ferromagnetic superconductor, with equal spin triplet pairing, mixed soliton solver (skyrmions + multi-component vortices).",
    version="1.0"
    )