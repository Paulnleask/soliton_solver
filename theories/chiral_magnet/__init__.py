# =========================
# soliton_solver/theories/chiral_magnet/__init__.py
# =========================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Chiral magnet",
    aliases=("Chiral ferromagnet", "Chiral magnetic skyrmion", "Magnetic skyrmion"),
    import_path="soliton_solver.theories.chiral_magnet",
    description="Solver for chiral ferromagnetic systems, with various DMI choices and also accounts for demagnetization,.",
    version="1.0"
    )