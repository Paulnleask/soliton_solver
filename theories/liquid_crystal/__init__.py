# =========================
# soliton_solver/theories/liquid_crystal/__init__.py
# =========================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Liquid crystal",
    aliases=("Chiral liquid crystal", "Liquid crystal skyrmion", "Nematic liquid crystal"),
    import_path="soliton_solver.theories.liquid_crystal",
    description="Solver for liquid crystal systems, twist or splay & bend favoured, with flexoelectric depolarization.",
    version="1.0"
    )