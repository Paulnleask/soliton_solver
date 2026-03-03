# =====================================================================================
# soliton_solver/theories/baby_skyrme/__init__.py
# =====================================================================================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Baby Skyrme model",
    aliases=("Baby Skyrme", "Baby skyrmion", "Planar skyrmion"),
    import_path="soliton_solver.theories.baby_skyrme",
    description="Solver for the baby Skyrme model, with numerous choices of potentials.",
    version="1.0"
    )