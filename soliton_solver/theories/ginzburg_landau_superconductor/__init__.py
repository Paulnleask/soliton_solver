# =========================
# soliton_solver/theories/ginzburg_landau_superconductor/__init__.py
# =========================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Ginzburg-Landau superconductor",
    aliases=("Abelian Higgs", "Ginzburg-Landau", "Abelian Higgs model", "Ginzburg-Landau model"),
    import_path="soliton_solver.theories.ginzburg_landau_superconductor",
    description="Solver for Ginzburg-Landau superconductors with superconducting vortices).",
    version="1.0"
    )