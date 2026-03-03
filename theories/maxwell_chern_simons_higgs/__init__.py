# =========================
# soliton_solver/theories/maxwell_chern_simons_higgs/__init__.py
# =========================
from soliton_solver.theories.registry import TheorySpec

from . import params
from . import kernels
from . import initial_config
from . import observables
from . import io
from . import render_gl

THEORY_SPEC = TheorySpec(
    name="Anyon superconductor",
    aliases=("Maxwell Chern-Simons Higgs", "Chern-Simons-Landau-Ginzburg", "Abelian Chern-Simons Higgs"),
    import_path="soliton_solver.theories.maxwell_chern_simons_higgs",
    description="Solver for ferromagnetic superconductors with mixed solitons (skyrmions + vortices).",
    version="1.0"
    )