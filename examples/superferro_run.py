# =========================
# superferro_run.py
# =========================
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Ferromagnetic superconductor")

params = theory.params.default_params(
    xlen=256, ylen=256, xsize=80.0, ysize=80.0,
    ansatz="bloch",
    killkinen=True, newtonflow=True, unit_magnetization=True)
sim = Simulation(params, theory)
sim.initialize({"mode": "initial", "ansatz": "bloch", "skyrmion_rotation": 0.0, "vortex_number": 1.0})
results = sim.minimize(tol=1e-4, max_steps=2000, log_every=100, verbose=True)
sim.save_output("output", precision=32)