# =====================================================================================
# spin_triplet_gl.py
# =====================================================================================
"""
To run: python -m soliton_solver.examples.spin_triplet_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Spin triplet superconducting ferromagnet")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=80.0, ysize=80.0, # Grid params
        alpha=-1.0, beta=1.0,                       # Magnetization params
        ha=-4.0, hb1=3.0, hb2=0.5, hc=-0.0,         # Superconductor params
        ansatz="bloch",                             # Skyrmion type
        newtonflow=False,                           # Start simulation with flow on/off (True/False)
        unit_magnetization=True                     # Required flag for magnetization
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})  # or {"mode": "initial", "ansatz": "bloch", "skyrmion_rotation": 0.0, "vortex_number": 1.0}
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()