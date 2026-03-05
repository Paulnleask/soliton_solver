# =====================================================================================
# super_ferro_gl.py
# =====================================================================================
"""
To run: python -m soliton_solver.examples.super_ferro_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Ferromagnetic superconductor")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=80.0, ysize=80.0, # Grid params
        alpha=-1.0, beta=1.0,                       # Magnetizaton params
        ha=-4.0, hb=1.0,                            # Superconductor params
        eta1=0.0,                                   # Spin-flip scattering param
        ansatz="bloch",                             # Skyrmion type
        newtonflow=False,                           # Start simulation with flow on/off (True/False)
        unit_magnetization=True                     # Required flag for magnetization
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()