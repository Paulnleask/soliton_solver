"""
To run: python -m soliton_solver.examples.anyon_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Anyon superconductor")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=50.0, ysize=50.0,     # Grid params 
        q=1.0, Lambda=0.5, u1=1.0, kappa=0.5,           # Abelian Chern-Simons Higgs parameters
        newtonflow=False                                # Start simulation with flow on/off (True/False)
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()