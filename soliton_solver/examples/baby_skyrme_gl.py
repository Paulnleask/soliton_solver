# =====================================================================================
# baby_skyrme_gl.py
# =====================================================================================
"""
To run: python -m soliton_solver.examples.baby_skyrme_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Baby Skyrme model")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=20.0, ysize=20.0, # Grid params
        time_step=0.006,                            # Manual time step override
        mpi=1.0, kappa=1.0,                         # Baby Skyrme params
        potential="standard",                       # Potential term(s)
        ansatz="neel",                              # Skyrmion type
        newtonflow=False,                           # Start simulation with flow on/off (True/False)
        unit_magnetization=True                     # Required flag for magnetization
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()