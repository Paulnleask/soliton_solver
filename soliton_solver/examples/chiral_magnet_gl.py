# =====================================================================================
# chiral_magnet_gl.py
# =====================================================================================
"""
To run: python -m soliton_solver.examples.chiral_magnet_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Chiral magnet")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=10.0, ysize=10.0,     # Grid params 
        J=40e-12, K=0.8e+6, D=4e-3, M=580e+3, B=0e-3,   # Chiral magnetic params
        mu0=1.25663706127e-6,                           # Override vacuum permeability
        dmi_term="Heusler", ansatz="anti",              # DMI + skyrmion type
        demag=True,                                     # Demagnetization flag
        newtonflow=False,                               # Start simulation with flow on/off (True/False)
        unit_magnetization=True                         # Required flag for magnetization
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()