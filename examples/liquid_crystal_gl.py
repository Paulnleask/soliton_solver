# =====================================================================================
# liquid_crystal_gl.py
# =====================================================================================
"""
To run: python -m soliton_solver.examples.liquid_crystal_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("liquid crystal")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=10.0, ysize=10.0, # Grid params 
        K=10e-12, P=7.0e-6, d=4e-6,                 # Liquid crystals params
        w0=1.0, voltage=4.0, delta_eps=3.7,         # Homeotropic anchoring & applied electric field
        e1=2e-12, e3=4e-12,                         # Flexoelectric coefficients
        eps0=8.854e-12,                             # Override vacuum permittivity
        deformation="Splay-bend", ansatz="neel",    # Deformation + skyrmion type
        depol=True,                                 # Depolarization flag
        newtonflow=False,                           # Start simulation with flow on/off (True/False)
        unit_magnetization=True                     # Required flag for magnetization
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()