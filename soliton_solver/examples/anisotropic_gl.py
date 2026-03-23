"""
To run: python -m soliton_solver.examples.anisotropic_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Anisotropic superconductor")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=25.0, ysize=25.0,     # Grid params
        alpha1=-1.0, alpha2=-1.24503,                   # Quadratic condensate coupling
        beta1=1.0, beta2=1.4959,                        # Quartic condensate coupling
        beta3=4.02498,                                  # Density-density coupling
        Q=1.0, kappa=0.673274,                          # Gauge & magnetic coupling
        gamma1=1.0, gamma2=0.877868, gamma12=0.534484,  # Anisotropy matrix coefficients
        newtonflow=False                                # Start simulation with flow on/off (True/False)
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()