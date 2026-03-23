"""
To run: python -m soliton_solver.examples.bose_einstein_condensate_gl
"""
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Bose-Einstein condensate")

def run_gl_simulation():
    params = theory.params.default_params(
        N=1000.0,                                       # Number of atoms
        m=1.45e-25,                                     # Atomic mass (kg)
        omega=200.0,                                    # Rotational frequency (Hz)
        omega_rot=0.99,                                 # Ratio of rotational frequencies (always < 1.0)
        a_0=0.529e-10,                                  # Bohr radius (m)
        a_s=109.0*0.529e-10,                            # s-wave scattering length (m)
        vortex_number=7.0,                              # Number of initial vortices
        xlen=320, ylen=320, xsize=24.0, ysize=24.0,     # Grid params 
        newtonflow=False                                # Start simulation with flow on/off (True/False)
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    sim.print_instructions()
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()