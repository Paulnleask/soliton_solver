<p align="center">
  <img <img src="https://raw.githubusercontent.com/Paulnleask/soliton_solver/main/soliton_solver/assets/soliton_solver_logo.png" width="500">
</p>

<h4 align="center">
GPU-based finite-difference PDE solver for topological solitons in 2D nonlinear field theories, with Numba CUDA and CUDA–OpenGL rendering.
</h4>

<p align="center">
<a href="https://pypi.org/project/soliton_solver/">
  <img src="https://img.shields.io/pypi/v/soliton_solver.svg" alt="PyPI">
</a>
<a href="https://github.com/paulnleask/soliton_solver/releases">
  <img src="https://img.shields.io/github/v/release/paulnleask/soliton_solver?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/paulnleask/soliton_solver/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#built-in-models">Models</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#background">Background</a> •
  <a href="#license">License</a>
</p>

`soliton_solver` is a GPU-accelerated scientific computing framework for nonlinear
partial differential equations describing topological solitons in two-dimensional
field theories.

The solver is designed around a theory-agnostic numerical core implemented with
Numba CUDA kernels. Physical models are introduced as modular components using a
dependency injection (DI) architecture, allowing new theories to be added
without modifying the numerical engine.

The framework supports a wide range of models spanning condensed matter physics,
topological magnetism, and high-energy gauge field theories.

Real-time visualization is provided via CUDA–OpenGL interoperability, enabling
interactive exploration of nonlinear field dynamics directly on the GPU.

The project follows modern software engineering practices including continuous
integration via GitHub Actions and continuous deployment to PyPI.

---

### System requirements

Before installing `soliton_solver`, a few system-level dependencies must be
available. These provide the OpenGL context and CUDA bindings required for
GPU computation and real-time visualization. The following system software must be installed:

- **NVIDIA GPU with CUDA support**
- **CUDA Toolkit** compatible with your GPU
- **OpenGL drivers** (normally included with NVIDIA drivers)

You can verify CUDA installation with:

```bash
nvidia-smi
```

and

```bash
nvcc --version
```

Once these dependencies are available, the `soliton_solver` package itself can be installed.

## Installation

Install from PyPI (https://pypi.org/project/soliton-solver/):

```bash
pip install soliton-solver
```

Or install from source:

```bash
git clone https://github.com/paulnleask/soliton_solver.git
cd soliton_solver
pip install -e .
```

The `soliton_solver` package also installs the following dependencies:

- **glfw** – window creation and input handling for the OpenGL viewer  
- **PyOpenGL** – Python bindings for OpenGL rendering  
- **cuda-python** – low-level CUDA driver bindings used for CUDA–OpenGL interoperability  

**Requirements**

- Python 3.10+
- CUDA-capable GPU
- NVIDIA drivers compatible with Numba CUDA

---

## Quickstart

Run a built-in example:

```bash
python -m soliton_solver.examples.chiral_magnet_gl
```

Typical Python usage (including OpenGL visualization): Simulate magnetic skyrmions in chiral magnets with the DM interaction and demagnetization

```python
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
```

Simulations execute entirely on the GPU. Field data streams directly from
CUDA memory into OpenGL buffers using zero-copy CUDA–OpenGL interop.
The results can be plotted by running

```bash
python -m soliton_solver.theories.chiral_magnet.results.plotting
```

<p align="center">
  <img <img src="https://raw.githubusercontent.com/Paulnleask/soliton_solver/main/soliton_solver/assets/Plot_Densities.png" width="800">
</p>

---

## Currently Supported Built-In Theories

| Theory | Fields | Energy functional |
|--------|--------|--------|
| Abelian Higgs / Ginzburg-Landau | $\psi \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^2$ | $E_{\textup{AH}}[\psi, \vec{A}] = \int_{\mathbb{R}^2} \textup{d}^2x \left( \frac{1}{2}\|\vec{D}\psi\|^2 + \frac{1}{2}\|\vec{\nabla}\times\vec{A}\|^2 + \frac{\lambda}{8} \left( u^2 - \|\psi\|^2 \right)^2\right)$ |
| Anisotropic s+id Ginzburg-Landau | $\Delta_\alpha \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^2$ | $E_{s+id}[\Delta_s, \Delta_d,\vec{A}] = \int_{\mathbb{R}^2}\textup{d}^2x \left( \frac{1}{2} \gamma_{jk}^{\alpha\beta} (D_j \Delta_\alpha)^* D_k \Delta_\beta + \frac{1}{2}\|\vec{\nabla}\times\vec{A}\|^2 + \alpha_1 \|\Delta_s\|^2 + \alpha_2 \|\Delta_d\|^2 + \beta_1 \|\Delta_s\|^4 + \beta_2 \|\Delta_d\|^4 + \beta_3 \|\Delta_s\|^2 \|\Delta_d\|^2 + \beta_4 \left( \Delta_s^2 \bar{\Delta}_2^2 + \bar{\Delta}_1^2 \Delta_d^2 \right) \right)$ |
| Baby Skyrme | $\vec{m} \in \mathbb{R}^3$ | $E_{\textup{BS}}[\vec{m}] = \int_{\mathbb{R}^2} \textup{d}^2x \left( \frac{1}{2} \|\nabla \vec{m}\|^2 + \frac{\kappa^2}{4} \left( \partial_i \vec{m} \times \partial_j \vec{m} \right)^2+ V(\vec{m}) \right)$ |
| Bose-Einstein Condensate | $\Psi \in \mathbb{C}$ | $E_{\textup{BEC}}[\Psi] = \int_{\mathbb{R}^2} \textup{d}^2x \left( \frac{\hbar^2}{2m}\|\vec{\nabla} \Psi\|^2 + \frac{1}{2}m \omega^2 \|\vec{r}\|^2\|\Psi\|^2 + \frac{g}{2}\|\Psi\|^4 - \Omega \Psi^* \hat{L}_z \Psi \right), \quad \hat{L}_z = -i\hbar \left( x \partial_y - y \partial_x \right)$ |
| Chern-Simons-Landau-Ginzburg | $\psi \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^3$ | $E_{\textup{CSLG}}[\psi,\vec{A}] = E_{\textup{AH}}[\psi,\vec{A}] + \int_{\mathbb{R}^2} \textup{d}^2x  \left( \frac{1}{2}\|\vec{\nabla} A_0\|^2 + \frac{1}{2}q^2\|\psi\|^2 A_0^2 \right), \quad \left(\Delta+q^2\|\psi\|^2\right)A_0 = -\kappa B[\vec{A}]$ |
| Chiral Ferromagnet | $\vec{n} \in \mathbb{R}^3$, $\psi \in \mathbb{R}$ | $E_{\textup{CM}}[\vec{n}] = \int_{\mathbb{R}^2}\textup{d}^2x \left( \frac{J}{2}\|\textup{d}\vec{n}\|^2 + \mathcal{D} \sum_{i=1}^3 \vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) + M_sV(\vec{n}) + \frac{1}{2\mu_0}  \|\vec{\nabla}\psi\|^2 \right), \quad \Delta \psi = -\mu_0\vec{\nabla}\cdot(M_s\vec{n})$ |
| Chiral Liquid Crystal (Oseen-Frank) | $\vec{n} \in \mathbb{R}^3$, $\phi \in \mathbb{R}$ | $E_{\textup{LC}}[\vec{n}] = \int_{\mathbb{R}^2} \textup{d}^2x  \left( \frac{K}{2} \|\nabla \vec{n}\|^2 + K q_0 \vec{n}\cdot(\vec{\nabla} \times \vec{n}) + V(\vec{n}) + \frac{\varepsilon_0}{2} \|\vec{\nabla} \Phi\|^2 \right), \quad \Delta \Phi = -\frac{1}{\varepsilon_0}\vec{\nabla} \cdot \vec{P}[\vec{n}]$ |
| Ferromagnetic Superconductor | $\vec{m} \in \mathbb{R}^3$, $\psi \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^3$ | $E_{\textup{FS}}[\vec{m}, \psi, \vec{A}] = \int_{\mathbb{R}^2} \textup{d}^2x \left( \frac{1}{2}\|\vec{D}\psi\|^2 + \frac{1}{2}\|\vec{\nabla}\times\vec{A}\|^2 + \frac{\gamma^2}{2}\|\nabla \vec{m}\|^2 - \vec{m} \cdot (\vec{\nabla} \times \vec{A}) + \eta_1\|\vec{m}\|^2\|\psi\|^2 + \eta_2 \|\nabla\vec{m}\|^2\|\psi\|^2 + \frac{a}{2}\|\psi\|^2 + \frac{b}{4}\|\psi\|^4 + \frac{\alpha}{2}\|\vec{m}\|^2 + \frac{\beta}{4}\|\vec{m}\|^4 - \frac{\alpha^2 b + a^2\beta - 4a\alpha\eta_1}{16\eta_1^2-4b\beta} \right)$ |
| Spin-Triplet Superconducting Ferromagnet | $\vec{m} \in \mathbb{R}^3$, $\psi_1, \psi_2 \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^3$ | $E_{\textup{ST}}[\vec{m}, \psi_\alpha, \vec{A}] = \int_{\mathbb{R}^2} \textup{d}^2x \left( \frac{1}{2} \|\vec{D}\psi_\alpha\|^2 + \frac{1}{2}\|\vec{\nabla}\times\vec{A}\|^2 + \frac{a}{2} \|\psi_\alpha\|^2 + \frac{b_1}{4} \|\psi_\alpha\|^4  + b_2 \|\psi_1\|^2 \|\psi_2\|^2 + c \left( \psi_1 \psi_2^* + \psi_1^* \psi_2 \right) + \frac{\alpha}{2}\|\vec{m}\|^2 + \frac{\beta}{4}\|\vec{m}\|^4 + \frac{\gamma^2}{2}\|\nabla\vec{m}\|^2 - \vec{m} \cdot (\vec{\nabla}\times\vec{A}) +\frac{(a+2c)^2}{2(b_1+2b_2)} +\frac{\alpha^2}{4\beta} \right)$ |

All models are GPU-accelerated and compatible with real-time rendering. New theories can be added by implementing a theory module and registering it
with the **theory registry**.

---

## Features

- GPU-accelerated nonlinear PDE solver
- Numba CUDA kernels for high-performance finite-difference evolution
- Real-time visualization via CUDA–OpenGL interoperability
- Object-oriented simulation architecture
- Plug-and-play theory registry
- Theory-agnostic numerical core
- Dependency injection design
- Continuous integration via GitHub Actions
- Continuous deployment to PyPI

---

## Architecture

```text
soliton_solver/
├── core/            GPU numerical engine
├── theories/        modular physics models
├── visualization/   OpenGL rendering backend
├── examples/        runnable demonstrations
├── version.py
└── pyproject.toml
```

### Core

The numerical core implements the GPU PDE solver:

- finite-difference operators
- time-stepping integrators
- simulation driver
- GPU memory management

These components are independent of any specific physical theory.

### Theories

Each theory defines:

- energy functional
- field variables
- parameter set
- initialization routines
- optional visualization helpers

Theories are injected into the solver via **dependency injection**, allowing
the numerical infrastructure to remain completely **theory agnostic**.

---

## Numerical Method

Topological solitons are obtained by relaxing the field configuration toward a
local minimum of the discrete energy functional.
Given a discretized energy $E_h[\phi]$, we evolve the field using arrested Newton flow

$$
\ddot{\phi} = - \nabla_\phi E_h[\phi].
$$

The system is integrated numerically using explicit time-stepping schemes such
as Runge–Kutta methods implemented on the GPU.

A flow arrest condition ensures stability: If the energy increases,

$$
E_h[\phi^{n+1}] > E_h[\phi^{n}],
$$

the velocity is reset

$$
\dot{\phi}^{n+1} = 0.
$$

This second-order relaxation scheme accelerates convergence along shallow
directions of the energy landscape while preventing oscillations near the
minimum.

In practice this method converges significantly faster than gradient descent
for multi-soliton configurations.

---

## Research Doamins

Intended research domains include:

- Topological defects, vortices, and flux structures in superconductors, including multi-component Ginzburg-Landau models, ferromagnetic superconductors, and cosmic string analogues
- Anyons in a Chern-Simons-Landau-Ginzburg theory of the fractional quantum Hall effect
- Skyrmions, and other topological textures, in chiral magnets and liquid crystals (including the effects of demagnetization/depolarization)
- Mixed skyrmion-vortex states and coupled order parameter defects in ferromagnetic superconductors
- Vortices in rotating Bose-Einstein condensates

---

## License

MIT License. See `LICENSE` for details.
