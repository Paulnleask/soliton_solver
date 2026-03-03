<p align="center">
  <img src="soliton_solver/assets/soliton_solver_logo.png" width="300">
</p>

<h4 align="center">
GPU-based finite-difference PDE solver for topological solitons in 2D nonlinear field theories, with Numba CUDA and CUDA–OpenGL rendering.
</h4>

<p align="center">
<a href="https://pypi.org/project/soliton-solver/">
  <img src="https://img.shields.io/pypi/v/soliton-solver.svg" alt="PyPI">
</a>
<a href="https://github.com/yourusername/soliton_solver/actions/workflows/tests.yml">
  <img src="https://github.com/yourusername/soliton_solver/actions/workflows/tests.yml/badge.svg" alt="Tests">
</a>
<a href="https://github.com/yourusername/soliton_solver/releases">
  <img src="https://img.shields.io/github/v/release/yourusername/soliton_solver?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/yourusername/soliton_solver/blob/main/LICENSE">
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

`soliton_solver` is a modern scientific computing framework for nonlinear field
theories with topological solitons. It targets GPU-first workflows: Numba CUDA
kernels for evolution, and CUDA–OpenGL interoperability for low-latency,
zero-copy visualization. The API is object-oriented, dependency-injection
friendly, and designed for rapid experimentation.

---

## Installation

Install from PyPI (in progress, not currently implemented):

```bash
pip install soliton-solver
```

Or install from source:

```bash
git clone https://github.com/yourusername/soliton_solver.git
cd soliton_solver
pip install -e .
```

**Requirements**

- Python 3.10+
- CUDA-capable GPU
- NVIDIA drivers compatible with Numba CUDA

---

## Quickstart

Run a built-in example:

```bash
python -m soliton_solver.examples.abelian_higgs_gl
```

Typical Python usage (including OpenGL visualization) for an example theory:

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

theory = load_theory("Ginzburg-Landau superconductor")

def run_gl_simulation():
    params = theory.params.default_params(
        xlen=320, ylen=320, xsize=50.0, ysize=50.0,     # Grid params 
        q=1.0, Lambda=1.0, u1=1.0,                      # Abelian Higgs parameters
        newtonflow=False,                               # Start simulation with flow on/off (True/False)
        unit_magnetization=False                        # Required flag for magnetization (False if no magnetization)
        )
    sim = Simulation(params, theory)
    sim.initialize({"mode": "ground"})
    theory.render_gl.run_viewer(sim, sim.rp, steps_per_frame=5)

if __name__ == "__main__":
    run_gl_simulation()
```

Simulations execute entirely on the GPU. Field data streams directly from
CUDA memory into OpenGL buffers using zero-copy CUDA–OpenGL interop.

---

## Currently supported built-in models

| Model | Fields | Energy functional |
|--------|--------|--------|
| Abelian Higgs (Ginzburg-Landau) | $\psi \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^2$ | $E_{\textup{AH}}[\psi, \vec{A}] = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2}\|\vec{D}\psi\|^2 + \frac{1}{2}\|\vec{\nabla}\times\vec{A}\|^2 + \frac{\lambda}{8} \left( u^2 - \|\psi\|^2 \right)^2\right\}$ |
| Baby Skyrme | $\vec{m} \in \mathbb{R}^3$ | $E[\vec{m}] = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} \|\nabla \vec{m}\|^2 + \frac{\kappa^2}{4} \left( \partial_i \vec{m} \times \partial_j \vec{m} \right)^2+ V(\vec{m}) \right\}$ |
| Chiral Magnet | $\vec{n} \in \mathbb{R}^3$, $\psi \in \mathbb{R}$ | $E_{\textup{CM}}[\vec{n}] = \int_{\mathbb{R}^2}\textup{d}^2x \left\{ \frac{J}{2}\|\textup{d}\vec{n}\|^2 + \mathcal{D} \sum_{i=1}^3 \vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) + M_sV(\vec{n}) + \frac{1}{2\mu_0}  \|\bm{\nabla}\psi\|^2 \right\}, \quad \Delta \psi = -\mu_0\,\bm{\nabla}\cdot(M_s\vec{n})$ |
| Ferromagnetic Superconductor | $\vec{m} \in \mathbb{R}^3$, $\psi \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^3$ | $E_{\textup{FS}}[\vec{m}, \psi, \vec{A}] = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2}\|\vec{D}\psi\|^2 + \frac{1}{2}\|\vec{\nabla}\times\vec{A}\|^2 + \frac{b}{4} \left( u^2 - \|\psi\|^2 \right)^2 + \frac{1}{2}\|\nabla\vec{m}\|^2 - \vec{m}\cdot(\vec{\nabla}\times\vec{A})\right\}$ |
| Liquid Crystal (GL-type) | $\vec{n} \in \mathbb{R}^3$, $\phi \in \mathbb{R}$ | $E_{\textup{LC}}[\vec{n}] = \int_{\mathbb{R}^2} \textup{d}^2x \, \left\{ \frac{K}{2} \|\nabla \vec{n}\|^2 + K q_0 \vec{n}\cdot(\bm{\nabla} \times \vec{n}) + V(\vec{n}) + \frac{\varepsilon_0}{2} \|\bm{\nabla} \Phi\|^2 \right\}, \quad \Delta \Phi = -\frac{1}{\varepsilon_0}\bm{\nabla} \cdot \vec{P}[\vec{n}]$ |
| Maxwell-Chern-Simons-Higgs | $\psi \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^3$ | $E_{\textup{CSLG}}[\psi,\vec{A}] = E_{\textup{AH}}[\psi,\vec{A}] + \int_{\mathbb{R}^2} \textup{d}^2x \, \left\{ \frac{1}{2}\|\bm{\nabla} A_0\|^2 + \frac{1}{2}q^2\|\psi\|^2 A_0^2 \right\}, \quad \left(-\nabla^2+q^2\|\psi\|^2\right)A_0 = -\kappa B[\vec{A}]$ |
| Spin-Triplet Superconducting Magnet | $\vec{m} \in \mathbb{R}^3$, $\psi_1, \psi_2 \in \mathbb{C}$, $\vec{A} \in \mathbb{R}^3$ | TBD |

All models are GPU-accelerated and compatible with real-time rendering.

---

## Features

- Numba CUDA GPU kernels
- CUDA–OpenGL zero-copy rendering
- Real-time visualization and interactive steering
- Object-oriented simulation components
- Dependency injection architecture
- Clear separation of physics and numerics
- Pluggable theory modules
- Minimal boilerplate API

---

## Architecture

```text
soliton_solver/
├── core/            numerical engine (GPU kernels, integrators)
├── theories/        pluggable physics modules
├── visualization/   OpenGL backend (CUDA interop)
├── examples/        runnable scripts
├── version.py
└── pyproject.toml
```

### Core

- Finite-difference operators (GPU)
- Time integration schemes
- Simulation driver
- Parameter management
- IO utilities

Physics-agnostic and reusable.

### Theories

Each theory provides:
- Parameter definitions
- Initial configuration
- Evolution kernels (Numba CUDA)
- Observables
- Optional rendering hooks

Components are composed via dependency injection. The numerical engine
remains independent of specific physical models.

---

## Background

To compute topological solitons using `soliton_solver`, it relaxes the field $\phi$ toward a local
minimizer of the discrete energy $E_h[\phi]$, equivalently a stationary point
of the nonlinear field equations.

A simple gradient descent in $\phi$ is robust but often inefficient. Soliton
energy landscapes are typically stiff: short-wavelength modes relax rapidly,
while long-wavelength modes and collective coordinates (such as soliton
separation or internal phases) relax slowly. This leads to very slow
convergence for multi-soliton configurations.

To accelerate minimization, we use **arrested Newton flow**. This introduces a
fictitious second-order dynamics in an artificial time variable $t$:
$$
\ddot{\phi}(t) = - \nabla_\phi E_h[\phi(t)].
$$
The initial condition is taken from rest,
$\dot{\phi}(0) = 0$, with $\phi(0)$ chosen to encode the desired topology.
The system is rewritten as a first-order system in $(\phi, \dot{\phi})$ and
integrated numerically using a standard explicit scheme, e.g. fourth-order
Runge–Kutta with time step $\delta t$.

The key feature is the **flow arrest condition**. After each time step we
compare the discrete energies:
$$
E_h[\phi^{n+1}] \quad \text{and} \quad E_h[\phi^{n}].
$$
If the energy increases,
$$
E_h[\phi^{n+1}] > E_h[\phi^{n}],
$$
the velocity is reset to zero:
$$
\dot{\phi}^{\,n+1} \leftarrow 0.
$$

Intuitively, the second-order dynamics accelerates motion along shallow
directions of the energy landscape, while the arrest criterion prevents
overshooting and oscillations near a minimum. In practice, this approach
converges significantly faster than first-order gradient descent for
multi-soliton relaxation problems.

The framework emphasizes:

- Explicit finite-difference discretizations
- Direct GPU kernel implementations
- Real-time visual exploration of dynamics
- Interactive parameter steering

Intended research domains include:

- Topological defects and vortices
- Skyrmion systems
- Multi-component Ginzburg–Landau models
- Competing order parameter systems

---

## License

MIT License. See `LICENSE` for details.
