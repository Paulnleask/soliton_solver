<p align="center">
  <img src="docs/imgs/logo.svg" alt="soliton_solver logo" width="300">
</p>

<h4 align="center">
Extremely user-friendly GPU soliton simulations with real-time interactivity
via Numba CUDA and CUDA–OpenGL interoperability.
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

<p align="center">
  <img src="soliton_solver/assets/soliton_solver_logo.png" width="300">
</p>

`soliton_solver` is a modern scientific computing framework for nonlinear field
theories with solitonic structure. It targets GPU-first workflows: Numba CUDA
kernels for evolution, and CUDA–OpenGL interoperability for low-latency,
zero-copy visualization. The API is object-oriented, dependency-injection
friendly, and designed for rapid experimentation.

---

## Installation

Install from PyPI:

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

Minimal Python usage:

```python
from soliton_solver.core.simulation import Simulation
from soliton_solver.theories.registry import get_theory

theory = get_theory("baby_skyrme")
sim = Simulation(theory=theory, params=params)
sim.run()
```

Simulations execute entirely on the GPU. Field data streams directly from
CUDA memory into OpenGL buffers using zero-copy CUDA–OpenGL interop.

---

## Built-in Models

### Gauge / GL-type Systems

| Model | Fields | Gauge | Dimension |
|--------|--------|--------|------------|
| Abelian Higgs (Ginzburg–Landau) | Complex scalar | U(1) | 2D |
| Maxwell–Chern–Simons–Higgs | Complex scalar | U(1) | 2D |
| Ferromagnetic Superconductor | Multi-component | Yes | 2D |
| Spin-Triplet Superconducting Magnet | Multi-component | Yes | 2D |

### Topological Soliton Models

| Model | Fields | Dimension |
|--------|--------|------------|
| Baby Skyrme | O(3) vector | 2D |
| Chiral Magnet | Magnetization vector | 2D |
| Liquid Crystal (GL-type) | Tensor / vector | 2D |

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

`soliton_solver` targets classical nonlinear field theories of the form

\[
\partial_t \Phi = \mathcal{F}(\Phi, \nabla \Phi, A, \nabla A)
\]

including scalar, vector, and gauge fields. The framework emphasizes:

- Explicit finite-difference discretizations
- Direct GPU kernel implementations
- Real-time visual exploration of dynamics
- Interactive parameter steering

Intended research domains include:

- Gauge-field dynamics
- Topological defects and vortices
- Skyrmion systems
- Multi-component Ginzburg–Landau models
- Competing order parameter systems

---

## License

MIT License. See `LICENSE` for details.