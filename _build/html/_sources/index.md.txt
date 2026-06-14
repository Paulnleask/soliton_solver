# $\texttt{soliton_solver}$ documentation

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

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
theories
numerical_solver
gpu_acceleration
visualization
architecture
extending
api
```