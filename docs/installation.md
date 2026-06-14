# Installation

This page contains detailed instructions for installing `soliton_solver` and verifying your setup.

## System Requirements

Before installing `soliton_solver`, the following system-level dependencies must be available. These provide the OpenGL context and CUDA bindings required for GPU computation and real-time visualization.

### Required software

- **NVIDIA GPU with CUDA support** — An NVIDIA graphics card with compute capability 3.0 or higher
- **CUDA Toolkit** — Compatible with your GPU and operating system ([download from NVIDIA](https://developer.nvidia.com/cuda-downloads))
- **OpenGL drivers** — Normally included with NVIDIA drivers; verify with your system

### Verify system setup

Check NVIDIA driver installation:

```bash
nvidia-smi
```

This should display your GPU information. Check CUDA Toolkit installation:

```bash
nvcc --version
```

This should display the CUDA compiler version. If either command fails, install the missing software before proceeding.

## Python Requirements

- **Python 3.10 or later**
- pip or conda package manager

## Installation Methods

### From PyPI (recommended)

The simplest way to install `soliton_solver`:

```bash
pip install soliton-solver
```

### From source

For development or to use the latest code:

```bash
git clone https://github.com/paulnleask/soliton_solver.git
cd soliton_solver
pip install -e .
```

The `-e` flag installs the package in editable mode, allowing you to modify the code and see changes immediately.

### With documentation dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

This installs the package along with Sphinx, sphinx-book-theme, and related documentation tools.

## Dependencies

The `soliton_solver` package automatically installs the following Python dependencies:

- **numpy** — Array operations and numerical computing
- **numba-cuda** — JIT compilation for CUDA kernels
- **moderngl** — Modern OpenGL rendering abstraction
- **glfw** — Window creation and input handling for the OpenGL viewer
- **PyOpenGL** — Python bindings for OpenGL rendering
- **cuda-python** — Low-level CUDA driver bindings used for CUDA–OpenGL interoperability

## Verify Installation

Test that the installation was successful:

```python
import soliton_solver
print(soliton_solver.__version__)
```

Try running a built-in example:

```bash
python -m soliton_solver.examples.chiral_magnet_gl
```

This will launch an interactive visualization of magnetic skyrmions. You should see a real-time simulation window with a colormap visualization of the magnetic field.

## Troubleshooting

### CUDA not detected

If you get errors related to CUDA not being found:
- Ensure NVIDIA drivers are installed (`nvidia-smi` should work)
- Ensure CUDA Toolkit is installed (`nvcc --version` should work)
- Check that your GPU has CUDA support (compute capability 3.0+)

### OpenGL errors

If you get OpenGL-related errors:
- Ensure your NVIDIA drivers are up to date
- Verify OpenGL support with `glxinfo` (Linux) or check System Information (Windows/Mac)
- Ensure you have a display attached (SSH connections may require X11 forwarding)

### Import errors

If `import soliton_solver` fails:
- Verify the package installed with `pip list | grep soliton`
- Try reinstalling with `pip install --upgrade --force-reinstall soliton-solver`

For additional help, please open an issue on the [GitHub repository](https://github.com/paulnleask/soliton_solver/issues).

