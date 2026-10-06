# Validation

This page documents numerical results for the representative models used as
baselines in the paper. It is not an exhaustive verification of every theory
supported by `soliton_solver`; models not included here have not been
individually validated in this section. The tables report grid-refinement and
benchmark comparisons for the selected cases.

The example scripts define the model parameters and initial conditions for
these baseline runs. Their numerical results are then compared across grid
resolutions and, where available, with published benchmarks. These checks
demonstrate the solver's behavior on the selected paper cases, but should not
be interpreted as validation of all model implementations or parameter
regimes.

## Paper baseline models

The model parameters and initial states are those specified by the cited
example scripts, with the grid varied for the refinement calculations. The
listed $N_x$ and $N_y$ are the full grid dimensions, including halo points;
the domain lengths are $L_x$ and $L_y$. Unless overridden in a model
configuration, the shared solver settings are `halo=2`, `courant=0.5`,
`killkinen=True`, and `time_step=None`. The lattice spacings are calculated as
$h_x=L_x/(N_x-1)$ and $h_y=L_y/(N_y-1)$. The tables report the discrete energy
and, for models with a topological invariant, the corresponding topological
charge. Energies are given in the normalization of each model and compared
with the cited benchmark where available.

### Baby Skyrme model

Source: `soliton_solver/examples/baby_skyrme_gl.py`.

| Category | Baseline setting |
|---|---|
| Domain | `xsize=20.0`, `ysize=20.0` |
| Ansatz / constraint | `ansatz="neel"`, `unit_magnetization=True` |
| Initialization | `mode="ground"` |
| Run command | `python -m soliton_solver.examples.baby_skyrme_gl` |

The standard and aloof calculations use the same domain, ansatz, constraint,
and grid-refinement sequence, with the potential choice varied between runs.
The broken-potential calculation uses `N=3` and its separately listed model
parameters. Within each potential's refinement sequence, the physical domain
and model parameters are held fixed.

#### Standard potential

| Category | Baseline setting |
|---|---|
| Model | `mpi=0.3162`, `kappa=1.0`, `potential="standard"` |

| Grid Size $N_x$ | Grid Size $N_y$ | Grid Spacing $h_x$ | Grid Spacing $h_y$ | Topological Charge $\mathcal{Q}$ | Energy $E$ |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 0.15748 | 0.15748 | -1.0000 | 19.6781 |
| 256 | 256 | 0.07843 | 0.07843 | -1.0000 | 19.6778 |
| 512 | 512 | 0.03914 | 0.03914 | -1.0000 | 19.6778 |

- **Reference:** The benchmark is taken from [D. Foster, *Baby Skyrmion chains*, [Nonlinearity **23**, 465 (2010)](https://doi.org/10.1088/0951-7715/23/3/001)].
- **Benchmark:** Foster reports a standard-potential energy of $E=19.65$; the computed 512-by-512 energy is $E=19.6778$, a difference of $0.0278$ (about $0.14\%$).
- **Conclusion:** The energy is unchanged to four decimal places between the 256-by-256 and 512-by-512 grids, and the topological charge remains $\mathcal{Q}=-1$. This indicates grid convergence at these resolutions, while the finest-grid energy is about $0.14\%$ above the cited value.

#### Aloof potential

| Category | Baseline setting |
|---|---|
| Model | `mpi=0.3162`, `kappa=1.0`, `potential="Aloof"` |

| Grid Size $N_x$ | Grid Size $N_y$ | Grid Spacing $h_x$ | Grid Spacing $h_y$ | Topological Charge $\mathcal{Q}$ | Energy $E$ |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 0.15748 | 0.15748 | -0.9999 | 20.3163 |
| 256 | 256 | 0.07843 | 0.07843 | -1.0000 | 20.3161 |
| 512 | 512 | 0.03914 | 0.03914 | -1.0000 | 20.3161 |

- **Reference:** The benchmark is taken from [P. Salmi and P. Sutcliffe, *Aloof baby Skyrmions*, [J. Phys. A **48**, 035401 (2015)](https://doi.org/10.1088/1751-8113/48/3/035401)].
- **Benchmark:** Salmi and Sutcliffe report an energy of $E=20.27$; the computed 512-by-512 energy is $E=20.3161$, a difference of $0.0461$ (about $0.23\%$).
- **Conclusion:** The energy changes by only $0.0002$ from the 128-by-128 grid to the 256-by-256 grid and is unchanged to four decimal places at 512-by-512; the topological charge converges to $\mathcal{Q}=-1$. The finest-grid energy is about $0.23\%$ above the cited value.

#### Broken potential

| Category | Baseline setting |
|---|---|
| Model | `mpi=1.0`, `kappa=1.0` , `N=3`, `potential="Broken"` |

| Grid Size $N_x$ | Grid Size $N_y$ | Grid Spacing $h_x$ | Grid Spacing $h_y$ | Topological Charge $\mathcal{Q}$ | Energy $E$ |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 0.15748 | 0.15748 | -0.9994 | 34.7516 |
| 256 | 256 | 0.07843 | 0.07843 | -1.0000 | 34.7843 |
| 512 | 512 | 0.03914 | 0.03914 | -1.0000 | 34.7864 |

- **Reference:** The benchmark is taken from [J. Jäykkä, M. Speight, and P. Sutcliffe, *Broken baby Skyrmions*, [Proc. R. Soc. A. **468**, 1085 (2012)](https://doi.org/10.1098/rspa.2011.0543)].
- **Benchmark:** Jäykkä, Speight, and Sutcliffe report an energy of $E=34.79$; the computed 512-by-512 energy is $E=34.7864$, a difference of $-0.0036$ (about $0.010\%$ below the reference).
- **Conclusion:** The topological charge converges to $\mathcal{Q}=-1$, and the energy approaches the cited value as the grid is refined. Between 256-by-256 and 512-by-512, the energy changes by $0.0021$; further refinement would help establish convergence of the energy.

### Bose-Einstein condensate

Source: `soliton_solver/examples/bose_einstein_condensate_gl.py`.

| Category | Baseline setting |
|---|---|
| Physical parameters | `beta=1000.0` |
| Domain | `xsize=24.0`, `ysize=24.0` |
| Solver | `time_step=0.006` |
| Initialization | `mode="ground"` |
| Run command | `python -m soliton_solver.examples.bose_einstein_condensate_gl` |

#### Slowly rotating BEC

| Category | Baseline setting |
|---|---|
| Physical parameters | `omega_rot=0.5` |

| Grid Size ($N_x$) | Grid Size ($N_y$) | Grid Spacing ($h_x$) | Grid Spacing ($h_y$) | Energy ($E$) |
|---:|---:|---:|---:|---:|
| 128 | 128 | 0.18898 | 0.18898 | 11.1000 |
| 256 | 256 | 0.09412 | 0.09412 | 11.1052 |
| 512 | 512 | 0.04697 | 0.04697 | 11.1054 |

- **Reference:** H. Chena, G. Dongb, W. Liuc, and Z. Xie, *Second-order flows for computing the ground states of rotating Bose-Einstein condensates*, [J. Comput. Phys. **475**, 111872 (2023)](https://doi.org/10.1016/j.jcp.2022.111872).
- **Benchmark:** For $\beta=1000$ and $\Omega=0.5$, the reference energy is $E=11.0954$. The computed 512-by-512 energy is $E=11.1054$, which is $0.0100$ (about $0.0901\%$) above the reference.
- **Conclusion:** The energy changes by $0.0002$ between the 256-by-256 and 512-by-512 grids. The finest-grid energy is about $0.0901\%$ above the cited benchmark.

#### Rapidly rotating BEC

| Category | Baseline setting |
|---|---|
| Physical parameters | `omega_rot=0.9` |

| Grid Size ($N_x$) | Grid Size ($N_y$) | Grid Spacing ($h_x$) | Grid Spacing ($h_y$) | Energy ($E$) |
|---:|---:|---:|---:|---:|
| 128 | 128 | 0.18898 | 0.18898 | 6.4873 |
| 256 | 256 | 0.09412 | 0.09412 | 6.3649 |
| 512 | 512 | 0.04697 | 0.04697 | 6.3651 |

- **Reference:** The benchmark is taken from [H. Chena, G. Dongb, W. Liuc, and Z. Xie, *Second-order flows for computing the ground states of rotating Bose-Einstein condensates*, [J. Comput. Phys. **475**, 111872 (2023)](https://doi.org/10.1016/j.jcp.2022.111872)].
- **Benchmark:** For $\beta=1000$ and $\Omega=0.9$, the reference energy is $E=6.3601$; the computed 512-by-512 result is $E=6.3651$, a difference of $0.0050$ (about $0.079\%$).
- **Conclusion:** The computed energy changes by only $0.0002$ between the 256-by-256 and 512-by-512 grids, indicating grid convergence at these resolutions, although the finest-grid result remains about $0.079\%$ above the reference value.