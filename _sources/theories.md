# Supported Theories

Complete documentation of all supported physical theories in `soliton_solver`. Each theory includes its governing equations, field variables, parameters, and built-in initial configurations.

## Overview

The `soliton_solver` framework provides GPU-accelerated simulations of topological solitons across 10 distinct physics theories spanning condensed matter, magnetism, and gauge field theory:

| Theory | Fields | Canonical Name |
|--------|--------|---|
| Abelian Higgs / Ginzburg-Landau | Complex scalar + gauge field | "Ginzburg-Landau superconductor" |
| Anisotropic s+id Ginzburg-Landau | Two complex scalars + gauge field | "Anisotropic superconductor" |
| Baby Skyrme | Three-component vector | "Baby Skyrme model" |
| Bose-Einstein Condensate | Complex scalar + rotation | "Bose-Einstein condensate" |
| Chern-Simons-Landau-Ginzburg | Complex scalar + 3D gauge field | "Maxwell-Chern-Simons-Higgs" |
| Chiral Ferromagnet | Three-component magnetization | "Chiral magnet" |
| Chiral Liquid Crystal | Three-component director + potential | "Liquid crystal" |
| Ferromagnetic Superconductor | Magnetization + complex scalar + gauge field | "Ferromagnetic superconductor" |
| Spin-Triplet Superconducting Magnet | Magnetization + two complex scalars + gauge field | "Spin-triplet superconducting magnet" |

Load a theory using its canonical name or any alias:

```python
from soliton_solver.theories import load_theory

theory = load_theory("Baby Skyrme")  # Canonical name
theory = load_theory("Planar skyrmion")  # Alias (equivalent)
```

## Ginzburg-Landau Superconductor (Abelian Higgs)

**Canonical name:** `"Ginzburg-Landau superconductor"`  
**Aliases:** `"Abelian Higgs"`, `"Ginzburg-Landau"`, `"Abelian Higgs model"`

### Fields

- $\psi \in \mathbb{C}$ — Complex scalar order parameter (superconducting condensate wavefunction)
- $\vec{A} = (A_x, A_y) \in \mathbb{R}^2$ — In-plane gauge field (vector potential)

### Energy Functional

$$E_{\text{AH}}[\psi, \vec{A}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{1}{2}\|\vec{D}\psi\|^2 + \frac{1}{2}\|\vec{\nabla} \times \vec{A}\|^2 + \frac{\lambda}{8} \left( u^2 - |\psi|^2 \right)^2 \right]$$

where the covariant derivative is $\vec{D}\psi = (\nabla - i\vec{A})\psi$ and $u$ is the characteristic condensate amplitude.

### Key Parameters

- `u` — Characteristic order parameter amplitude (default: 1.0)
- `lambda` — Higgs potential coupling strength (default: 1.0)
- `kappa` — Ginzburg-Landau parameter $\kappa = \lambda / (2 e^2)$

### Physics

The Ginzburg-Landau model describes superconductivity via a complex scalar condensate with **U(1) gauge symmetry** (electromagnetism). The energy has two competing terms:

- **Kinetic energy:** $\frac{1}{2}|\nabla\psi|^2$ tends to spread the condensate
- **Potential energy:** $\frac{\lambda}{8}(u^2 - |\psi|^2)^2$ drives the condensate to vacuum value $|\psi| = u$
- **Gauge field:** Minimizes magnetic energy subject to Gauss law

**Type I vs. Type II:** The model transitions from Type I (vortices repel) to Type II (vortices attract) depending on $\kappa$. At $\kappa = 1/\sqrt{2}$, the transition occurs.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Vacuum state $\psi = u$, $\vec{A} = 0$
- Default or any other mode — Vortex configuration with winding number parameter `vortex_number` (default: 1.0)

### Observable Quantities

- **Energy:** $E = \int [|\nabla\psi|^2 + |\nabla \times \vec{A}|^2 + \lambda(u^2 - |\psi|^2)^2]$
- **Topological charge** (vorticity): $Q = \frac{1}{2\pi} \oint \nabla\arg(\psi) \cdot d\vec{\ell}$
- **Condensate density:** $|\psi|^2$
- **Magnetic field:** $\nabla \times \vec{A}$

### References

See [Extending the Solver](extending.md) for details on implementing similar theories.

---

## Anisotropic Superconductor (s+id Ginzburg-Landau)

**Canonical name:** `"Anisotropic superconductor"`

### Fields

- $\Delta_s \in \mathbb{C}$ — s-wave pairing condensate
- $\Delta_d \in \mathbb{C}$ — d-wave pairing condensate
- $\vec{A} = (A_x, A_y) \in \mathbb{R}^2$ — Gauge field

### Energy Functional

$$E_{s+id}[\Delta_s, \Delta_d, \vec{A}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{1}{2} \gamma_{jk}^{\alpha\beta} (D_j \Delta_\alpha)^* D_k \Delta_\beta + \frac{1}{2}|\nabla \times \vec{A}|^2 + \alpha_1 |\Delta_s|^2 + \alpha_2 |\Delta_d|^2 + \beta_1 |\Delta_s|^4 + \beta_2 |\Delta_d|^4 + \beta_3 |\Delta_s|^2 |\Delta_d|^2 + \beta_4 (\Delta_s^2 \bar{\Delta}_d^2 + \text{c.c.}) \right]$$

where $\gamma_{jk}^{\alpha\beta}$ includes crystal anisotropy.

### Key Parameters

- `alpha_s`, `alpha_d` — Linear (quadratic) temperature coefficients for s and d channels
- `beta_s`, `beta_d` — Quartic coupling strengths
- `beta_sd` — Inter-channel coupling
- Anisotropy parameters for $\gamma_{jk}$ tensor

### Physics

The two-channel superconductor exhibits **competing s-wave and d-wave pairing symmetries**. The interplay between channels creates novel defect structures. The $\beta_4$ term couples $s$-wave pairs to $d$-wave pairs, creating mixed-parity or "hidden order" states in certain parameter regimes.

**Fractional flux quantization:** Vortices can bind fractional flux quanta due to the two-component order parameter.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Lowest-energy s+d coexistence state
- Default or any other mode — Mixed s+d vortex configuration

### Observable Quantities

- **Energy components:** $E_s$, $E_d$, $E_{\text{coupling}}$
- **Order parameters:** $|\Delta_s|$, $|\Delta_d|$
- **Topological charge:** Vorticity
- **Energy density maps:** Theory-specific visualization

---

## Baby Skyrme Model

**Canonical name:** `"Baby Skyrme model"`  
**Aliases:** `"Baby Skyrme"`, `"Baby skyrmion"`, `"Planar skyrmion"`

### Fields

- $\vec{m} \in \mathbb{R}^3$ — Three-component unit vector (normalized to $|\vec{m}| = 1$)

Constraint: $\vec{m} \cdot \vec{m} = 1$ (enforced during evolution)

### Energy Functional

$$E_{\text{BS}}[\vec{m}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{1}{2} |\nabla \vec{m}|^2 + \frac{\kappa^2}{4} (\partial_x \vec{m} \times \partial_y \vec{m})^2 + V(\vec{m}) \right]$$

where $V(\vec{m})$ is a potential (e.g., magnetic anisotropy).

### Key Parameters

- `kappa` — Skyrme number coupling strength (controls skyrmion size and shape)
- `potential_type` — Choice of potential: `"standard"`, `"easy plane"`, etc.
- Potential coefficients: `A`, `B` (depends on type)

### Physics

The Baby Skyrme model is the 2D reduction of the 3D Skyrme model. The competition between:

- **Kinetic energy (gradient):** Minimized by uniform fields
- **Skyrme term:** $\frac{\kappa^2}{4}(\partial_x \vec{m} \times \partial_y \vec{m})^2$ stabilizes extended skyrmions
- **Potential:** Determines easy/hard directions

produces **topological skyrmion solutions** with integer winding number. The energy has a **topological lower bound** $E \geq 4\pi|Q|$ (the Bogomolny bound) achieved by BPS skyrmions.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Ferromagnetic state $\vec{m} = (0, 0, 1)$
- Default or any other mode — Skyrmion configuration with `ansatz` parameter:
  - `"bloch"` (default) — Bloch-type skyrmion
  - `"neel"` — Néel-type skyrmion
  - `"anti"` — Anti-skyrmion
  - Additional parameter: `skyrmion_rotation` (default: 0.0)

### Observable Quantities

- **Energy:** $E = \int [|\nabla \vec{m}|^2 + \kappa^2 (\partial_x \vec{m} \times \partial_y \vec{m})^2 + V]$
- **Topological charge (winding number):** $Q = \frac{1}{4\pi} \int d^2x (\partial_x \vec{m} \times \partial_y \vec{m}) \cdot \vec{m}$
- **Magnetization:** Local direction of $\vec{m}$
- **Energy density:** Reveals skyrmion core structure

---

## Bose-Einstein Condensate (BEC)

**Canonical name:** `"Bose-Einstein condensate"`

### Fields

- $\Psi \in \mathbb{C}$ — Complex order parameter (condensate wavefunction)

### Energy Functional

$$E_{\text{BEC}}[\Psi] = \int_{\mathbb{R}^2} d^2x \left[ \frac{\hbar^2}{2m} |\nabla \Psi|^2 + \frac{1}{2}m\omega^2 r^2 |\Psi|^2 + \frac{g}{2}|\Psi|^4 - \Omega \Psi^* \hat{L}_z \Psi \right]$$

where $\hat{L}_z = -i\hbar(x \partial_y - y \partial_x)$ is the angular momentum operator and $\Omega$ is the rotation frequency.

### Key Parameters

- `hbar` — Reduced Planck constant (default: 1.0)
- `mass` — Particle mass
- `omega_trap` — Harmonic trap frequency
- `g_interaction` — s-wave scattering length (interaction strength)
- `rotation_freq` — Rotational frequency $\Omega$ (relative to trap)

### Physics

The BEC model describes ultracold atoms in a rotating harmonic potential. The competing effects are:

- **Kinetic energy:** Quantum pressure $\frac{\hbar^2}{2m}|\nabla\Psi|^2$
- **Trap potential:** Confines atoms to harmonic well
- **s-wave interactions:** Either repulsive ($g > 0$) or attractive ($g < 0$)
- **Rotation:** Creates centrifugal effects and vortices

In rapidly rotating traps, vortex lattices form (quantum Hall-like states). The winding number of vortices quantizes angular momentum.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Thomas-Fermi ground state
- Default or any other mode — Vortex configuration with winding number parameter `vortex_number` (default: 1.0)

### Observable Quantities

- **Energy:** Total energy
- **Particle number:** $N = \int |\Psi|^2 d^2x$
- **Vorticity:** Topological charge
- **Density profile:** $|\Psi|^2$
- **Angular momentum:** $\langle L_z \rangle$

---

## Chiral Magnet (Chiral Ferromagnet)

**Canonical name:** `"Chiral magnet"`  
**Aliases:** `"Chiral ferromagnet"`

### Fields

- $\vec{n} \in \mathbb{R}^3$ — Three-component magnetization unit vector ($|\vec{n}| = 1$)
- $\psi \in \mathbb{R}$ — Scalar auxiliary field (demagnetization potential)

### Energy Functional

$$E_{\text{CM}}[\vec{n}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{J}{2}|\nabla\vec{n}|^2 + \mathcal{D} \sum_{i=1}^2 \vec{d}_i \cdot (\vec{n} \times \partial_i \vec{n}) + M_s V(\vec{n}) + \frac{1}{2\mu_0}|\nabla\psi|^2 \right]$$

where the Dzyaloshinskii-Moriya (DMI) term breaks mirror symmetry via $\vec{d}_i = (D_i^{(1)}, D_i^{(2)}, D_i^{(3)})$.

### Key Parameters

**Magnetic parameters:**
- `J` — Exchange stiffness
- `D` — DMI strength
- `K` — Magnetic anisotropy
- `M` — Saturation magnetization
- `B` — Applied field magnitude

**DMI choice:**
- `dmi_term` — `"Dresselhaus"`, `"Rashba"`, `"Heusler"`, or `"hybrid"`
- Different DMI terms produce different skyrmion orientations

**Other:**
- `demag` (bool) — Include demagnetization energy
- `mu0` — Vacuum permeability

### Physics

The chiral magnet features **Dzyaloshinskii-Moriya (DMI) interactions** that stabilize topological spin textures (skyrmions) even without easy-axis anisotropy. The DMI breaks parity and chirality, inducing **helical or skyrmion ground states** depending on field and anisotropy.

The anisotropy direction determines skyrmion shape:
- **Easy-axis** → Néel-type skyrmions (radial spin arrangement)
- **In-plane** → Bloch-type skyrmions (tangential spin arrangement)

Demagnetization couples magnetization to long-range dipole fields via a Poisson equation.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Ferromagnetic state
- Default or any other mode — Skyrmion configuration with `ansatz` parameter:
  - `"bloch"` (default) — Bloch-type skyrmion (tangential spin arrangement)
  - `"neel"` — Néel-type skyrmion (radial spin arrangement)
  - `"anti"` — Anti-skyrmion
  - Additional parameter: `skyrmion_rotation` (default: 0.0)

### Observable Quantities

- **Energy components:** Exchange, DMI, anisotropy, Zeeman, demagnetization
- **Total energy:** $E$
- **Topological charge:** $Q = \frac{1}{4\pi} \int (\partial_x \vec{n} \times \partial_y \vec{n}) \cdot \vec{n} \, d^2x$
- **Magnetization:** Components $m_x, m_y, m_z$
- **Chirality:** Handedness of spin texture
- **Energy density maps:** Visualize magnetic domain structure

### References

The chiral magnet is a central model in topological magnetism, with applications to skyrmion-based spintronics.

---

## Chiral Liquid Crystal (Oseen-Frank)

**Canonical name:** `"Liquid crystal"`  
**Aliases:** `"Chiral liquid crystal"`, `"Oseen-Frank"`

### Fields

- $\vec{n} \in \mathbb{R}^3$ — Three-component director vector (unit vector indicating molecular orientation)
- $\phi \in \mathbb{R}$ — Electric potential (for polar interactions)

### Energy Functional

$$E_{\text{LC}}[\vec{n}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{K}{2}|\nabla\vec{n}|^2 + K q_0 \vec{n} \cdot (\vec{\nabla} \times \vec{n}) + V(\vec{n}) + \frac{\varepsilon_0}{2}|\nabla\phi|^2 \right]$$

where $K$ is the Frank elastic constant and $q_0$ is the chiral pitch wavenumber.

### Key Parameters

- `K` — Frank elastic constant
- `q0` — Chiral pitch wavenumber (determines helical twist)
- Potential strength `A`, anisotropy direction
- `epsilon_0` — Permittivity for dipole interactions

### Physics

Liquid crystals are partially ordered fluids with **anisotropic molecular interactions**. The Oseen-Frank energy combines:

- **Splay/bend/twist:** Elastic cost of molecular distortion
- **Chiral term:** $K q_0$ induces helical modulation, stabilizing **cholesteric phases**
- **Anisotropy potential:** Favors specific orientations

The chiral pitch and elastic anisotropies compete to form topological defects like **disclinations** (point defects) and **cholesteric layers**.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Uniform director field
- Default or any other mode — Director defect configuration with `ansatz` parameter:
  - `"bloch"` (default) — Bloch-type defect
  - `"neel"` — Néel-type defect
  - `"anti"` — Anti-defect
  - Additional parameter: `skyrmion_rotation` (default: 0.0)

### Observable Quantities

- **Energy:** $E = \int [K |\nabla\vec{n}|^2 + K q_0 \vec{n} \cdot (\nabla \times \vec{n}) + V]$
- **Director field:** $\vec{n}(x, y)$
- **Topological defects:** Disclination line defects with fractional winding
- **Energy density:** Visualizes domain walls and twist regions

---

## Ferromagnetic Superconductor

**Canonical name:** `"Ferromagnetic superconductor"`

### Fields

- $\vec{m} \in \mathbb{R}^3$ — Three-component magnetization ($|\vec{m}| = 1$, constraint enforced)
- $\psi \in \mathbb{C}$ — Complex superconducting order parameter
- $\vec{A} = (A_x, A_y) \in \mathbb{R}^2$ — Gauge field

### Energy Functional

$$E_{\text{FS}}[\vec{m}, \psi, \vec{A}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{1}{2}|\vec{D}\psi|^2 + \frac{1}{2}|\nabla \times \vec{A}|^2 + \frac{\gamma^2}{2}|\nabla\vec{m}|^2 - \vec{m} \cdot (\nabla \times \vec{A}) + \eta_1 |\vec{m}|^2 |\psi|^2 + \eta_2 |\nabla\vec{m}|^2 |\psi|^2 + \frac{a}{2}|\psi|^2 + \frac{b}{4}|\psi|^4 + \frac{\alpha}{2}|\vec{m}|^2 + \frac{\beta}{4}|\vec{m}|^4 \right]$$

where $\eta_1, \eta_2$ couple magnetization and superconductivity.

### Key Parameters

- **Magnetic:** `gamma` (magnetic stiffness), `alpha`, `beta` (magnetic potential)
- **Superconducting:** `a`, `b` (order parameter potential)
- **Coupling:** `eta_1`, `eta_2` (strength of magnetic-superconducting interaction)

### Physics

The ferromagnetic superconductor combines two **competing order parameters**: ferromagnetism and superconductivity. The interplay can produce:

- **Coexistence states:** Both orders present
- **Competition:** One order suppresses the other
- **Intertwined defects:** Magnetic domain walls pinning superconducting vortices, or vice versa

The term $-\vec{m} \cdot (\nabla \times \vec{A})$ produces **magnon-vortex coupling**, allowing magnetic textures to mediate vortex interactions.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Coexisting magnetic and superconducting order
- Default or any other mode — Mixed magnetic-superconducting soliton with `ansatz` parameter:
  - `"bloch"` (default) — Bloch-type magnetic texture
  - `"neel"` — Néel-type magnetic texture
  - `"anti"` — Anti-aligned magnetic texture
  - Additional parameters: `skyrmion_rotation` (default: 0.0), `vortex_number` (default: 1.0)

### Observable Quantities

- **Energy components:** Magnetic, superconducting, interaction
- **Order parameters:** $|\psi|$, $|\vec{m}|$
- **Vortex position and charge:** Topological defects
- **Magnetization field:** $\vec{m}(x, y)$
- **Condensate density:** $|\psi|^2$

---

## Spin-Triplet Superconducting Magnet

**Canonical name:** `"Spin-triplet superconducting magnet"`

### Fields

- $\vec{m} \in \mathbb{R}^3$ — Three-component magnetization ($|\vec{m}| = 1$)
- $\psi_1, \psi_2 \in \mathbb{C}$ — Two complex order parameters (spin-triplet components)
- $\vec{A} = (A_x, A_y) \in \mathbb{R}^2$ — Gauge field

### Energy Functional

$$E_{\text{ST}}[\vec{m}, \psi_\alpha, \vec{A}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{1}{2}|\vec{D}\psi_\alpha|^2 + \frac{1}{2}|\nabla \times \vec{A}|^2 + \frac{a}{2}|\psi_\alpha|^2 + \frac{b_1}{4}|\psi_\alpha|^4 + b_2 |\psi_1|^2 |\psi_2|^2 + c(\psi_1 \psi_2^* + \text{c.c.}) + \frac{\alpha}{2}|\vec{m}|^2 + \frac{\beta}{4}|\vec{m}|^4 + \frac{\gamma^2}{2}|\nabla\vec{m}|^2 - \vec{m} \cdot (\nabla \times \vec{A}) \right]$$

### Key Parameters

- **Magnetic:** Stiffness $\gamma$, potential coefficients $\alpha$, $\beta$
- **Superconducting:** Linear coefficient $a$, quartic coefficients $b_1$, $b_2$, inter-band coupling $c$
- **Coupling:** Magnetic-superconducting interaction strength

### Physics

Spin-triplet superconductivity with embedded ferromagnetism produces a **rich phase diagram** with:

- **Coexistence phases:** Both orders robust
- **Competition:** One order suppresses the other in certain regions
- **Cooper pair formation:** Enhanced by spin alignment (in certain channels)

The two-component order parameter allows **interband interactions** ($b_2$ term) and **pair mixing** ($c$ term). Vortices can bind magnetic defects or even fractional flux, depending on parameter regime.

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Coexisting ferromagnetic and triplet-superconducting order
- Default or any other mode — Mixed magnetic-superconducting soliton with `ansatz` parameter:
  - `"bloch"` (default) — Bloch-type magnetic texture
  - `"neel"` — Néel-type magnetic texture
  - `"anti"` — Anti-aligned magnetic texture
  - Additional parameters: `skyrmion_rotation` (default: 0.0), `vortex_number` (default: 1.0)

### Observable Quantities

- **Energy components:** Magnetic, superconducting (per channel), interactions
- **Order parameters:** $|\psi_1|$, $|\psi_2|$, $|\vec{m}|$
- **Vortex structure:** Position, charge, bound magnetic moment
- **Phase information:** $\arg(\psi_1)$, $\arg(\psi_2)$
- **Energy density maps:** Visualizes intertwined order parameters

---

## Maxwell-Chern-Simons-Higgs Model

**Canonical name:** `"Maxwell-Chern-Simons-Higgs"`  
**Aliases:** `"Chern-Simons-Landau-Ginzburg"`, `"CSLG"`

### Fields

- $\psi \in \mathbb{C}$ — Complex scalar order parameter
- $\vec{A} = (A_x, A_y) \in \mathbb{R}^2$ — In-plane gauge field components
- $A_0 \in \mathbb{R}$ — Temporal gauge component (auxiliary)

### Energy Functional

$$E_{\text{CSLG}}[\psi, \vec{A}] = \int_{\mathbb{R}^2} d^2x \left[ \frac{1}{2}|\vec{D}\psi|^2 + \frac{1}{2}|\nabla \times \vec{A}|^2 + \frac{\lambda}{8}(u^2 - |\psi|^2)^2 + \frac{1}{2}|\nabla A_0|^2 + \frac{1}{2}q^2 |\psi|^2 A_0^2 \right]$$

with constraint: $(\Delta + q^2 |\psi|^2) A_0 = -\kappa B[\vec{A}]$

### Key Parameters

- `lambda` — Higgs potential coupling
- `u` — Vacuum expectation value
- `q` — Charge coupling to $A_0$
- `kappa` — Chern-Simons coupling strength

### Physics

The Maxwell-Chern-Simons-Higgs (MCSH) model combines:

- **Maxwell term:** Standard electromagnetic energy $\frac{1}{2}|\nabla \times \vec{A}|^2$
- **Chern-Simons term:** Topological term $\kappa \epsilon^{\mu\nu\lambda} A_\mu F_{\nu\lambda}$ (implicitly via $B$ dependence)
- **Higgs field:** Scalar condensate with Ginzburg-Landau potential
- **Auxiliary scalar:** $A_0$ captures induced magnetic screening

The Chern-Simons term modifies the dispersion relation of excitations and produces **anomalous statistics** for anyonic defects. Vortices in CSLG models are anyons (particles with fractional charge and fractional spin).

### Initial Configurations

- `"ground"` (aliases: `"uniform"`, `"vacuum"`, `"gs"`) — Vacuum state with no topological defects
- Default or any other mode — Anyon vortex configuration with winding number parameter `vortex_number` (default: 1.0)

### Observable Quantities

- **Energy:** Total energy
- **Topological charge (vorticity):** $Q = \frac{1}{2\pi}\oint \nabla \arg(\psi) \cdot d\vec{\ell}$
- **Magnetic field:** $B = \nabla \times \vec{A}$
- **Flux quantization:** Fractional flux units (anyons carry fractional Chern-Simons flux)
- **Energy density:** Reveals vortex core structure

### References

The CSLG model is fundamental in fractional quantum Hall effect theory and topological quantum computing.

---

## Using theories in simulations

### Basic workflow

```python
from soliton_solver.theories import load_theory
from soliton_solver.core.simulation import Simulation

# Load a theory
theory = load_theory("Baby Skyrme model")

# Create parameters
params = theory.params.default_params(
    xlen=512, ylen=512,
    xsize=20.0, ysize=20.0,
    kappa=0.5  # Theory-specific parameter
)

# Create and initialize simulation
sim = Simulation(params, theory)
sim.initialize({"mode": "skyrmion", "Q": 1})

# Run minimization
energy = sim.observables()["energy"]
for step in range(1000):
    energy, err = sim.step(prev_energy=energy)
    if err < 1e-4:
        break

# Compute observables
obs = sim.observables()
print(f"Energy: {obs['energy']}")
print(f"Topological charge: {obs.get('topological_charge', 'N/A')}")

# Save results
sim.save_output("results")
```

### Interactive visualization

```python
theory = load_theory("Chiral magnet")
params = theory.params.default_params(xlen=512, ylen=512)
sim = Simulation(params, theory)
sim.initialize({"mode": "skyrmion"})

# Launch interactive viewer with real-time rendering
theory.render_gl.run_viewer(sim, params, steps_per_frame=5)
```

### Theory-specific examples

Each theory has a built-in example:

```bash
python -m soliton_solver.examples.baby_skyrme_gl
python -m soliton_solver.examples.chiral_magnet_gl
python -m soliton_solver.examples.bose_einstein_condensate_gl
```

List all examples:

```bash
ls soliton_solver/examples/
```

---

## Adding a new theory

For instructions on implementing your own physics theory, see [Extending the Solver](extending.md).


