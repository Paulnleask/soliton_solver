# Supported Theories

This page gives a more technical account of the theories implemented in `soliton_solver`. The emphasis is on the continuum field theories that support vortices and skyrmions, the fields carried by each model, the dimensional energy functionals for the three theories that are written in physical units, the dimensionless forms used in the implementation for the others, and the Euler-Lagrange equations that govern the relaxed states.

## Overview

The package currently implements a family of two-dimensional field theories that support topological defects. The common theme is that the relaxed configurations are obtained by minimizing a discrete energy functional, and the solver evolves the fields by an arrested Newton-flow relaxation scheme. The theories span superconductivity, magnetism, Bose-Einstein condensates, liquid crystals, and gauge-field models.

The canonical theories in the package are:

| Theory | Fields | Canonical name in the registry |
|--------|--------|---|
| Ginzburg-Landau superconductor | Complex scalar $\psi$ and gauge field $A_i$ | `Ginzburg-Landau superconductor` |
| Anisotropic superconductor | Two complex scalars $\Delta_s,\Delta_d$ and gauge field $A_i$ | `Anisotropic superconductor` |
| Baby Skyrme model | Three-component magnetization $\vec{m}$ | `Baby Skyrme model` |
| Bose-Einstein condensate | Complex scalar $\Psi$ | `Bose-Einstein condensate` |
| Maxwell-Chern-Simons-Higgs / anyon superconductor | Complex scalar $\psi$, gauge fields $A_i$ and scalar potential $A_0$ | `Anyon superconductor` |
| Chiral magnet | Three-component magnetization $\vec{n}$ and scalar potential $\psi$ | `Chiral magnet` |
| Liquid crystal | Three-component director $\vec{n}$ and scalar potential $\phi$ | `Liquid crystal` |
| Ferromagnetic superconductor | Magnetization $\vec{m}$, complex scalar $\psi$ and gauge field $A_i$ | `Ferromagnetic superconductor` |
| Spin-triplet superconducting magnet | Magnetization $\vec{m}$, two complex scalars $\psi_1,\psi_2$ and gauge field $A_i$ | `Spin triplet superconducting ferromagnet` |

The implementation uses the same numerical machinery for all models: a finite-difference spatial discretization, GPU-resident arrays, and a non-linear relaxation dynamics. The continuum equations below are the natural theoretical starting point for the discretized scheme used in the code.

## Ginzburg-Landau superconductor

### Physics modeled

This theory models a two-dimensional superconductor as a complex order parameter $\psi$ coupled to a $U(1)$ gauge field. It is the canonical model for vortex physics in superconductors and is the natural starting point for flux quantization and type-I/type-II vortex behavior.

### Fields and mappings

- $\psi(\vec{x}) \in \mathbb{C}$: the superconducting order parameter. In the code, the complex field is represented by two real components, one for the real part and one for the imaginary part.
- $\vec{A}(\vec{x}) \in \mathbb{R}^2$: the in-plane vector potential. The two real components are stored as gauge-field channels.
- The covariant derivative is $D_j = \partial_j - iA_j$.

### Dimensionless formulation used in the implementation

The implementation works directly with the scaled fields $\varphi$ and $\vec{A}'$, with the reduced functional

$$
\mathcal{E}[\varphi,\vec{A}'] = \int d^2x'\left[
\frac12 |(\nabla' - i\vec{A}')\varphi|^2 + \frac12 |\nabla'\times\vec{A}'|^2 + \frac{1}{8}\left(1 - |\varphi|^2\right)^2
\right].
$$

The scaling $\psi = u\,\varphi$, $\vec{x} = \ell_0\vec{x}'$, and $\vec{A} = \ell_0^{-1}\vec{A}'$ is the standard rescaling used to pass from the physical variables to the implementation variables.

### Scaling conventions

Choose the characteristic length $\ell_0$ and energy scale $\mathcal{E}_0$ so that

$$
\psi = u\,\varphi, \qquad \vec{x} = \ell_0\vec{x}', \qquad \vec{A} = \frac{1}{\ell_0}\vec{A}',
$$

and define the dimensionless constants

$$
\alpha = \frac{\lambda u^2\ell_0^2}{2m_s}, \qquad \beta = \frac{\lambda u^4\ell_0^2}{8}\,\frac{1}{\mathcal{E}_0}.
$$

With the standard scaling above, the implementation functional is

$$
\mathcal{E}[\varphi,\vec{A}'] = \int d^2x'\left[
\frac12 |(\nabla' - i\vec{A}')\varphi|^2 + \frac12 |\nabla'\times\vec{A}'|^2 + \frac{1}{8}\left(1 - |\varphi|^2\right)^2
\right].
$$

### Euler-Lagrange equations

The stationary equations are

$$
\left(\nabla' - i\vec{A}'\right)^2\varphi + \frac12\left(|\varphi|^2 - 1\right)\varphi = 0,
$$

and

$$
-\nabla'^2\vec{A}' + \operatorname{Im}\left[\varphi^*\left(\nabla' - i\vec{A}'\right)\varphi \right] = 0.
$$

These equations admit quantized vortices with winding number $n\in\mathbb{Z}$, and the scalar field vanishes at the core while the phase winds around the vortex.

## Anisotropic superconductor

### The Anisotropic s+id Model

The model consists of a two-component complex order parameter $\psi_\alpha\in\mathbb{C}$ with $\alpha=1,2$ and an electromagnetic vector gauge field $\vec{A}=(A_1,A_2)\in\mathbb{R}^2$.
Let the local coordinate be $x=(x_1,x_2)\in\mathbb{R}^2$ and define the local frame $\partial_j=\partial/\partial x_j$ for $j=1,2$.
The Ginzburg--Landau energy functional in index notation is
$$
    F[\psi,\vec{A}] = \int_{\mathbb{R}^2}\textup{d}^2x \left\{ \frac{1}{2} \gamma_{jk}^{\alpha\beta} \overline{D_j \psi_\alpha} D_k \psi_\beta + \frac{1}{2}|\textup{curl}\vec{A}|^2 + F_{p}(\psi_1,\psi_2) \right\},
$$
where $\alpha,\beta,j,k=1,2$ and the gauge covariant derivative is defined by the vector $\vec{D}=\vec{\nabla}+iQ\vec{A}$, and $\vec{\nabla}=(\partial_1,\partial_2)$ is the gradient operator.
Note that we define
$$
    \overline{D_j\psi_\alpha} = \partial_j \bar{\psi}_\alpha -iQA_j\bar{\psi}_\alpha,
$$
and set $Q=2q$ since it describes the motion of a Cooper pair.
The anisotropy matrices for a $s+id$ order parameter are given by
$$
    \gamma_{11} = 2
    \begin{pmatrix}
        \gamma_1 & -\gamma_{12} \\
        -\gamma_{12} & \gamma_2
    \end{pmatrix},
    \quad
    \gamma_{22} = 2
    \begin{pmatrix}
        \gamma_1 & \gamma_{12} \\
        \gamma_{12} & \gamma_2
    \end{pmatrix}.
$$
The potential energy we are considering is defined by
$$
    F_{p}(\psi_1,\psi_2) = \alpha_1 |\psi_1|^2 + \alpha_2 |\psi_2|^2 + \beta_1 |\psi_1|^4 + \beta_2 |\psi_2|^4 + \beta_3 |\psi_1|^2 |\psi_2|^2 + \beta_4 \left( \psi_1^2 \bar{\psi}_2^2 + \bar{\psi}_1^2 \psi_2^2 \right).
$$

To minimize the potential energy $F_p$ with respect to the order parameters $\psi_\alpha$, let us write the order parameters as
$$
    \psi_1 = \rho_1 e^{i\theta_1}, \quad \psi_2 = \rho_2    e^{i\theta_2}.
$$
Then, the only term in the potential energy that has angular dependence is the biquadratic Josephson term
$$
    \psi_1^2 \bar{\psi}_2^2 + \bar{\psi}_1^2 \psi_2^2 = 2 \rho_1^2 \rho_2^2 \cos[2(\theta_1 - \theta_2)].
$$
So the potential becomes
$$
    F_p = \alpha_1 \rho_1^2 + \alpha_2 \rho_2^2 + \beta_1 \rho_1^4 + \beta_2 \rho_2^4 + \beta_3 \rho_1^2 \rho_2^2 + 2\beta_4 \rho_1^2 \rho_2^2 \cos[2(\theta_1 - \theta_2)].
$$
Consider now the phase difference $\theta_{12} = \theta_1 - \theta_2$.
The only term depending on the phase difference is
$$
    2\beta_4 \rho_1^2 \rho_2^2 \cos(2\theta_{12}).
$$
If $\beta_4 < 0$, the minimum occurs at $\cos(2\theta_{12}) = 1 \Rightarrow \theta_{12} = 0, \pi$.
If $\beta_4 > 0$, the minimum occurs at $\cos(2\theta_{12}) = -1 \Rightarrow \theta_{12} = \pm \pi/2$.
So we choose
$$
    \theta_{12} =
    \begin{cases}
        \pm\pi/2 & \text{if } \beta_4 > 0 \\
        0,\pi & \text{if } \beta_4 < 0
    \end{cases}.
$$
With the optimal phase chosen, the potential becomes
$$
    F_p^{\text{min phase}}(\rho_1, \rho_2) = \alpha_1 \rho_1^2 + \alpha_2 \rho_2^2 + \beta_1 \rho_1^4 + \beta_2 \rho_2^4 + \left(\beta_3 + 2\epsilon|\beta_4|\right)\rho_1^2 \rho_2^2,
$$
where $\epsilon = \operatorname{sign}(\beta_4)$.
So, the effective potential is
$$
    F_p^{\text{eff}}(\rho_1, \rho_2) = \alpha_1 \rho_1^2 + \alpha_2 \rho_2^2 + \beta_1 \rho_1^4 + \beta_2 \rho_2^4 + \tilde{\beta}_3 \rho_1^2 \rho_2^2,
$$
with $\tilde{\beta}_3 = \beta_3 + 2\beta_4\cos2\theta_{12}$.

Now we want to minimize $F_p^{\text{eff}}$ with respect to $\rho_1^2, \rho_2^2 \geq 0$.
We compute the variation of the potential energy, which yields
$$
\begin{aligned}
    \left.\frac{\partial F_p}{\partial \rho_1}\right|_{\rho_i=u_i} &= 2\alpha_1 u_1 + 4\beta_1 u_1^3 + 2\tilde{\beta}_3 u_1 u_2^2 = 0, \\
    \left.\frac{\partial F_p}{\partial \rho_2}\right|_{\rho_i=u_i} &= 2\alpha_2 u_2 + 4\beta_2 u_2^3 + 2\tilde{\beta}_3 u_2 u_1^2 = 0.
\end{aligned}
$$
Divide both by $u_1, u_2$ (assume not zero), to get
$$
\begin{aligned}
    \alpha_1 + 2\beta_1 u_1^2 + \tilde{\beta}_3 u_2^2 &= 0, \\
    \alpha_2 + 2\beta_2 u_2^2 + \tilde{\beta}_3 u_1^2 &= 0.
\end{aligned}
$$
These are coupled nonlinear equations in $u_1^2, u_2^2$. Solutions (if real and non-negative) give minima candidates.
We introduce the variables
$$
    x = u_1^2, \quad y = u_2^2.
$$
Then the equations become
$$
\begin{aligned}
    \alpha_1 + 2\beta_1 x + \tilde{\beta}_3 y = 0, \\
    \alpha_2 + 2\beta_2 y + \tilde{\beta}_3 x = 0.
\end{aligned}
$$
Solving the first equation for $y$ gives
$$
    y = -\frac{\alpha_1 + 2\beta_1 x}{\tilde{\beta}_3}.
$$
Substitute this into the second equation,
$$
    \alpha_2 + 2\beta_2 \left( -\frac{\alpha_1 + 2\beta_1 x}{\tilde{\beta}_3} \right) + \tilde{\beta}_3 x = 0.
$$
Multiply through to simplify,
$$
    \alpha_2 - \frac{2\beta_2 \alpha_1}{\tilde{\beta}_3} - \frac{4\beta_1\beta_2}{\tilde{\beta}_3} x + \tilde{\beta}_3 x = 0.
$$
Multiply both sides by $\tilde{\beta}_3$ to eliminate the denominator, such that we have
$$
    \tilde{\beta}_3 \alpha_2 - 2\beta_2 \alpha_1 - 4\beta_1\beta_2 x + \tilde{\beta}_3^2 x = 0.
$$
This can be solved for $x = u_1^2$, giving
$$
    x = \frac{2\beta_2 \alpha_1 - \tilde{\beta}_3 \alpha_2}{\tilde{\beta}_3^2 - 4\beta_1 \beta_2}.
$$
Similarly, using symmetry, we can solve for $y = u_2^2$,
$$
    y = \frac{2\beta_1 \alpha_2 - \tilde{\beta}_3 \alpha_1}{\tilde{\beta}_3^2 - 4\beta_1 \beta_2}.
$$
Therefore, the ground state is determined to be given by
$$
\begin{aligned}
    u_1^2 &= \frac{2\beta_2 \alpha_1 - \tilde{\beta}_3 \alpha_2}{\tilde{\beta}_3^2 - 4\beta_1 \beta_2} \\
    u_2^2 &= \frac{2\beta_1 \alpha_2 - \tilde{\beta}_3 \alpha_1}{\tilde{\beta}_3^2 - 4\beta_1 \beta_2}
\end{aligned}
\quad \text{with } \tilde{\beta}_3 = \beta_3 + 2\beta_4\cos2\theta_{12}.
$$

### $\mathbb{C}P^1$ Skyrmions

When integer flux vortices split into spatially separated fractional vortices in each order parameter $\psi_\alpha$, such that their cores (zeroes of the order parameters $\psi_\alpha=0$) are never coincident, skyrmions form.
For such structures, we may construct a gauge invariant field $\phi:\mathbb{R}^2\rightarrow S^2$, given by
$$
    \phi= \frac{1}{|\Delta_s|^2+|\Delta_d|^2} 
    \begin{pmatrix}
        \Delta_s^*\Delta_d + \Delta_s\Delta_d^* \\
        i(\Delta_s^*\Delta_d - \Delta_s\Delta_d^*) \\
        |\Delta_d|^2-|\Delta_s|^2        
    \end{pmatrix}.
$$
We can describe the skyrmion field using the gauge invariant quantities $\phi$, $\rho=\sqrt{|\psi_1|^2+|\psi_2|^2}$ and the supercurrent $J$, where we are regarding the supercurrent as a 1-form on $\mathbb{R}^2$.
We can express the magnetic field (as a 2-form on $\mathbb{R}^2$) in terms of these gauge invariant quantities as
$$
    B = \frac{1}{2}\phi^*\omega - \textup{d}\left( \frac{J}{\rho^2} \right),
$$
where $\omega$ is the usual area form on the target $S^2$ and $\textup{d}$ is the exterior derivative.
Then, using Stoke's Theorem, we observe that the skyrmion number is precisely the number of magnetic flux quanta in the field configuration, that is
$$
    \int_{\mathbb{R}^2} B = \frac{1}{2}\int_{\mathbb{R}^2} \phi^*\omega = 2\pi \mathcal{Q}(\phi).
$$
The skyrmion number $\mathcal{Q}$ can be computed using the following integral representation,
$$
    \mathcal{Q}(\phi) = \frac{i\epsilon_{ji}}{2\pi}\int_{\mathbb{R}^2} \textup{d}^2x \, \frac{1}{|\Psi|^4} \left( |\Psi|^2 \partial_i\Psi^\dagger \partial_j\Psi + \Psi^\dagger \partial_i \Psi \partial_j \Psi^\dagger \Psi \right),
$$
where $\Psi=(\psi_1,\psi_2)$.

### Numerical Implementation

The associated Ginzburg--Landau field equations are found to be
$$
\begin{align*}
    \frac{\delta F}{\delta \bar{\psi}_\alpha} = \, & \frac{\partial F}{\partial \bar{\psi}_\alpha} - \frac{1}{2}\gamma_{jk}^{\alpha\beta}D_j D_k \psi_\beta, \\
    \frac{\delta F}{\delta A_j} = \, & \frac{iQ}{2} \left( \gamma_{kj}^{\alpha\beta} \psi_\beta \overline{D_k\psi_\alpha} - \gamma_{jk}^{\beta\alpha}\bar{\psi}_\beta D_k \psi_\alpha \right) + \partial_j \partial_k A_k - \partial_k \partial_k A_j.
\end{align*}
$$
So, `soliton_solver` is solving the second order coupled system
$$
\begin{align*}
    \frac{\textup{d}^2\psi_\alpha}{\textup{d}t^2} = \, & -\frac{\delta F}{\delta \bar{\psi}_\alpha}, \\
    \frac{\textup{d}^2 A_j}{\textup{d}t^2} = \, & -\frac{\delta F}{\delta A_j},
\end{align*}
$$
where $t$ is a fictitious time coordinate.

### References

A. Talkachov, P. Leask and E. Babaev, *Multiple correlation lengths and type-1.5 superconductivity in U(1) superconductors due to hidden competition between irreducible representations of nonlocal pairing*, [Phys. Rev. B **113**, 224520 (2026)](https://doi.org/10.1103/bf2v-wtkt)

## Baby Skyrme model

### Physics modeled

The Baby Skyrme model is the two-dimensional version of the Skyrme model and is used to model topological spin textures in magnetic systems. It supports Bloch, Néel and anti-skyrmions and is closely related to the magnetization textures stabilized in chiral magnets.

### Fields and mappings

- $\vec{m}(\vec{x}) \in \mathbb{R}^3$: the magnetization vector. The solver stores three real components.
- The field is constrained to unit length, $|\vec{m}|=1$, by the non-linear dynamics.

### Dimensionless formulation used in the implementation

The implementation uses the dimensionless functional

$$
\mathcal{E}[\vec{m}] = \int d^2x'\left[
\frac12 |\nabla'\vec{m}|^2 + \frac{\lambda}{4}\left(\partial_{x'}\vec{m}\times\partial_{y'}\vec{m}\right)^2 + \bar{V}(\vec{m})
\right],
$$

where $\lambda = \kappa^2/(\ell_0^2\mathcal{J})$ is the scaled Skyrme coupling and $\bar{V}$ is the rescaled potential.

### Scaling conventions

The implementation uses the scaled coordinate $\vec{x}'$ directly, and the corresponding dimensionless coupling is

$$
\lambda = \frac{\kappa^2}{\ell_0^2\mathcal{J}}.
$$

The reduced functional used by the solver is

$$
\mathcal{E}[\vec{m}] = \int d^2x'\left[
\frac12 |\nabla'\vec{m}|^2 + \frac{\lambda}{4}\left(\partial_{x'}\vec{m}\times\partial_{y'}\vec{m}\right)^2 + \bar{V}(\vec{m})
\right].
$$

### Euler-Lagrange equations

The corresponding field equation is

$$
\partial_t\vec{m} = \vec{m}\times\left(\nabla'^2\vec{m} - \lambda\,\vec{J}[\vec{m}] - \frac{\partial \bar{V}}{\partial \vec{m}}\right),
$$

where $\vec{J}[\vec{m}]$ denotes the Skyrme current term generated by the quartic topological stabilizer. In the static limit, the right-hand side vanishes and the resulting equation describes the balance between gradient, Skyrme and potential terms.

## Rotating Bose-Einstein Condensates

### Fields and parameters

- Condensate wavefunction amplitude: $\Psi \in \mathbb{C}$, $[\Psi] = \mathrm{m}^{-3/2}$
- Particle density: $n \in \mathbb{R}$,  $[n] = \mathrm{m}^{-3}$
- Atomic mass: $[m] = \mathrm{kg}$
- Trap frequency: $[\omega] = \mathrm{s}^{-1}$
- Short-range interaction strength: $[g] = \mathrm{m}^{3}\,\mathrm{s}^{-2}$
- S-wave scattering length: $[a_s] = \mathrm{m}$
- Rotation frequency: $[\Omega] = \mathrm{s}^{-1}$

### The BEC Model

This theory describes a dilute Bose-Einstein condensate as a complex order parameter in a harmonic trap.
It is the standard mean-field model for trapped condensates, Thomas-Fermi profiles, and vortex formation under rotation.

The order parameter for the Bose-Einstein condensate (BEC) is the single complex scalar field $\Psi \in \mathbb{C}$.
The theory is built from the usual mean-field energy functional for a trapped, weakly interacting gas, with a short-range contact interaction parameter $g = \tfrac{4\pi \hbar^2 a_s}{m}$ and a harmonic trapping potential.

The Hamiltonian we are considering is given by
$$
    E[\Psi] = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{r} \left\{ \frac{\hbar^2}{2m}|\bm{\nabla} \Psi|^2 + V_{\textup{trap}}(\mathbf{r})|\Psi|^2 + \frac{g}{2}|\Psi|^4 \right\},
$$
where $g=\tfrac{4\pi \hbar ^2 a_s}{m}$ is known as the short-range interaction parameter, and the trapping potential we will consider is the harmonic trap
$$
    V_{\textup{trap}}(\mathbf{r}) = \frac{1}{2}m \omega^2 |\mathbf{r}|^2.
$$
There is a conserved quantity, the number of atoms $N$, which is defined by
$$
    N = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{r} \, |\Psi(\mathbf{r})|^2.
$$
Hence, the number density of the BEC gas is
$$
    n(\mathbf{r}) = |\Psi(\mathbf{r})|^2.
$$
We see that, since $N$ is the number of atoms and is therefore dimensionless, the order parameter necessarily has units $[\Psi]=\textup{m}^{-3/2}$ and $[n]=\textup{m}^{-3}$.

Let us consider the following energy, length and condensate rescalings
$$
    E = E_0 H, \quad \mathbf{r}=L_0\mathbf{x}, \quad \Psi (\mathbf{r})=\Psi_0 \psi(\mathbf{x}),
$$
where $H$, $\mathbf{x}$ and $\psi$ are dimensionless.
Let us choose $\Psi_0 = \sqrt{N} L_0^{-3/2}$ such that the normalization is now
$$
    \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \, |\psi|^2 = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{r} \frac{1}{\Psi_0^2 L_0^3} |\Psi|^2 = \frac{N}{N} = 1.
$$
The rescaled energy becomes
$$
    H = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \left\{ \frac{1}{2} \frac{\hbar^2 L_0^3 \Psi_0^2}{m E_0 L_0^2} |\bm{\nabla}_{\mathbf{x}} \psi|^2 + \frac{1}{2}\frac{L_0^3 \Psi_0^2m \omega^2 L_0^2}{E_0}|\mathbf{x}|^2 |\psi|^2+ \frac{1}{2} \frac{L_0^3 \Psi_0^4 g}{E_0} |\psi|^4\right\}
$$
Since we have chosen to fix $\Psi_0$ by the normalization condition, we have freedom in choice of $L_0$ and $E_0$.
Let us choose these such that
$$
    \frac{\hbar^2 L_0^3 \Psi_0^2}{m E_0 L_0^2} = \frac{L_0^3 \Psi_0^2m \omega^2 L_0^2}{E_0} = 1.
$$
Therefore, the relevant rescalings are
$$
    L_0 = \sqrt{\frac{\hbar}{m\omega}}, \quad E_0 = N \hbar \omega.
$$
With these rescalings, the dimensionless energy reduces to
$$
    H = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \left\{ \frac{1}{2}|\bm{\nabla}_{\mathbf{x}} \psi|^2 + \frac{1}{2}|\mathbf{x}|^2|\psi|^2 + \frac{\beta}{2}|\psi|^4 \right\},
$$
with the rescaled quartic potential
$$
    \beta = \frac{L_0^3 \Psi_0^4}{E_0}g = 4\pi N a_s \sqrt{\frac{m\omega}{\hbar}}
$$
A quick dimensional analysis shows that $\beta$ is indeed dimensionless,
$$
    [\beta] = [a_s][m]^{1/2}[\omega]^{1/2}[\hbar]^{-1/2} = \textup{m} \, \textup{kg}^{1/2} \, \textup{s}^{-1/2} (\textup{m}^2\,\textup{kg} \, \textup{s}^{-1})^{-1/2} = 1.
$$

### The Rotating BEC Model

If the condensate is rotated about the $z$-axis with angular frequency $\Omega$, then in the rotating frame the dimensional energy becomes
$$
    E_\Omega[\Psi] = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{r} \left\{ \frac{\hbar^2}{2m}|\bm{\nabla} \Psi|^2 + V_{\textup{trap}}(\mathbf{r})|\Psi|^2 + \frac{g}{2}|\Psi|^4 \right\} - \Omega \int_{\mathbb{R}^3} \textup{d}^3\mathbf{r} \, \left(\Psi^* \hat{L}_z \Psi \right)
$$
where
$$
    \hat{L}_z = -i\hbar \left( x \partial_y - y \partial_x \right).
$$

Under the rescaling
$$
    E = E_0 H, \quad \mathbf{r}=L_0\mathbf{x}, \quad \Psi (\mathbf{r})=\Psi_0 \psi(\mathbf{x}),
$$
with
$$
    \Psi_0 = \sqrt{N} L_0^{-3/2},
$$
the rotational contribution becomes
$$
    H_{\textup{rot}} = - \frac{\Omega L_0^3 \Psi_0^2 \hbar}{E_0} \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \, \psi^* \hat{\ell}_z \psi - \frac{\hbar \Omega N}{E_0} \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \, \psi^* \hat{\ell}_z \psi,
$$
where the dimensionless angular momentum operator is
$$
    \hat{\ell}_z = -i\left( x \partial_y - y \partial_x \right).
$$

Using
$$
    E_0 = N \hbar \omega
$$
we obtain
$$
    \frac{\hbar \Omega N}{E_0} = \frac{\Omega}{\omega}.
$$
Hence, the full dimensionless energy is
$$
    H_\Omega = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \left\{ \frac{1}{2}|\bm{\nabla}_{\mathbf{x}} \psi|^2 + \frac{1}{2}|\mathbf{x}|^2|\psi|^2 + \frac{\beta}{2}|\psi|^4 - \frac{\Omega}{\omega} \psi^* \hat{\ell}_z \psi \right\},
$$
with
$$
    \beta = 4\pi N a_s \sqrt{\frac{m\omega}{\hbar}}.
$$

### The Thomas-Fermi Profile

Now that the energy is in a dimensionless form, we need to determine the ground state configuration for the condensate $\psi$.
Consider the potential energy
$$
    E_{\textup{pot}}[\psi]= \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \left\{ \frac{1}{2}|\mathbf{x}|^2|\psi|^2 + \frac{\beta}{2}|\psi|^4 \right\},
$$
and now introduce a Lagrange multiplier to ensure the normalization condition,
$$
    L[\psi,\lambda] = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \left\{ \frac{1}{2}|\mathbf{x}|^2|\psi|^2 + \frac{\beta}{2}|\psi|^4 - \lambda|\psi|^2 \right\} + \lambda \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \, |\psi|^2
$$
Define the number density $n(\mathbf{x})=|\psi(\mathbf{x})|^2$ such that
$$
    L[n,\lambda] = \int_{\mathbb{R}^3} \textup{d}^3\mathbf{x} \left\{ \frac{1}{2}|\mathbf{x}|^2n + \frac{\beta}{2}n^2 - \lambda n \right\} + \lambda,
$$
where we have used the normalization condition.
Variation of this with respect to the dimensionless condensate $\psi$ gives the Karush--Kuhn--Tucker (KKT) condition
$$
    \frac{\delta L}{\delta n} = \frac{1}{2}|\mathbf{x}|^2 + \beta n(\mathbf{x}) - \lambda = 0, \quad n > 0.
$$
Hence, the ground state configuration is given by the Thomas--Fermi (TF) profile
$$
    n_{\textup{TF}}(\mathbf{x}) = \max\left(0, \frac{1}{\beta}\left[\lambda - \frac{1}{2}|\mathbf{x}|^2\right] \right).
$$

Using the harmonic trapping potential, the TF profile is
$$
    n_{\textup{TF}}(\mathbf{x}) = \frac{1}{\beta}\left(\lambda - \frac{1}{2}|\mathbf{x}|^2\right), \quad |\mathbf{x}|^2 \leq R^2 = 2\lambda.
$$
We now need to determine the Lagrange multiplier $\lambda$.
This depends on the dimension of the system we consider.
We will be working in two dimensions, so our normalization condition for the ground state becomes
$$
    \int_{\mathbb{R}^2} \textup{d}^2 \mathbf{x} \, |\psi(\mathbf{x})|^2 = \int_{\mathbb{R}^2} \textup{d}^2 \mathbf{x} \, n_{\textup{TF}}(\mathbf{x}) = 1.
$$
We can use this to determine $\lambda$,
$$
    1 = \frac{2\pi}{\beta} \int_0^R \textup{d}r \left( \lambda - \frac{1}{2}r^2 \right)r = \frac{\pi\lambda^2}{\beta}.
$$
Hence, we see that the Lagrange multiplier $\lambda$ and the TF radius $R$ are
$$
    \lambda = \sqrt{\frac{\beta}{\pi}}, \quad R^2 = 2\sqrt{\frac{\beta}{\pi}}.
$$
Finally, the axially symmetric ground state configuration is given by the TF profile
$$
    n_{\textup{TF}}(r) = \frac{1}{\sqrt{\beta\pi}} - \frac{1}{2\beta}r^2, \quad r^2 \leq 2\sqrt{\frac{\beta}{\pi}}.
$$

### The Numerical Implementation

The algorithm deals with solving the static BEC equation
$$
    \frac{\delta H_\Omega}{\delta \psi^*} = -\frac{1}{2}\nabla^2\psi + \frac{1}{2}|\mathbf{x}|^2\psi + \beta \psi |\psi|^2 - \frac{\Omega}{\omega} \hat{\ell}_z \psi.
$$
This is achieved using `soliton_solver`.
We formulate the minimization as a second order dynamical problem and solve the second order system
$$
    \frac{\textup{d}^2\psi}{\textup{d}t^2} = -\frac{\delta H_\Omega}{\delta \psi^*} = \frac{1}{2}\nabla^2\psi - \frac{1}{2}|\mathbf{x}|^2\psi - \beta\psi |\psi|^2 + \frac{\Omega}{\omega} \hat{\ell}_z \psi,
$$
where $t$ is a fictitious time coordinate and with some appropriate initial configuration $\psi(0)=\psi_0$.
This system can then be reduced to a coupled first order system, which we solve using a fourth order Runge-Kutta method.

## Chern-Simons-Landau-Ginzburg Theory of Vortex Anyons

### Fields and parameters

- Order parameter: $\psi \in \mathbb{C}$
- Gauge field: $\vec{A}=(A_0,A_1,A_2)\in\mathbb{R}^{2+1}$
- Ginzburg-Landau parameter: $\lambda$
- Dimensionless Higgs mass: $m$
- Chern-Simons level: $\kappa$

### The CSLG Model

The Chern-Simons-Landau-Ginzburg (CSLG) model is described by the superconducting order parameter $\psi: \mathbb{R}^{2+1}\rightarrow\mathbb{C}$, also known as the Higgs field, and an abelian gauge field $\vec{A}=(A_0,A_1,A_2)\in\mathbb{R}^{2+1}$.
Associated to the abelian gauge field is the gauge covariant derivative $D_\mu=\partial_\mu + iqA_\mu$, where $q$ is the gauge charge.
The gauge field strength is given by the curvature $F_{\mu\nu}=\partial_\mu A_\nu - \partial_\nu A_\mu$.
From the field strength we define the magnetic field $B=F_{12}$ and the electric field $E_i=F_{0i}$.
We will consider the model defined on Minkowski spacetime $\mathbb{R}^{2+1}$, which is endowed with the Minkowski metric $\eta$ and metric signature $(+--)$.

This GL model of anyon superconductivity is a gauge field theory that exhibits spontaneous symmetry breaking.
The local $U(1)$ invariance is realized by the gauge transformation $A_\mu \mapsto A_\mu +\partial_\mu \alpha(x)$ and $\psi \mapsto\psi e^{i\alpha(x)}$.
The action of the anyonic theory is $S=\int \textup{d}^3x\mathcal{L}$, where the Lagrangian is given b
$$
    \mathcal{L} = \frac{1}{2} D_\mu \psi \overline{D^\mu \psi} - \frac{1}{4} F^{\mu\nu} F_{\mu\nu} - V(|\psi|) + \frac{\kappa}{4} \epsilon^{\alpha\beta\gamma} A_\alpha F_{\beta\gamma}.
$$
The first three terms describe the GL model, also known as the abelian Higgs model in this context, where the second term is the Yang-Mills, or Maxwell, contribution.
The last term is the topological Chern--Simons (CS) term,
$$
    \mathcal{L}_{\textup{CS}} = \frac{\kappa}{4} \epsilon^{\alpha\beta\gamma} A_\alpha F_{\beta\gamma} = \frac{\kappa}{2} \left( A_0 B - \epsilon_{ij}A_i E_j \right).
$$

In the regular Ginzburg-Landau model (or abelian Higgs), the electric field is absent when considering statics. 
However, due to the presence of the CS term, we see that the electric field does not vanish $E_i=F_{0i}=-\partial_i A_0 \neq 0$ and neither does the kinetic term $D_0 \psi \overline{D_0 \psi}=q^2 A_0^2|\psi|^2 \neq 0$.
So, the static Lagrangian of the model is
$$
    \mathcal{L}_{\textup{static}} = \frac{1}{2}(\partial_i A_0)^2 + \frac{\kappa}{2} \left( A_0 B + \epsilon_{ij}A_i \partial_j A_0 \right) + \frac{1}{2}q^2 A_0^2|\psi|^2 - \left[ \frac{1}{2} D_i \psi \overline{D_i \psi} + \frac{1}{2}B^2 + V(|\psi|) \right].
$$
We can simplify this a bit by applying an integration by parts,
$$
    \int_{\mathbb{R}^2} \textup{d}^2x \, \epsilon_{ij}A_i \partial_j A_0  = \int_{\mathbb{R}^2} \textup{d}^2x \, A_0 B.
$$
Hence, the static Lagrangian can be expressed as
$$
    \mathcal{L}_{\textup{static}} = \frac{1}{2}(\partial_i A_0)^2 + \kappa A_0 B + \frac{1}{2}q^2 A_0^2|\psi|^2 - \left[ \frac{1}{2} D_i \psi \overline{D_i \psi} + \frac{1}{2}B^2 + V(|\psi|) \right].
$$
Varying the static Lagrangian with respect to the electric potential $A_0$ reveals Gauss' law as an elliptic PDE,
$$
    \left( -\nabla^2 + q^2|\psi|^2 \right) A_0 = -\kappa B.
$$

Since static vortex anyons are minimizers of the static energy functional, we introduce the static energy of the theory by defining
$$
    E = -\int_{\mathbb{R}^2} \textup{d}^2x \, \mathcal{L}_{\textup{static}} = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} D_i \psi \overline{D_i \psi} + \frac{1}{2}B^2 + V(|\psi|) - \frac{1}{2}(\partial_i A_0)^2 - \kappa A_0 B - \frac{1}{2}q^2 A_0^2|\psi|^2 \right\}.
$$
At a first glance this appears not to be bounded from below.
However, the static energy can be transformed into a form that is bounded below and positive (semi-)definite as follows.
Let us take the inner product of Gauss' law with the potential $A_0$ and then integrate by parts to obtain
$$
    -\int_{\mathbb{R}^2} \textup{d}^2x \, \kappa A_0 B = \int_{\mathbb{R}^2} \textup{d}^2x \left[ -A_0 \partial_i \partial_i A_0 + q^2|\psi|^2 A_0^2 \right] = \int_{\mathbb{R}^2} \textup{d}^2x \left[ (\partial_i A_0)^2 + q^2|\psi|^2 A_0^2 \right].
$$
Substituting this into the static energy gives an expression for the static energy that is clearly bounded below by 0,
$$
    E = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} |\vec{D} \psi|^2 + \frac{1}{2}B^2 + V(|\psi|) + \frac{1}{2}|\vec{\nabla} A_0|^2 + \frac{1}{2}q^2 A_0^2|\psi|^2 \right\}.
$$
This energy is positive definite and bounded below, therefore it is amenable to gradient descent methods.

In `soliton_solver`, we consider the conventional quartic Higgs potential
$$
    V(|\psi|) = \frac{\lambda}{8} \left( m^2-|\psi|^2 \right)^2,
$$
where the Higgs mass is $m_H=\sqrt{\lambda}m$ and $\lambda$ is the GL parameter dictating the type of superconductivity.

### Anyon Nature

In the presence of a CS term the relationship between magnetic flux and electric charge requires special care. 
The starting point is the Gauss constraint, which ultimately relates the flux to the charge.
This equation encapsulates the mixing between electric and magnetic fields: a localized magnetic flux distribution acts as a source for the electrostatic potential $A_0$.

The physical electric field is $\vec{E}=-\vec{\nabla}A_0$, so we can compute the electric charge density via the Maxwell equation
$$
    \rho_e = \vec{\nabla}\cdot\vec{E} = -\nabla^2 A_0 = -\kappa B - q^2|\psi|^2 A_0,
$$
where we have used Gauss' law to express the charge in terms of the matter and magnetic fields.
Hence, the total physical electric charge $Q_e$ is
$$
    Q_e = \int_{\mathbb{R}^2} \textup{d}^2x \, \rho_e = -\kappa \Phi - q^2 \int_{\mathbb{R}^2} \textup{d}^2x\,A_0 |\psi|^2,
$$
where $\Phi = \int_{\mathbb{R}^2} \textup{d}^2x B$ is the total magnetic flux.
A single magnetic flux quantum is $\Phi_0=2\pi/q$, so the total quantized magnetic flux is $\Phi=N\Phi_0$.
For localized static solutions, $A_0$ decays exponentially and $E_i\to 0$ at spatial infinity.
Therefore, the total electric charge should be zero for localized solutions, $Q_e=0$, which gives the relation
$$
    \int_{\mathbb{R}^2} \textup{d}^2x \, q^2A_0 |\psi|^2 = -\kappa\Phi.
$$

Although the total Maxwell charge $Q_e$ vanishes, the condensate carries a nontrivial internal $U(1)$ charge $Q_m$.
Under a global phase rotation of the scalar, $\psi \mapsto e^{i\alpha}\psi$, the associated Noether current is
$$
    J_\mu = \frac{iq}{2}(\psi \partial_\mu \bar{\psi} - \bar{\psi} \partial_\mu \psi) + q^2 A_\mu |\psi|^2.
$$
From the supercurrent, we obtain the Noether electric charge density
$$
    \rho_m=J_0=q^2A_0 |\psi|^2.
$$
Therefore, we find that the magnetic flux $\Phi$ and the Noether electric charge $Q_m$ are related by
$$
    Q_m = \int_{\mathbb{R}^2} \textup{d}^2x\,\rho_m = -\kappa \Phi.
$$
This is the electric charge carried by the condensate relative to the internal $U(1)$ gauge symmetry; it is non-zero and entirely tied to the magnetic flux by Gauss' law.
It shows that each unit of magnetic flux carries a quantized amount of Noether charge proportional to the CS level $\kappa$.
This is the hallmark of anyon superconductivity: the non-trivial matter charge signals that vortices in this model are anyonic objects, with each magnetic flux quantum binding a fixed electric charge determined by the CS level.

### Numerical Implementation

Static vortex anyons are critical points of the static energy, so we must solve the associated Euler-Lagrange field equations of the model and also satisfy the Gauss constraint.
The Euler-Lagrange field equations are obtained by varying the unreduced static energy functional with respect to the Higgs field $\psi$ and the gauge field $(A_1,A_2)$.
This gives us the static Ginzburg--Landau equations
$$
\begin{align*}
    D_i D_i \psi = \, & 2\frac{\partial V}{\partial \bar{\psi}} - q^2A_0^2 \psi, \\
    \partial_j (\partial_j A_i - \partial_i A_j) = \, &  J_i - \kappa \epsilon_{ij}\partial_j A_0.
\end{align*}
$$
We formulate the minimization as a second order dynamical problem and `soliton_solver` solves the second order coupled system
$$
\begin{align*}
    \frac{\textup{d}^2\psi}{\textup{d}t^2} = \, & \frac{1}{2}D_i D_i \psi - \frac{\partial V}{\partial \bar{\psi}} + \frac{1}{2}q^2A_0^2 \psi, \\
    \frac{\textup{d}^2 A_i}{\textup{d}t^2} = \, & \partial_j (\partial_j A_i - \partial_i A_j) - J_i + \kappa \epsilon_{ij}\partial_j A_0, \\
    \frac{\textup{d}^2 A_0}{\textup{d}t^2} = \, & \nabla^2 A_0 - q^2|\psi|^2 A_0 -\kappa B,
\end{align*}
$$
where $t$ is a fictitious time coordinate.

### References

P. Leask, *Anyon bound states and hybrid superconductivity*, [Phys. Rev. Lett. **137**, 026003 (2026)](https://doi.org/10.1103/15fc-r786)

## Chiral Magnet with Demagnetization

### Fields and parameters

- Magnetization order parameter: $\vec{m} \in \mathbb{R}^3$, $[\vec{m}] = \textup{Am}^{-1}$.
- Magnetostatic potential: $\psi \in \mathbb{R}$, $[\psi] = \textup{Tm}$
- Exchange stiffness: $J$, $[J]=\textup{Jm}^{-1}$
- Dzyaloshinskii–Moriya interaction strength: $\mathcal{D}$, $[\mathcal{D}]=\textup{Jm}^{-2}$
- Magnetic saturation density: $M_s$, $[M_s] = \textup{Am}^{-1}$
- Anisotropy constant: $K_m$, $[K_m] = \textup{Jm}^{-3}$
- External magnetic field $B_{\textup{ext}}$, $[B_{\textup{ext}}] = \textup{T}$
- Permeability of free space: $\mu_0$, $[\mu_0] = \textup{mkgs}^{-2}\textup{A}^{-2}$

### The Chiral Ferromagnet Model

The model is a translationally invariant three-dimensional chiral ferromagnet whose magnetization field is assumed to vary smoothly and have constant magnitude, and so can be written $\vec{m}(\vec{x})=M_s\vec{n}(\vec{x})$ where $|\vec{n}|=1$ and $M_s$ is a constant parameter called the magnetic saturation density.
The total energy of such a field is assumed to take the form
$$
    E = \int_{\mathbb{R}^3}\textup{d}^3x \left\{\frac{J}{2}\,|\textup{d}\vec{n}|^2 + \mathcal{D}  \sum_{i=1}^3 \vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) + K_m(1-n_3^2) + M_s B_{\textup{ext}}(1-n_3) \right\} +E_{\textup{DDI}},
$$ 
where $E_{\textup{DDI}}$ is the dipole-dipole interaction energy.
The potential has two terms: an intrinsic anisotropy term, with coupling constant $K_m$ (which can be any real constant) and a Zeeman term representing the interaction of $\vec{m}$ with a constant external applied magnetic field

The DMI energy is determined by three constant vectors $\vec{d}_i$, $i=1,2,3$.
Having restricted attention to translation invariant fields, only $\vec{d}_1$, $\vec{d}_2$ are relevant.
The overall
DMI strength $\mathcal{D}$ is set by choosing the longer of these two vectors, without loss of generality $\vec{d}_1$, to have length $1$.
We will consider three different choices of the DMI vectors $\vec{d}_1,\vec{d}_2$.
The most standard choice, deriving from Dresselhaus spin-obit coupling, yields the DMI vectors $\{\vec{d}_1=-\vec{e}_1, \, \vec{d}_2=-\vec{e}_2\}$, and the corresponding DMI term is
$$
    \sum_{i}\vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) = \vec{n} \cdot \left( \vec{\nabla} \times \vec{n} \right) = \Lambda_{xz}^{(y)}-\Lambda_{yz}^{(x)},
$$
where we have introduced the Lifshitz invariants,
$$
    \Lambda_{ij}^{(k)} = n_i \frac{\partial n_j}{\partial x_k} - n_j \frac{\partial n_i}{\partial x_k}.
$$
Rashba spin orbit coupling corresponds to the DMI vectors $\{\vec{d}_1=\vec{e}_2, \, \vec{d}_2=-\vec{e}_1\}$, which gives
$$
    \sum_{i}\vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) = n_3\left( \vec{\nabla}\cdot \vec{n}\right) - \vec{n}\cdot \vec{\nabla}n_3 = \Lambda_{zx}^{(x)}-\Lambda_{yz}^{(y)}.
$$
The final DMI term we consider arises from the vectors $\{\vec{d}_1=\vec{e}_1, \, \vec{d}_2=-\vec{e}_2\}$ and corresponds to tetragonal compounds with $D_{2d}$ symmetry (such as Heusler compounds),
$$
    \sum_{i}\vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) = \Lambda_{xz}^{(y)}+\Lambda_{yz}^{(x)}.
$$
In the absence of $E_{\textup{DDI}}$ all these choices are equivalent up to field redefinitions. That is,
$$
    \begin{split}
        E_\textup{Dresselhaus}(n_1,n_2,n_3)=E_\textup{Rashba}(n_2,-n_1,n_3),\\
        E_\textup{Heusler}(n_1,n_2,n_3)=E_\textup{Rashba}(-n_2,-n_1,n_3).
    \end{split}
$$
The Rashba DMI term favours skyrmions of N\'eel type, that is,
$$
    \vec{n}_{\textup{N\'eel}}(r,\theta)=(\sin f(r)\cos\theta,\sin f(r)\sin\theta,\cos f(r)),
$$
where $f(r)$ decreases monotonically from $\pi$ to $0$.
Note that this field has degree $N_{\textup{Sk}}=-1$, so we call it a skyrmion, following the standard convention in the magnetism literature.
It follows that the Dresselhaus DMI term favours Bloch skyrmions
$$
    \vec{n}_{\textup{Bloch}}(r,\theta)=(-\sin f(r)\sin\theta,\sin f(r)\cos\theta,\cos f(r)),
$$
while the Heusler DMI term favours *antiskyrmions* of the form
$$
    \vec{n}_{\textup{Heusler}}(r,\theta)=(-\sin f(r)\sin\theta,-\sin f(r)\cos\theta,\cos f(r)),
$$
and that these three solutions are exactly degenerate in energy without the DDI accounted for.

### Demagnetization

In this section we define the magnetostatic self-energy, or dipole-dipole interaction energy, $E_{\textup{DDI}}$ and compute its first variation.
The magnetic field induced by an isolated dipole of moment $\vec{m}$ at $\vec{0}$ is
$$
    \vec{B} = - \frac{\mu_0}{4\pi r^3} \left( \vec{m} - 3 \frac{\vec{m}\cdot \vec{x}}{r^2}\vec{x}\right), 
$$
where $\mu_0$ is the permeability of free space.
This is often referred to as the *demagnetizing* field and can be expressed as the gradient of a magnetic potential $\psi:\mathbb{R}^3\rightarrow\mathbb{R}$, that is,
$$
    \vec{B} = -\vec{\nabla}\psi
$$
where
$$
    \psi = -\mu_0 \vec{m}\cdot \vec{\nabla}\left( \frac{1}{4\pi r}\right).
$$

Consider now a continuous distribution of magnetic dipole density $\vec{m}: \Omega \rightarrow \mathbb{R}^3$, where $\Omega \subseteq \mathbb{R}^3$ is some domain.
The magnetic field it induces, at a point $\vec{x}\in\mathbb{R}^3$, is given by integrating the field induced at $\vec{x}$ by $\vec{m}(\vec{y})$ at $\vec{y}\in\Omega$ over $\vec{y}\in\Omega$: 
$$
    \vec{B}(\vec{x}) = - \frac{\mu_0}{4\pi} \int_\Omega \textup{d}^3\vec{y} \frac{1}{|\vec{x}-\vec{y}|^3} \left\{ \vec{m}(\vec{y}) - 3 \frac{\vec{m}(\vec{y}) \cdot (\vec{x}-\vec{y})}{|\vec{x}-\vec{y}|^2} (\vec{x}-\vec{y}) \right\}.
$$
Again, this is the gradient of the magnetic potential $\psi: \mathbb{R}^3 \rightarrow \mathbb{R}$,
$$
    \psi(\vec{x}) = -\mu_0 \int_\Omega \textup{d}^3\vec{y} \, \left\{\vec{m}(\vec{y}) \cdot \vec{\nabla}_x \left( \frac{1}{4\pi|\vec{x}-\vec{y}|} \right) \right\}.
$$

We now note that
$$
    G(\vec{x},\vec{y})=\frac{1}{4\pi|\vec{x}-\vec{y}|}
$$
is the Green's function for the Laplacian $\Delta = -\nabla^2$ on $\mathbb{R}^3$, that is
$$
    \Delta_x G(\vec{x},\vec{y}) = \delta(\vec{x}-\vec{y}).
$$
We see that the magnetic potential $\psi$ satisfies the PDE
$$
    \Delta\psi = -\nabla^2\psi = \mu_0 \rho, \quad  \rho = -(\vec{\nabla}\cdot\vec{m}),
$$
that is, Poisson's equation with source $\rho=-(\vec{\nabla}\cdot\vec{m})$. Up to a multiplicative constant, the magnetic field induced by the dipole distribution $\vec{m}$ coincides, therefore, with the magnetic field induced by the charge distribution $-(\vec{\nabla}\cdot\vec{m})$.
Roughly speaking, we may think of $-(\vec{\nabla}\cdot\vec{m})$ as ``magnetic charge density''.

The interaction energy of a pair of magnetic dipoles $\vec{m}^{(1)}$, $\vec{m}^{(2)}$, is $-\vec{m}^{(1)}\cdot\vec{B}^{(2)}$, where $\vec{B}^{(2)}$ is the magnetic field induced by $\vec{m}^{(2)}$ at the position of $\vec{m}^{(1)}$. Hence, the total dipole-dipole interaction energy of a continuous dipole density distribution is the magnetostatic energy
$$
    E_{\textup{DDI}} = -\frac{1}{2} \int_\Omega \textup{d}^3\vec{x} \, \left(\vec{B}(\vec{x}) \cdot \vec{m}(\vec{x}) \right),
$$
where $\vec{B}$ is the demagnetizing field induced by the magnetization $\vec{m}$.
(The factor of $1/2$ arises since we sum over dipole pairs.)
It is useful to rewrite this as a functional of the magnetic potential $\psi$:
$$
\begin{align*}
    E_{\textup{DDI}} = \, & \frac{1}{2} \int_\Omega \textup{d}^3\vec{x} \, \vec{m} \cdot \vec{\nabla}\psi -\frac{1}{2} \int_\Omega \textup{d}^3\vec{x} \, \left( \vec{\nabla} \cdot \vec{m} \right) \psi + \frac{1}{2} \oint_{\partial\Omega} \textup{d}\vec{s} \cdot \left( \psi\vec{m} \right) \\
    = \, & \frac{1}{2\mu_0} \int_\Omega \textup{d}^3\vec{x} \, \psi \Delta\psi + \frac{1}{2} \oint_{\partial\Omega} \textup{d}\vec{s} \cdot \left( \psi\vec{m} \right),
\end{align*}
$$
by the Divergence Theorem.
We will use this formula only in situations where the boundary conditions ensure that the boundary term vanishes.
In this case, $E_{\textup{DDI}}$ coincides with the magnetostatic self-energy of the "magnetic charge" distribution $-(\vec{\nabla}\cdot\vec{m})$.

We now specialize to the case of immediate interest: the dipole-dipole interaction energy of a chiral ferromagnet in a translation invariant configuration.
The dipole field has constant length $M_s$, so $\vec{m}=M_s\vec{n}$ where $\vec{n}$ is valued on the unit sphere.
Further, we impose translation invariance in the direction $\vec{e}_3=(0,0,1)$, so $\vec{n}$ is independent of $x_3$.
The boundary conditions require some care.
We assume that some anisotropy in the system (either intrinsic or generated by an external applied magnetic field) imposes an energetic preference for the dipole orientation $\vec{n}=\vec{e}_3$, and consider fields $\vec{n}:\mathbb{R}^2\rightarrow S^2$ which have compact support in the sense that there exists $R_0>0$ such that, for all $r:=|(x_1,x_2)|\geq R_0$, $\vec{n}(x_1,x_2)=\vec{e}_3$.
Since the field $\vec{n}$ is translation invariant, the total dipole interaction energy either vanishes (for example, if $\vec{n}$ is constant) or diverges.
The energy per unit length (in the $\vec{e}_3$ direction) may be finite however, and this coincides with the total energy of the slab $\Omega=\mathbb{R}^2\times[0,1]$, which we compute as the limit of the energy of the thick disk
$\Omega_R=\{\vec{x}\: :\: x_1^2+x_2^2\leq R^2,\: 0\leq x_3\leq 1\}$ as $R\rightarrow\infty$.
Note that, in this case, the boundary term vanishes identically for all $R>R_0$: the flux of $\psi\vec{m}$ through the cylindrical wall vanishes since $\vec{m}$ has no normal component on this part of the boundary, and the flux through the top disk (at $x_3=1$) is exactly canceled by the flux through the bottom disk (at $x_3=0$) since $\psi\vec{m}$ is translation invariant.
Hence
$$
    E_{\textup{DDI}}=\frac{1}{2\mu_0}\int_{\mathbb{R}^2}\textup{d}^2x\, \psi\Delta\psi,
$$
where $\Delta=-\partial_1^2-\partial_2^2$ is the Laplacian on $\mathbb{R}^2$, and $\psi$ satisfies
$$
    \Delta\psi=-\mu_0 M_s\left(\frac{\partial n_1}{\partial x_1}+\frac{\partial n_2}{\partial x_2}\right).
$$

Consider the large $r$ behaviour of the magnetic potential $\psi:\mathbb{R}^2\rightarrow\mathbb{R}$.
Any solution of the Poisson equation $\Delta\psi=\mu_0\rho$ on the plane has a multipole expansion
$$
\psi=-\frac{q\mu_0}{2\pi}\log r +O(r^{-1})
$$
where $q=\int_{\mathbb{R}^2}\textup{d}^2x\,\rho$ is the total ``charge''.
In general, such functions are logarithmically unbounded.
In our case
$$
\begin{align*}
    q=\int_{\mathbb{R}^2}\textup{d}^2x\,\rho = -\int_{\mathbb{R}^2}\textup{d}^2x\, (\vec{\nabla} \cdot \vec{m}) = 0,
\end{align*}
$$
and we conclude that $\psi$ is (at least) $1/r$ localized.

### Variation of the Magnetostatic Energy

Let us consider an energy and length rescaling with $E=\frac{J^2}{\mathcal{D}}\hat{E}$ and $x=\frac{J}{\mathcal{D}}\hat{x}$.
Further, let us also consider the rescaling of the magnetic potential $\psi = \frac{\mathcal{D}}{M_s} \hat\psi$.
Then the dimensionless energy $\hat{E}$ is (having dropped all $\hat{}$ decorations) 
$$
    E = \int_{\mathbb{R}^3} \left\{ \frac{1}{2}|\textup{d}\vec{n}|^2 + \vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) + K(1-n_3^2) + h(1-n_3)  + \frac{1}{2}\vec{n} \cdot \vec{\nabla}\psi \right\} \textup{d}^3x,
$$
where the magnetic potential satisfies the dimensionless Poisson equation
$$
    \Delta \psi = -\mu \vec{\nabla} \cdot \vec{n}.
$$
The dimensionless coupling constants are
$$
    K = \frac{K_mJ}{\mathcal{D}^2}, \quad h = \frac{M_s B_\textup{ext}J}{\mathcal{D}^2}, \quad \mu = \frac{\mu_0JM_s^2}{\mathcal{D}^2}.
$$

We now return to the case of interest where we have translation invariance in the $\vec{e}_3$ direction.
Before, we were considering the energy per unit length in the translation invariant direction.
So, our domain was $\Omega=\mathbb{R}^2\times[0,1]$, which now becomes $\Omega'=\mathbb{R}^2\times[0,t]$ where $t=L_0^{-1}$ is the thickness of the slab under the rescaling.
To maintain adimensionality, let us consider the energy per unit thickness $E/t$.
Therefore, the adimensional energy functional we are really considering is
$$
    E = \int_{\mathbb{R}^2} \left\{ \frac{1}{2}|\textup{d}\vec{n}|^2 + \vec{d}_i\cdot(\vec{n}\times\partial_i\vec{n}) + K(1-n_3^2) + h(1-n_3) + \frac{1}{2}\vec{n} \cdot \vec{\nabla}\psi \right\} \textup{d}^2x.
$$

We seek fields $\vec{n}:\R^2\rightarrow S^2$ which (at least locally) minimize $E$ so, for all smooth variations $\vec{n}_t$ of $\vec{n}=\vec{n}_0$ through fields of compact support, we require that
$$
    \frac{\textup{d}}{\textup{d}t}\bigg|_{t=0}E(\vec{n}_t)=0.
$$
This condition is equivalent to the Euler-Lagrange equations for $E$, which we now derive.

Let $\vec{\epsilon}=\partial_t\vec{n}_t|_{t=0}$, the generator of the variation $\vec{n}_t$, and note that $\vec{\epsilon}\cdot\vec{n}= 0$ everywhere since $|\vec{n}|=1$.
The induced variations of $E_{\textup{exch}}$, $E_{\textup{DMI}}$ and
$E_{\textup{pot}}$ are easily computed:
$$
\begin{align*}
    \frac{\textup{d}}{\textup{d}t}\bigg|_{t=0}E_{\textup{exch}}(\vec{n}_t)&=&\int_{\R^2} \textup{d}^2x\, \vec{\epsilon}\cdot\Delta\vec{n}, \\
    \frac{\textup{d}}{\textup{d}t}\bigg|_{t=0}E_{\textup{DMI}}(\vec{n}_t)&=&\int_{\R^2}\textup{d}^2x\,\vec{\epsilon}\cdot(-2\vec{d}_i\times\partial_i\vec{n}), \\
    \frac{\textup{d}}{\textup{d}t}\bigg|_{t=0}E_{\textup{pot}}(\vec{n}_t)&=&\int_{\R^2}\textup{d}^2x\, \vec{\epsilon}\cdot(0,0,-h-2Kn_3).
\end{align*}
$$
We will also need the variation of $E_{\textup{DDI}}$ which is rather more subtle. 

Denote by $\psi_t$ the unique solution of with source $-\vec{\nabla} \cdot \vec{n}_t$ decaying to $0$ at infinity, and $\dot\psi=\partial_t\psi_t|_{t=0}$.
It is important to note that, while $\vec{\epsilon}$ has compact support, neither $\psi=\psi_0$ nor $\dot\psi$ do: as argued above, they are $1/r$ localized. Care must be taken with boundary terms when computing the variation of $E_{\textup{DDI}}$, therefore.  The variation of $E_{\textup{DDI}}$ induced by $\vec{n}_t$ is
$$
\begin{align*}
    \frac{\textup{d}\: }{\textup{d}t}\bigg|_{t=0}E_{\textup{DDI}}(\vec{n}_t) = \, &\frac{1}{2\mu} \int_{\mathbb{R}^2} \textup{d}^2x \left(\dot\psi\Delta\psi+\psi\Delta\dot\psi\right) \nonumber \\
    = \, &\frac{1}{\mu}\int_{\mathbb{R}^2}\textup{d}^2x\,\psi\Delta\dot\psi +\lim_{R\rightarrow\infty}\frac{1}{2\mu}\int_{\partial B_R(0)}(\dot\psi*\textup{d}\psi-\psi *\textup{d}\dot\psi) \nonumber \\
    = \, &\frac{1}{\mu}\int_{\mathbb{R}^2}\textup{d}^2x\,\psi\Delta\dot\psi +\lim_{R\rightarrow\infty}\frac{R}{2\mu_0}\int_0^{2\pi}(\dot\psi\psi_r-\psi\dot\psi_r) \textup{d}\theta\nonumber \\
    = \, &\frac{1}{\mu}\int_{\mathbb{R}^2}\textup{d}^2x\, \psi\Delta\dot\psi
\end{align*}
$$
since $\psi,\dot\psi=O(r^{-1})$ and $\psi_r,\dot\psi_r=O(r^{-2})$. Differentiating the Poisson equation with respect to $t$, we deduce that
$$
    \Delta\dot\psi = -\mu\vec{\nabla}\cdot\vec{\epsilon},
$$
and hence
$$
\begin{align*}
    \frac{\textup{d}\: }{\textup{d}t}\bigg|_{t=0}E_{\textup{DDI}}(\vec{n}_t)= \, & -\int_{\mathbb{R}^2}\textup{d}^2x\, \psi \left(\vec{\nabla}\cdot\vec{\epsilon}\right) \nonumber \\
    = \, & \int_{\mathbb{R}^2}\textup{d}^2x\, {\vec{\epsilon}}\cdot\vec{\nabla}\psi-\lim_{R\rightarrow\infty}\int_{\partial B_R(0)}\psi *\varepsilon \nonumber \\
    = \, & \int_{\mathbb{R}^2}\textup{d}^2x\,\vec{\epsilon}\cdot\vec{\nabla}\psi
\end{align*}
$$
by Stokes's Theorem, since $\varepsilon=\varepsilon_1 \textup{d} x_1+\varepsilon_2\textup{d} x_2$ has compact support. 

Therefore, we see that
$$
    \frac{\textup{d}}{\textup{d}t}\bigg|_{t=0}E(\vec{n}_t) = \int_{\R^2}\textup{d}^2x\,\vec{\epsilon}\cdot (\Delta\vec{n}-2\vec{d}_i\times\partial_i\vec{n} -(h+2K\vec{n}\cdot\vec{e}_3)\vec{e}_3+\vec{\nabla}\psi)
$$
and this must vanish for arbitrary $\vec{\epsilon}$ pointwise orthogonal to $\vec{n}$.
Hence, the Euler-Lagrange equation for $E$ is
$$
    \textup{grad} E(\vec{n}):=P_{\vec{n}}(\Delta\vec{n}-2\vec{d}_i\times\partial_i\vec{n}-(h+2Kn_3)\vec{e}_3+\vec{\nabla}\psi)=0,
$$
where the projector $P_{\vec{n}}(\vec{u}):=\vec{u}-(\vec{u}\cdot\vec{n})\vec{n}$ accounts for the constraint that $|\vec{n}|\equiv 1$ and $\psi$ is determined by the Poisson equation.
Our notation calls attention to the fact that $\textup{grad} E$ is formally the $L^2$ gradient of $E$, that is, the direction in field space in which $E(\vec{n})$ grows fastest.

### Numerical Implementation

The Euler-Lagrange field equations are
$$
\begin{align*}
    \frac{\delta E}{\delta \vec{n}} = \, & -\nabla^2\vec{n} - 2\vec{d}_i\times\partial_i\vec{n} - (h+2Kn_3)\vec{e}_3 + \vec{\nabla}\psi, \\
    \frac{\delta E}{\delta \psi} = \, & -\nabla^2 \psi + \mu \vec{\nabla} \cdot \vec{n},
\end{align*}
$$
and, hence, `soliton_solver` is solving the coupled system of nonlinear equations
$$
\begin{align*}
    \frac{\textup{d}^2 \vec{n}}{\textup{d}t^2} = \, & -\frac{\delta E}{\delta \vec{n}} = \nabla^2\vec{n} + 2\vec{d}_i\times\partial_i\vec{n} + (h+2Kn_3)\vec{e}_3 - \vec{\nabla}\psi, \\
    \frac{\textup{d}^2 \psi}{\textup{d}t^2} = \, & -\frac{\delta E}{\delta \psi} = \nabla^2 \psi - \mu \vec{\nabla} \cdot \vec{n},
\end{align*}
$$
where $t$ is a ficticious time.

### References

P. Leask and M. Speight, *Demagnetization in micromagnetics: Magnetostatic self-interactions of bulk chiral magnetic skyrmions*, [Phys. Rev. B **113**, 064406 (2026)](https://doi.org/10.1103/zy2n-m4tn)

## Liquid Crystal with flexoelectric depolarization

### Fields and parameters

- Director order parameter: $\vec{n} \in \mathbb{R}^3$, $[\vec{n}] = 1$.
- Electrostatic potential: $\varphi \in \mathbb{R}$, $[\varphi] = \textup{V}$
- Frank elastic constant: $K$, $[K] = \textup{N}$
- Cholestric pitch: $p$, $[p] = \textup{m}$
- Cholestric twist: $q_0$, $[q_0] = \textup{m}^{-1}$
- External electric field: $\vec{E}_{\textup{ext}}$, $[\vec{E}_{\textup{ext}}] = \textup{Vm}^{-1}$
- Vacuum permittivity: $\epsilon_0$, $[\epsilon_0] = \textup{C}^2\textup{N}^{-1}\textup{m}^{-2}$
- Dielectric anisotropy: $\Delta\epsilon$, $[\Delta\epsilon] = 1$
- Effective surface anchoring strength: $W_0$, $[W_0] = \textup{Jm}^{-3}$
- Flexoelectric polarization: $\vec{P}_f$, $[\vec{P}_f] = \textup{Cm}^{-2}$
- Piezoelectric constants: $e_1, e_3$, $[e] = \textup{Cm}^{-1}$

### The Frank-Oseen energy

The system that we wish to model is an apolar chiral liquid crystal, described by a director field $\vec{n}(\vec{x})\in \mathbb{R}P^2 \cong S^2/\mathbb{Z}_2$.
That is, the director $\vec{n}$ is a vector in $\mathbb{R}^3$ of unit length $|\vec{n}|=1$, such that $\vec{n}$ and $-\vec{n}$ describe the same state, since the director $\vec{n}$ is the average molecular alignment direction.
In liquid crystal physics terminology, the standard bend vector is
$$
    \vec{B} = -(\vec{n}\cdot\vec{\nabla})\vec{n} = \vec{n} \times (\vec{\nabla}\times\vec{n}),
$$
the standard pseudoscalar twist is
$$
    T = \vec{n} \cdot (\vec{\nabla}\times\vec{n}),
$$
and the standard splay vector is
$$
    \vec{S} = S\vec{n}, \quad S = \vec{\nabla}\cdot\vec{n}.
$$

Let us consider a liquid crystal composed of chiral molecules, with different elastic deformation costs.
Then the chirality of these molecules is characterized by some pseudoscalar $q_0$ that couples to the twist $T$.
The associated Frank-Oseen free energy, neglecting the energy cost due to saddle-splay, can be expressed as
$$
\begin{align*}
    F_{\textup{FO}} = \, & \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2}K_1 |\vec{S}|^2 + \frac{1}{2}K_2 (T+q_0)^2 + \frac{1}{2}K_3 |\vec{B}|^2 \right\}  \\
    = \, & \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{K_1}{2} (\vec{\nabla}\cdot \vec{n})^2 + \frac{K_2}{2} \left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) + \frac{2\pi}{p} \right]^2 + \frac{K_3}{2} \left[\vec{n} \times (\vec{\nabla} \times \vec{n})\right]^2\right\},
\end{align*}
$$
where $q_0=2\pi/p$ is the cholesteric twist and $p$ is the cholesteric pitch with a defined length at which the director twists by $2\pi$.
The cholesteric phase is characterized by the presence of enantiomorphy ($q_0\neq0$), and is distinguishable from the nematic phase ($q_0=0$).
The Frank elastic constants $K_1$, $K_2$, and $K_3$ determine the energy cost of splay, twist, and bend deformations, respectively.
Skyrmion solutions in nematic liquid crystal correspond to setting $q_0=0$.

Chiral liquid crystals are dielectric materials that respond to external electric fields.
This generates a corresponding coupled electric energy of the form
$$
    \mathcal{E}_{\textup{elec}}=-\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2
$$
where $\vec{E}_{\textup{ext}}$ is the external electric field, $\epsilon_0$ is the vacuum permittivity and $\Delta\epsilon$ is the dielectric anisotropy.
We will only consider the applied electric field orthogonal to topological defects in the $(x,y)$-plane, that is $\vec{E}_{\textup{ext}}=(0,0,E_z)$.

In experimental realizations, liquid crystals are placed between parallel plates with a potential difference.
This imposes boundary conditions orthogonal to the plates on the liquid crystal director field.
In particular, this can impose strong homeotropic anchoring conditions
$$
    \vec{n}(x,y,z=\pm d/2)= \vec{e}_z = (0,0,1).
$$
This can be accounted for in two dimensional systems by including the Rapini-Papoular homeotropic surface anchoring potential
$$
    \mathcal{E}_{\textup{anch}} = -\frac{1}{2}W_0 n_z^2,
$$
where $W_0$ is the effective surface anchoring strength which favors director alignment in the $z$-direction, that is $\vec{n}=\pm\vec{e}_{z}=(0,0,\pm1)$ director configurations.
This term acts to mimic homeotropic anchoring conditions at the cell surfaces of a three-dimensional system.

The Frank-Oseen free energy we are interested in, including the electric energy and homeotropic anchoring, is given by the energy functional
$$
    F_{\textup{FO}} = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{K_{1}}{2} (\vec{\nabla}\cdot \vec{n})^2 +  \frac{K_{2}}{2} \left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) \right]^2 + \frac{K_{3}}{2} \left[\vec{n} \times \vec{\nabla} \times \vec{n}\right]^2 + K_2 q_0\left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) \right] -\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2 -\frac{1}{2}W_0 n_z^2 \right\},
$$
While this model does account for an external applied electric field, it does not include the electrostatic self-interaction energy arising from the flexoelectric effect.
We will now show how to do this.

### Flexoelectric polarization

When liquid crystals possess macroscopic electric polarization $\vec{P}_f$ (where $\vec{P}_f$ is spontaneous or induced by some external, non-electric field related, factors), then they induce a linear-in-field energy contribution.
One such source of macroscopic electric polarization generation is related to orientational distortions in liquid crystals.
The case we consider here is molecules with permanent dipole moments.
This leads to piezoelectric effects and generates a dipolar piezoelectric-like polarization.
However, piezoelectricity is due to uniform strain, whereas this polarization is caused by the mechanical curvature, or flexion, of the director field $\vec{n}$, and is called *flexoelectric*.
A formal theory of these flexoelectric effects was developed by Meyer.
This can be expressed as
$$
    \vec{P}_f = e_1 \left[ (\vec{\nabla} \cdot \vec{n}) \vec{n} \right] + e_3 \left[ \vec{n} \times (\vec{\nabla} \times \vec{n}) \right],
$$
where $e_1$ and $e_3$ are, respectively, the piezoelectric constants for the splay and bend of the molecules.
These coefficients are material dependent and can be measured directly via the electric current produced by the periodic mechanical flexing of the liquid crystals bounding surfaces.

Analogous to demagnetization in chiral magnets, the flexoelectric polarization produces internal sources of electric fields i.e. it induces an electric dipole moment $\vec{p}$, where $\vec{p}=\vec{P}_f$.
In fact, it generates a continuous electric dipole moment distribution $\vec{P}_f: \mathbb{R}^2 \rightarrow \mathbb{R}^3$.
The electric potential $\varphi:\mathbb{R}^2\rightarrow\mathbb{R}$ associated to this continuous dipole distribution induces an internal electric field $\vec{E} = -\vec{\nabla}\varphi$. 
It satisfies a Poisson equation for electrostatics
$$
    \Delta\varphi = -\nabla^2\varphi = \frac{1}{\epsilon_0} \rho, \quad \rho = -(\vec{\nabla}\cdot\vec{P}_f),
$$
where $\rho_e$ is the electric charge density and the Laplacian is $\Delta = -\nabla^2$ on $\mathbb{R}^2$.

Using the definition of the electric potential, we see that Gauss' law is
$$
    \vec{\nabla} \cdot \vec{E} = \frac{\rho}{\epsilon_0}, \quad \rho = -(\vec{\nabla}\cdot\vec{P}_f)
$$
where $\rho$ is the electric charge density.
The electric field induced by the dipole distribution $\vec{P}_f$ coincides, therefore, with the electric field induced by the charge distribution $-(\nabla\cdot \vec{P}_f)$.
Hence, we may think of $-(\nabla\cdot\vec{P}_f)$ as an electric charge density.
Furthermore, we can write
$$
    \vec{\nabla} \cdot \left( \epsilon_0 \vec{E} + \vec{P}_f \right) = \vec{\nabla} \cdot\vec{D} = 0,
$$
where $\vec{D}$ is the electric displacement field.
Hence, there is no space charge.

Suppose we have a pair of electric dipole moments $\vec{P}_f^{(1)}$ and $\vec{P}_f^{(2)}$.
Their interaction energy is
$$
    E_{\textup{int}} = -\vec{P}_f^{(1)}\cdot\vec{E}^{(2)} = -\vec{P}_f^{(2)}\cdot\vec{E}^{(1)},
$$
where $\vec{E}^{(2)}$ is the electric field induced by the polarization $\vec{P}_f^{(2)}$ at the position of $\vec{P}_f^{(1)}$, and vice versa.
Therefore, the flexoelectric energy coincides with the energy of a continuous dipole density distribution, which is
$$
    F_{\textup{flexo}} = -\frac{1}{2} \int_{\mathbb{R}^2} \textup{d}^3\vec{x} \, \vec{E}(\vec{x}) \cdot \vec{P}_f(\vec{x}),
$$
where $\vec{E}$ is the induced electric field.
We will later want to compute the variation of the flexoelectric energy with respect to the director field $\vec{n}$.
For this reason, it proves more useful to express the flexoelectric energy in terms of the scalar electric potential $\varphi$,
$$
    F_{\textup{flexo}} = \frac{1}{2} \int_\Omega \textup{d}^3\vec{x} \, \vec{P}_f \cdot \vec{\nabla}\varphi = \frac{\epsilon_0}{2} \int_{\mathbb{R}^3} \textup{d}^3\vec{x} \, \varphi \Delta\varphi + \frac{1}{2} \oint_{\partial\Omega} \textup{d}\vec{s} \cdot \left( \varphi\vec{P}_f \right),
$$
by the Divergence Theorem.
We will restrict ourselves to situations where the boundary conditions ensure the boundary term vanishes.
In this case, the flexoelectric energy $F_{\textup{flexo}}$ coincides with the electrostatic self-energy of the charge distribution $-(\nabla\cdot\vec{P}_f)$.
To see this, we use the general identity $\varphi\Delta\varphi = \vec{\nabla}\varphi\cdot\vec{\nabla}\varphi - \vec{\nabla}\cdot \left( \varphi\vec{\nabla}\varphi \right)$ and the divergence theorem to express the flexoelectric energy as
$$
    F_{\textup{flexo}} = \frac{\epsilon_0}{2}\int_{\mathbb{R}^3} \textup{d}^3\vec{x} \, |\vec{\nabla}\varphi|^2 = \frac{\epsilon_0}{2}\int_{\mathbb{R}^3} \textup{d}^3\vec{x} \, |\vec{E}|^2.
$$

We now detail the cases of interest - the flexoelectric self-interaction energy of a translation invariant skyrmion in a chiral liquid crystal.

### Variation of the flexoelectric energy

So far, we have shown how to include the electrostatic self-energy and compute the electric scalar potential $\varphi$ by solving Poisson's equation for fixed director field configuration $\vec{n}$.
However, we need to compute the back-reaction of the self-induced electric field $\vec{E}$ on the director field $\vec{n}$.
To do this, we need to calculate the first variation of the flexoelectric energy $F_{\textup{flexo}}(\vec{n})$ with respect to the director field $\vec{n}$.

Before proceeding with the variation of the flexoelectric energy, we opt to work in dimensionless units.
This will also make numerical simulations more palatable.
Let us consider an energy and length rescaling with $E=E_0\hat{E}$ and $x=L_0\hat{x}$.
We choose to set our length and energy scales as
$$
    L_0 = \frac{1}{q_0}\frac{K_1}{K_2}, \quad E_0 = \frac{1}{q_0}\frac{K_1^2}{K_2}.
$$
Then the rescaled energy is 
$$
    \hat{F}_{\textup{FFO}} = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} (\vec{\nabla}\cdot \vec{n})^2 + \frac{1}{2} \frac{K_2}{K_1} \left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) \right]^2 + \frac{1}{2} \frac{K_3}{K_1} (\vec{n} \times \vec{\nabla} \times \vec{n})^2 + \left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) \right] + \frac{1}{q_0^2} \frac{K_1}{K_2^2} \left[ -\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2 -\frac{1}{2}W_0 n_z^2 \right] \right\} + \hat{F}_{\textup{flexo}},
$$
where the rescaled flexoelectric energy is determined to be
$$
    \hat{F}_{\textup{flexo}} = \frac{\epsilon}{2} \int_{\mathbb{R}^2} \hat\varphi \Delta_{\hat{x}} \hat\varphi \, \textup{d}^3\hat{x}, \quad \Delta_{\hat{x}} \hat\varphi = -\frac{1}{\epsilon} \vec{\nabla}_{\hat{x}} \cdot \vec{P}.
$$
Here, we have expressed the polarization $\vec{P}_f$ in terms of a rescaled polarization $\vec{P}$, where
$$
    \vec{P}_f = \frac{e_1}{L_0} \vec{P}, \quad \vec{P} =  (\vec{\nabla}_{\hat{x}} \cdot \vec{n}) \vec{n}  + \frac{e_3}{e_1} [\vec{n} \times (\vec{\nabla}_{\hat{x}} \times \vec{n})],
$$
and introduced the dimensionless vacuum electric permittivity
$$
    \epsilon = \frac{K_1\epsilon_0}{e_1^2}.
$$

We note that the flexoelectric self-energy is scale invariant in two dimensions and is thus unable to provide stability against spatial rescalings.
Whereas, in comparison with chiral ferromagnets, the magnetostatic self-energy there can stabilize skyrmions as it behaves like a potential under coordinate rescalings.

Let $\vec{n}_t$ be a smooth variation of $\vec{n}=\vec{n}_0$ through fields of compact support and define
$\delta\vec{n}=\partial_t\vec{n}_t|_{t=0}$.
Denote by $\varphi_t$ the associated unique solution of the Poisson equation with source $-\frac{1}{\epsilon} \vec{\nabla}\cdot \vec{P}_t$ decaying to $0$ at infinity, and $\dot\varphi=\partial_t\varphi_t|_{t=0}$.
It is important to note that, while $\delta\vec{n}$ has compact support, neither $\varphi=\varphi_0$ nor $\dot\varphi$ do: as they are $1/r$ localized.
The variation of $F_{\textup{flexo}}$ induced by $\vec{n}_t$ is found to be given by
$$
    \frac{\textup{d}}{\textup{d}t}\bigg|_{t=0}F_{\textup{flexo}}(\vec{n}_t) = \int_{\mathbb{R}^2}\textup{d}^2x\, (\textup{grad}_{\vec{n}}\,F_{\textup{flexo}}) \cdot \delta\vec{n},
$$
where the corresponding gradient is
$$
    \textup{grad}_{\vec{n}}\,F_{\textup{flexo}} = \frac{e_3}{e_1} \left[  \left( (\vec{\nabla} \times \vec{n}) \times \vec{\nabla}\varphi \right) + \left( \vec{\nabla} \times (\vec{\nabla}\varphi\times\vec{n}) \right) \right] - \vec{\nabla}(\vec{\nabla}\varphi \cdot \vec{n}) + (\vec{\nabla}\cdot\vec{n})\vec{\nabla}\varphi.
$$

### Relation to chiral magnets

The stability of two-dimensional skyrmions in chiral liquid crystals arises from the same mechanism responsible for the existence of skyrmions in chiral ferromagnetic systems.
This is due to the chiral interactions imposed by the handedness of the system.
Consider the one-constant approximation where the bend, splay and twist constants are all equal ($K_i=K$).
This corresponds to an apolar, chiral liquid crystal.
For such liquid crystals in an applied electric field $\vec{E}_{\textup{ext}}=(0,0,E_z)$, the free-energy in the one-constant approximation can be reduced to the following expression
$$
    F_{\textup{FFO}} = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} (\nabla \vec{n})^2 + \left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) \right] + \frac{1}{q_0^2} \frac{1}{K} \left[ -\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2 -\frac{1}{2}W_0 n_z^2 \right] + \frac{\epsilon}{2}\varphi\Delta\varphi \right\},
$$
where we have used the identity
$$
    \left( \nabla\vec{n}\right)^2 = \left( \vec{\nabla}\cdot\vec{n}\right)^2 + \left[ \vec{n} \cdot (\vec{\nabla} \times \vec{n})\right]^2 + \left[\vec{n} \times (\vec{\nabla} \times \vec{n}) \right]^2 + \vec{\nabla} \cdot \left[ (\vec{n}\cdot\vec{\nabla})\vec{n} - (\vec{\nabla}\cdot \vec{n})\vec{n} \right]
$$
which holds for any unit vector $\vec{n}$.
This is the energy density of a chiral ferromagnet in the absence of an external magnetic field with the Dzyaloshinskii-Moriya interaction (DMI) arising from the Dresselhaus spin-orbit coupling (SOC).
The dielectric anisotropy energy in liquid crystals plays the same role as uniaxial anisotropy in chiral magnets.

### Numerical implementation

Our interests lie in computing the self-induced flexoelectric polarization of topological solitons in the above system.
For simplicity, the one constant approximation is implemented.
Topological solitons in this model are minimizers of the adimensional flexoelectric Frank-Oseen free energy 
$$
    E_{\textup{FFO}} = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} (\nabla \vec{n})^2 + \left[ \vec{n} \cdot (\vec{\nabla}\times\vec{n}) \right] + \frac{1}{q_0^2} \frac{1}{K} \left[ -\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2 -\frac{1}{2}W_0 n_z^2 \right] + \frac{1}{2}\vec{P}\cdot\vec{\nabla}\varphi\right\},
$$
where the electric scalar potential $\varphi$ is subject to the constraint $\Delta\varphi = -\frac{1}{\epsilon} \vec{\nabla}\cdot\vec{P}$.
The adimensional polarization is
$$
    \vec{P} =  (\vec{\nabla} \cdot \vec{n}) \vec{n}  + \frac{e_3}{e_1} \left[\vec{n} \times (\vec{\nabla} \times \vec{n})\right],
$$
and the divergence of the polarization $\vec{P}$ is
$$
    \vec{\nabla}\cdot\vec{P} = \frac{e_3}{e_1} \left\{ (\vec{\nabla}\times\vec{n})^2 - \vec{n}\cdot[\vec{\nabla}(\vec{\nabla}\cdot\vec{n})] + \vec{n}\cdot \nabla^2\vec{n} + (\vec{\nabla}\cdot\vec{n})^2 + \vec{n}\cdot[\vec{\nabla}(\vec{\nabla}\cdot\vec{n})] \right\}.
$$
We develop a method to find director fields $\vec{n}\in\mathbb{R}P^2$ that simultaneously minimize the flexoelectric Frank-Oseen energy and solve the electric potential constraint.
The associated field equations for the arrested Newton flow algorithm are
$$
\begin{align*}
    \frac{\delta F}{\delta \vec{n}} = \, & -\nabla^2 \vec{n} + \vec{\nabla} \times \vec{n} - \frac{1}{q_0^2 K} \left[ \epsilon_0 \Delta\epsilon (\vec{E}_{\textup{ext}} \cdot \vec{n}) \vec{E}_{\textup{ext}} + W_0 (\vec{e}_z \cdot \vec{n}) \vec{e}_z \right] + \textup{grad}_{\vec{n}}\,F_{\textup{flexo}} \\
    \frac{\delta F}{\delta \varphi} = \, & -\nabla^2 \varphi + \frac{1}{\epsilon} \vec{\nabla}\cdot\vec{P}
\end{align*}
$$
Therefore, `soliton_solver` is solving the system
$$
\begin{align*}
    \frac{\textup{d}^2 \vec{n}}{\textup{d}t^2} = \, & -\frac{\delta F}{\delta \vec{n}}, \\
    \frac{\textup{d}^2 \varphi}{\textup{d}t^2} = \, & -\frac{\delta F}{\delta \varphi}.
\end{align*}
$$

### Splay and bend favored Neel skyrmions

What happens if we now consider liquid crystals which prefer splay and bend, opposed to twist.
Let us remain in the one constant approximation.
Then the Frank-Oseen free energy takes the form
$$
\begin{align*}
    F = \, & \frac{K}{2} \int_{\mathbb{R}^2} \textup{d}^2x \left\{ (\vec{S}+\vec{S}_0)^2 + T^2 + (\vec{B}+\vec{B}_0)^2 \right\} \\
    = \, & \frac{K}{2} \int_{\mathbb{R}^2} \textup{d}^2x  \left\{ (\vec{\nabla}\cdot\vec{n})^2 + 2 \vec{S}_0\cdot\vec{n}(\vec{\nabla}\cdot\vec{n}) + [\vec{n} \cdot (\vec{\nabla} \times \vec{n})]^2 + [\vec{n} \times (\vec{\nabla} \times \vec{n})]^2 - 2 \vec{B}_0\cdot[(\vec{n}\cdot\vec{\nabla})\vec{n}] +\textup{const.}\right\}.
\end{align*}
$$
If we choose $\vec{S}_0=\vec{B}_0=q_0\vec{e}_z$, then the model reduces to that of the chiral magnet with the DMI term arising from the Rashba SOC.
That is, the free energy becomes
$$
    F = \int_{\mathbb{R}^2} \textup{d}^2x  \left\{ \frac{K}{2}(\nabla\vec{n})^2 + Kq_0 [n_z(\vec{\nabla}\cdot \vec{n}) - \vec{n}\cdot \vec{\nabla}n_z] -\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2 -\frac{1}{2}W_0 n_z^2  \right\}.
$$
The factor of $q_0$ was chosen for convenience as we can pick the same energy and length scales as the twist favored model.
It also allows us to compare splay-bend favored Neel skyrmions with the twist favored Bloch skyrmions.

Let us now include the electrostatic self-energy, and employ the same length $L_0=1/q_0 $ and energy $E_0=K/q_0$ scales as before.
Then, in the translation invariant case, the normalized free energy of this splay-bend favored liquid crystal model becomes
$$
    F = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2} (\nabla \vec{n})^2 + \left[ n_z(\vec{\nabla}\cdot \vec{n}) - \vec{n}\cdot \vec{\nabla}n_z \right] + \frac{1}{q_0^2} \frac{1}{K} \left[ -\frac{\epsilon_0 \Delta\epsilon}{2} (\vec{E}_{\textup{ext}} \cdot \vec{n})^2 -\frac{1}{2}W_0 n_z^2 \right] + \frac{\epsilon}{2}\varphi\Delta\varphi \right\}.
$$
It is well-known that the Rashba DMI term prefers Neel hedgehog skyrmions, given by the ansatz
$$
    \vec{n}_{\textup{Neel}}(r,\theta) = \sin f(r) \vec{e}_r + \cos f(r) \vec{e}_z.
$$
The self-induced polarization, coming from the Neel ansatz, is
$$
    \vec{P}_{\textup{Neel}} = \left[\frac{1}{r}\sin^2f(r) + \left(1-\frac{e_3}{e_1}\right)\frac{1}{2}\sin2f(r) \frac{\textup{d}f}{\textup{d}r}\right] \vec{e}_r + \left[ \frac{1}{2r}\sin2f(r) + \left( \cos^2f(r) + \frac{e_3}{e_1}\sin^2f(r) \right) \frac{\textup{d}f}{\textup{d}r} \right] \vec{e}_z.
$$
Unlike the Bloch polarization, the Neel polarization picks up an out-of-plane component.
The divergence of this polarization is also non-zero.
We note that if the flexoelectric coefficients are equal, $e_1=e_3$, then the divergence of the polarization for both Bloch and Neel ans\"atze are the same,
$$
    \vec{\nabla}\cdot\vec{P}_{\textup{Bloch}} = \vec{\nabla}\cdot\vec{P}_{\textup{Neel}} = \frac{1}{r}\frac{\textup{d}f}{\textup{d}r} \sin2f(r).
$$
So, they will generate the same electrostatic potential and, thus, they will be energy degenerate for equal flexoelectric coefficients.
For unequal flexoelectric coefficients, $e_1 \neq e_3$, the associated self-induced polarizations yield different electric scalar potentials and, hence, distinct skyrmions. 

### References

P. Leask, *Topological transition from a hopfion to a toron via flexoelectric self-polarization in chiral liquid crystals*, [Phys. Rev. Res. **7**, 043001 (2025)](https://doi.org/10.1103/gy6m-m7ck)

## Ferromagnetic superconductor

### The free energy

The model we are interested in is that of an isotropic ferromagnetic superconductor.
It consists of a superconducting order parameter which is a single complex field $\psi\in\mathbb{C}$, where $|\psi|^2$ is a measure of local density of Cooper pairs, an electromagnetic gauge field $\vec{A}=(A_x,A_y,A_z)\in\mathbb{R}^3$, and a magnetization order parameter $\vec{m}=(m_x,m_y,m_z)\in \mathbb{R}^3$.
We are interested in translation invariant solutions, with the translation invariance imposed in the $z$-direction.
Then,  associated to the gauge field is the magnetic field
$$
    \vec{B}=\vec{\nabla}\times\vec{A}=(\partial_y A_z,-\partial_x A_z,\partial_x A_y-\partial_y A_x).
$$
The free energy functional of this system consists of three parts
$$
    F[\psi,\vec{A},\vec{m}] = F_{\textup{sc}}[\psi,\vec{A}] + F_{\textup{mag}}[\vec{m}] + F_{\textup{int}}[\psi,\vec{A},\vec{m}].
$$

The first part is the free energy functional for the superconductor $(\psi,\vec{A})$, which is given by the Ginzburg--Landau free energy density
$$
    \mathcal{F}_{\textup{sc}}[\psi,\vec{A}] = \frac{1}{2}|\vec{D}\psi|^2 + \frac{1}{2}|\vec{B}|^2  + \frac{a(T)}{2}|\psi|^2 + \frac{b}{4}|\psi|^4,
$$
where $\vec{D}\psi=\vec{\nabla}\psi + iq\vec{A}\psi$ is the gauge covariant derivative and $a(T)=a_0(T-T_c)/T_c$.
The critical temperature for superconductivity is $T_c$ and $q\sim2e$ is the effective charge of a Cooper pair.
For the magnetization we consider an isotropic ferromagnet in the absence of an applied magnetic field.
In the simplest approximation, the free energy is given by
$$
    \mathcal{F}_{\textup{mag}}[\vec{m}] =  \frac{\alpha(T)}{2}|\vec{m}|^2 + \frac{\beta}{4}|\vec{m}|^4 + \frac{1}{2}|\nabla\vec{m}|^2,
$$
where $\alpha(T)=\alpha_0(T-T_m)/T_m$ and $T_m$ is the Curie temperature.
We are interested in superconducting vortices in the presence of magnetic spin textures such as skyrmions.
So, in this model, we consider a regime where  we can restrict the magnetization to be of fixed length, that is $\vec{m}\in \mathbb{S}^2_{m_0} \subset\mathbb{R}^3$.

There are two main interactions of the superconducting state $(\psi,\vec{A})$ with the magnetization $\vec{m}$.
One is via the direct effects of spin-flip scattering of conduction electrons with the magnetic moments and conduction-electron polarization.
The second is an indirect interaction which arises from the coupling of the order parameter $\psi$ to the electromagnetic gauge field $\vec{A}$, and the coupling of the magnetic field $\vec{B}=\vec{\nabla}\times\vec{A}$ to the magnetization $\vec{m}$ through the Zeeman interaction
$$
    \mathcal{F}_{\textup{Zeeman}}[\vec{A},\vec{m}] = - \vec{m} \cdot (\vec{\nabla}\times\vec{A}).
$$
We first begin by ignoring the effects of spin-flip scattering and consider the interaction energy functional defined by
$$
    F_{\textup{int}}[\psi,\vec{A},\vec{m}] = F_{\textup{Zeeman}}[\vec{A},\vec{m}].
$$

In the present model, we first restrict attention to the minimal self-consistent model in which the dominant coupling between the magnetic and superconducting sectors arises through the Zeeman interaction.
However, the effects of spin-flip scattering are included in the model and are detailed below.
More general magnetoelectric coupling terms, such as Lifshitz-type invariants that can arise in systems with strong spin-orbit coupling or broken inversion symmetry, are neglected in this model.

The uniform ground state configurations for the superconducting order parameter $\psi$ and the magnetization $\vec{m}$ are determined by minimizing the potential energy
$$
    \mathcal{F}_p=\frac{a}{2}|\psi|^2 + \frac{b}{4}|\psi|^4 + \frac{\alpha}{2}|\vec{m}|^2 + \frac{\beta}{4}|\vec{m}|^4.
$$
which amounts to solving the system of equations
$$
\begin{align*}
    \left.\frac{\delta \mathcal{F}_p}{\delta |\psi|}\right|_{(u,m_0)} = \, & au + bu^3 = 0, \\ \left.\frac{\delta \mathcal{F}_p}{\delta |\vec{m}|}\right|_{(u,m_0)} = \, & \alpha m_0 + \beta m_0^3  = 0.
\end{align*}
$$
This gives us the ground state configurations
$$
    u^2 = -\frac{a}{b}, \quad m_0^2 = -\frac{\alpha}{\beta}.
$$
The corresponding ground state free energy density is determined to be 
$$
    \mathcal{F}_p^* = -\frac{a^2}{4b} - \frac{\alpha^2}{4\beta}.
$$

In order to study the interactions of composite SVPs, it will prove convenient to normalize the energy such that the ground state configuration has zero energy.
To do this, we consider the non-linear sigma model limit by requiring the magnetization to have fixed length $|\vec{m}|=m_0$.
That is, we define the normalized free energy of the theory to be
$$
    E = \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{1}{2}|\vec{D}\psi|^2 + \frac{1}{2}|\vec{\nabla}\times\vec{A}|^2 + \frac{b}{4} \left( u^2 - |\psi|^2 \right)^2 + \frac{1}{2}|\nabla\vec{m}|^2 - \vec{m}\cdot(\vec{\nabla}\times\vec{A})\right\}.
$$

In this model, we are interested in stationary configurations that take the form of local minima of the free energy.
These satisfy the (bulk) ferromagnetic Ginzburg--Landau equations that are obtained by variation of $E$ with respect to the fields $(\psi,\vec{A},\vec{m})$, which yields the Euler-Lagrange field equations
$$
\begin{align*}
    \frac{\delta E}{\delta \psi^*} = \, & -b\psi \left( u^2 - |\psi|^2 \right) - \frac{1}{2}\vec{D}\cdot\vec{D}\psi = 0, \\
    \frac{\delta E}{\delta \vec{A}} = \, & q^2|\psi|^2 \vec{A} + \frac{iq}{2}\left( \psi \vec{\nabla}\psi^* - \psi^*\vec{\nabla}\psi \right) + \vec{\nabla}\times\vec{\nabla}\times\vec{A} - \vec{\nabla}\times\vec{m} = \vec{0},\\
    \frac{\delta E}{\delta \vec{m}} = \, & - \Delta \vec{m}  - \vec{\nabla}\times\vec{A} = \vec{0}.
\end{align*}
$$
From the gauge field equation , we get the supercurrent
$$
    \vec{J} = \frac{iq}{2}\left( \psi \vec{\nabla}\psi^* - \psi^*\vec{\nabla}\psi \right) + q^2|\psi|^2 \vec{A},
$$
and the magnetization current
$$
    \vec{J}_{\textup{mag}} = \vec{\nabla}\times\vec{m}.
$$

### Including the effects of spin-flip scattering

We now look to include the effects of spin-flip scattering of conduction electrons with the magnetic moments and conduction-electron polarization.
This gives rise to terms in the free-energy of the form
$$
    \mathcal{F}_{\textup{spin-flip}}[\psi,\vec{m}] = \left(\eta_1|\vec{m}|^2 + \eta_2|\vec{\nabla}\vec{m}|^2\right)|\psi|^2,
$$
where $\eta_2=\xi_s^2\eta_1$ and $\xi_s$ is the ordinary superconducting coherence length.
The first term describes conduction-electron polarization, and arises from the polarization of conduction electrons by the ferromagnetic order.
The second term is spin-flip scattering which causes gradient-driven pair breaking.
From the previous section, we observed that the coherence length in the ferromagnetic superconducting phase is the coherence length of an ordinary superconductor, that is
$$
    \xi_s = \frac{1}{\sqrt{-2a}}, \quad a < 0.
$$
With the spin-flip scattering included, the potential energy becomes
$$
    \mathcal{F}_p=\frac{a}{2}|\psi|^2 + \frac{b}{4}|\psi|^4 + \frac{\alpha}{2}|\vec{m}|^2 + \frac{\beta}{4}|\vec{m}|^4 + \eta_1|\vec{m}|^2|\psi|^2.
$$
The associated uniform ground state configurations are found to by solving the system of equations
$$
\begin{align*}
    \left.\frac{\delta \mathcal{F}_p}{\delta |\psi|}\right|_{(v,n_0)} = \, & av + bv^3 + 2\eta_1 n_0^2 v = 0, \\ \left.\frac{\delta \mathcal{F}_p}{\delta |\vec{m}|}\right|_{(v,n_0)} = \, & \alpha n_0 + \beta n_0^3 + 2\eta_1 n_0 v^2 = 0,
\end{align*}
$$
which gives us the ground state
$$
    v^2 = \frac{2\alpha\eta_1-a\beta}{b\beta-4\eta_1^2}, \quad n_0^2 = \frac{2a\eta_1-\alpha b}{b\beta-4\eta_1^2}.
$$
The corresponding ground state free energy density is determined to be 
$$
    \mathcal{F}_p^* = \frac{\alpha^2 b + a^2\beta - 4a\alpha\eta_1}{16\eta_1^2-4b\beta}.
$$
So, the free energy now reads
$$
\begin{align*}
    E[\psi,\vec{A},\vec{m}] = \, & F_{\textup{sc}}[\psi,\vec{A}] + F_{\textup{mag}}[\vec{m}] + F_{\textup{zeeman}}[\vec{A},\vec{m}] + F_{\textup{spin-flip}}[\psi,\vec{m}] - F_p^* \\
    = \, &  \int_{\mathbb{R}^2} \textup{d}^2x \left\{ \frac{a}{2}|\psi|^2 + \frac{b}{4}|\psi|^4 + \frac{1}{2}|\vec{D}\psi|^2 + \frac{1}{2}|\vec{\nabla}\times\vec{A}|^2 + \frac{\alpha}{2}|\vec{m}|^2 + \frac{\beta}{4}|\vec{m}|^4 + \frac{1}{2}|\nabla\vec{m}|^2 - \vec{m}\cdot(\vec{\nabla}\times\vec{A}) + \eta_1|\vec{m}|^2|\psi|^2 + \eta_2|\vec{\nabla}\vec{m}|^2|\psi|^2 - \frac{\alpha^2 b + a^2\beta - 4a\alpha\eta_1}{16\eta_1^2-4b\beta} \right\}.
\end{align*}
$$
The associated Euler--Lagrange field equations including the spin-flip scattering terms are found to be given by
$$
\begin{align*}
    \frac{\delta E}{\delta \psi^*} = \, & \left( \frac{a}{2} + \frac{b}{2}|\psi|^2 +\eta_1|\vec{m}|^2 + \eta_2|\vec{\nabla}\vec{m}|^2\right)\psi - \frac{1}{2}\vec{D}\cdot\vec{D}\psi = 0, \\
    \frac{\delta E}{\delta \vec{A}} = \, & q^2|\psi|^2 \vec{A} + \frac{iq}{2}\left( \psi \vec{\nabla}\psi^* - \psi^*\vec{\nabla}\psi \right) + \vec{\nabla}\times\vec{\nabla}\times\vec{A} - \vec{\nabla}\times\vec{m} = \vec{0},\\
    \frac{\delta E}{\delta \vec{m}} = \, & \left(\alpha + \beta |\vec{m}|^2 + 2\eta_1|\psi|^2 \right)\vec{m} - \left(1 + 2\eta_2|\psi|^2 \right) \Delta \vec{m} - \vec{\nabla}\times\vec{A} = \vec{0}.
\end{align*}
$$

### Numerical implementation

In terms of the order parameters, the arrested Newton flow algorithm is reformulating the minimization as a second order dynamical problem.
That is, `soliton_solver` is solving the second order coupled system
$$
\begin{align*}
    \frac{\textup{d}^2\psi}{\textup{d}t^2} = \, & -\frac{\delta E}{\delta \psi^*}, \\
    \frac{\textup{d}^2 \vec{A}}{\textup{d}t^2} = \, & -\frac{\delta E}{\delta \vec{A}}, \\
    \frac{\textup{d}^2 \vec{m}}{\textup{d}t^2} = \, & -\frac{\delta E}{\delta \vec{m}},
\end{align*}
$$
where $t$ is a fictitious time coordinate.

### References

P. Leask, C. Ross and E. Babaev, *Interactions of composite magnetic-skyrmion–superconducting vortex pairs in ferromagnetic superconductors*, [Phys. Rev. B. **114**, 014509 (2026)](https://doi.org/10.1103/crwr-b57q)

## Spin-triplet superconducting magnet

### Physics modeled

This model describes a ferromagnet coupled to two superconducting components. It is a natural extension of the ferromagnetic superconductor in which both spin-triplet pairing channels interact with the magnetization texture.

### Fields and mappings

- $\vec{m}(\vec{x})\in\mathbb{R}^3$: the magnetization vector, stored as three real components.
- $\psi_1(\vec{x})\in\mathbb{C}$ and $\psi_2(\vec{x})\in\mathbb{C}$: two complex order parameters, stored as four real channels.
- $\vec{A}(\vec{x})\in\mathbb{R}^3$: the gauge field, stored as three real channels.

### Dimensionless formulation used in the implementation

A dimensionless functional used in the implementation is

$$
\mathcal{E}[\vec{m},\varphi_1,\varphi_2,\vec{A}'] = \int d^2x'\left[
\frac12\sum_{\alpha=1}^2|D_i'\varphi_\alpha|^2 + \frac12|\nabla'\times\vec{A}'|^2
+ \frac{\bar{\alpha}}{2}|\vec{m}|^2 + \frac{\bar{\beta}}{4}|\vec{m}|^4 + \frac{\bar{\gamma}^2}{2}|\nabla'\vec{m}|^2
+ \frac{\bar{a}}{2}\sum_{\alpha}|\varphi_\alpha|^2 + \frac{\bar{b}_1}{4}\sum_{\alpha}|\varphi_\alpha|^4 + \bar{b}_2|\varphi_1|^2|\varphi_2|^2 + \bar{c}(\varphi_1\varphi_2^* + \varphi_1^*\varphi_2)
\right].
$$

The couplings $b_1,b_2,c$ encode the inter-component structure of the two superconducting order parameters.

### Non-dimensionalisation

After scaling with a characteristic length and order-parameter amplitude, the result is a coupled non-linear system for $\varphi_1$, $\varphi_2$ and $\vec{m}$ with dimensionless parameters $\bar{a}$, $\bar{b}_1$, $\bar{b}_2$, $\bar{c}$ and $\bar{\gamma}$.

### Euler-Lagrange equations

The field equations are the natural two-component analogue of the ferromagnetic-superconductor equations,

$$
\left(\nabla' - i\vec{A}'\right)^2\varphi_\alpha + \bar{a}\varphi_\alpha + \bar{b}_1|\varphi_\alpha|^2\varphi_\alpha + \bar{b}_2|\varphi_{\bar{\alpha}}|^2\varphi_\alpha + \bar{c}\,\varphi_{\bar{\alpha}} = 0,
$$

and

$$
\vec{m}\times\left(\bar{\gamma}\,\nabla'^2\vec{m} + \bar{\alpha}\vec{m} + \bar{\beta}|\vec{m}|^2\vec{m} - \nabla'\times\vec{A}'\right)=0.
$$

## Initial configurations and multi-soliton construction

The solver does not need separate initial configurations for every theory, but the construction of initial conditions is crucial for producing vortices and skyrmions reliably.
For the theories that support topological defects, the initial state is usually built from a phase-winding ansatz or from superposing single-soliton profiles.
A robust strategy is to place a single vortex/skyrmion at a prescribed point and then combine several such profiles with controlled separations.

### Initial configurations

For the superconducting order parameter we use an extended version of the Nielsen-Olesen ansatz
$$
    \psi(r,\theta) = \sigma(r)e^{-iN\theta}, \quad \vec{A}(r,\theta) =  \left(-\frac{a(r)}{r}\sin\theta,\frac{a(r)}{r}\cos\theta,g(r)\right),
$$
where the profile functions satisfy the boundary conditions $\sigma(0) = 0, \sigma(\infty) = u$, $a(0)=0, a(\infty)=N/q$ and $g'(0)=g(\infty)=0$, and $N\in\mathbb{Z}$ is the winding number, or vortex number.
In the case of superconducting vortices not in ferromagnetic superconductors, the regular Nielsen-Olesen ansatz is obtained by setting $g(r)=0$.
Now, by Stoke's theorem, it follows that the total magnetic flux through the $xy$-plane is thus
$$
    \Phi = \int_{\mathbb{R}^2} B_3 \textup{d}^2x = 2\pi \int_0^\infty \frac{\textup{d}a}{\textup{d}r} \textup{d}r = N\frac{2\pi}{q}\equiv N\Phi_0,
$$
and
$$
    \int_{\mathbb{R}^2} B_1 \textup{d}^2x = \int_{\mathbb{R}^2} B_2 \textup{d}^2x = 0.
$$
Hence, the flux of the vortices are quantized, with $\Phi_0=2\pi/q$ being the quantum. 
For the magnetization we have the choice of the three ansatze
$$
    \vec{m}_{\textup{N\'eel}}(r,\theta)=
    \begin{pmatrix}
        \sin f(r)\cos\theta \\
        \sin f(r)\sin\theta \\
        \cos f(r)
    \end{pmatrix}, \quad
    \vec{m}_{\textup{Bloch}}(r,\theta)=
    \begin{pmatrix}
        -\sin f(r)\sin\theta \\
        \sin f(r)\cos\theta \\
        \cos f(r)
    \end{pmatrix}, \quad
    \vec{m}_{\textup{Heusler}}(r,\theta)=
    \begin{pmatrix}
        -\sin f(r)\sin\theta \\
        -\sin f(r)\cos\theta \\
        \cos f(r)
    \end{pmatrix}.
$$
where $f(r)$ is some monotonically increasing profile function that satisfies the boundary conditions $f(0)=-1$ and $f(\infty)=1$.
It can be seen that at $r=0$ we have spin down states whereas we have spin up states as $r\rightarrow\infty$.
There is also a topological invariant associated with the magnetization ansatz, which is
$$
    n =  \frac{1}{4\pi} \int_{\mathbb{R}^2}\vec{m} \cdot \left( \partial_x \vec{m} \times \partial_y \vec{m} \right)  \textup{d}^2x = \pm 1 \in \mathbb{Z}.
$$

To construct an initial configuration for multi-soliton configurations, we can use two different methods.
The first is straightforward where we consider axially symmetric initial configurations and simply set $N>1$ and $Q>1$.
While these ans\"atze are initially axially symmetric, they do not necessarily relax to axially symmetric configurations.
The second method involves separated solitons, where the soliton cores do not overlap.
For the magnetization, this is carried out using the $\mathbb{C}P^1$ formalism.
That is, we introduce the complex variable
$$
    W(\vec{x}) = \frac{m_x(\vec{x})+im_y(\vec{x})}{1+m_z(\vec{x})} \in \mathbb{C}.
$$
Then we can construct multi-skyrmions, of topological degree $n=-k$, using the product ansatz
$$
    W(\vec{x}) = \sum_{i=1}^{k} W_i(\vec{x}-\vec{x}_i),
$$
where $\vec{x}_i$ is the location of the $i$th skyrmion for each $W_i$.
The resulting magnetization can be recovered via
$$
    \vec{m} = \frac{1}{1+|W|^2}
    \begin{pmatrix}
        W+W^* \\
        i(W^*-W) \\
        1-|W|^2
    \end{pmatrix}.
$$
In a similar manner, we introduce the Abrikosov ansatz to obtain a superconducting $k$-vortex,
$$
    \psi(\vec{x}) = \prod_{i=1}^{k} \psi_i(\vec{x}-\vec{x}_i), \quad \vec{A}(\vec{x}) = \sum_{i=1}^{k} \vec{A}_i(\vec{x}-\vec{x}_i).
$$
Together, these ansatze construct $k$-separated solitons.

This is the same idea used in the composite magnetic skyrmion-superconducting vortex problem. In that setting one builds an initial state by placing a magnetic skyrmion and a superconducting vortex at controlled positions, and then relaxing the coupled system. The resulting state can be interpreted as a bound or repulsive pair depending on the separation and the relative phase. The same construction generalizes to multi-soliton states with several vortices and skyrmions: one places the defects at different locations, assigns the appropriate winding numbers, and allows the solver to relax the coupled fields. The package’s initial-config kernels implement this logic in a GPU-friendly way.

## Summary

The theories in `soliton_solver` are all built on the same principle: a field theory whose energy functional supports stable topological defects. The models differ in the field content and the couplings, but the common numerical strategy is the same: define an energy, relax the field configuration, and track the resulting defect interactions. The superconducting models emphasize gauged vortices, the magnetic models emphasize skyrmionic textures, and the coupled superconducting-magnetic models describe composite states in which vortices and skyrmions interact strongly.

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


