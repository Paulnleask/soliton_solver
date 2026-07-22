## Chern-Simons-Landau-Ginzburg Theory of Vortex Anyons

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