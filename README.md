# Learning Ising CFT Characters from Modular Bootstrap Constraints

This project reconstructs the **Ising conformal field theory (CFT) characters**

- $\chi_1(t)$ — identity  
- $\chi_\varepsilon(t)$ — energy  
- $\chi_\sigma(t)$ — spin  

using only the **modular S-transformation (Cardy) equations**, IR asymptotics, and one normalization anchor.

A small neural network learns $\chi_\varepsilon$ and $\chi_\sigma$ with high accuracy, matching the exact $\theta/\eta$ expressions across:

$$
t \in [10^{-1}\, 5].
$$

---

## Features

- High-precision PyTorch implementations of **Jacobi $\theta_2,\theta_3,\theta_4$** and **Dedekind $\eta$**, in float64.
- Correct canonical nomes:  
  - $q_\theta = e^{-\pi t}$ for $\theta$–functions  
  - $q_\eta = e^{-2\pi t}$ for $\eta$
- Exact identity character:
  $$
  \chi_1(t) = \frac{\sqrt{\theta_3} + \sqrt{\theta_4}}{2\sqrt{\eta}}
  $$
- Neural network learns only the smooth deformations:
  $$
  \chi_i(t)=A_i(t)\,e^{u_i(\log t)}.
  $$
- Loss = Cardy residuals + IR regularisation + smoothness + single anchor.
- χ₁ validated against **50-digit mpmath**.
- Automatically generates comparison plots (NN vs exact).

---

## Background

In any 2D CFT with characters $\chi_i(\tau)$, modular invariance implies:

$$
\chi_i(-1/\tau) = \sum_j S_{ij}\,\chi_j(\tau).
$$

For the Ising model, the modular $S$-matrix is:

$$
S =
\begin{pmatrix}
 \tfrac12 & \tfrac12 & \tfrac1{\sqrt2} \\
 \tfrac12 & \tfrac12 & -\tfrac1{\sqrt2} \\
 \tfrac1{\sqrt2} & -\tfrac1{\sqrt2} & 0
\end{pmatrix}.
$$

Setting $\tau = i t$ gives functional relations between values at $t$ and $1/t$.

Thus:

$$
\chi_1(t)=\tfrac12\chi_1(1/t)+\tfrac12\chi_\varepsilon(1/t)+\tfrac1{\sqrt2}\chi_\sigma(1/t),
$$

$$
\chi_\varepsilon(t)=\tfrac12\chi_1(1/t)+\tfrac12\chi_\varepsilon(1/t)-\tfrac1{\sqrt2}\chi_\sigma(1/t),
$$

$$
\chi_\sigma(t)=\tfrac1{\sqrt2}\chi_1(1/t)-\tfrac1{\sqrt2}\chi_\varepsilon(1/t).
$$

These three Cardy equations are exactly the constraints used for training.

---

## Installation

```bash
git clone <repository-url>
cd <repository>
pip install torch numpy matplotlib mpmath