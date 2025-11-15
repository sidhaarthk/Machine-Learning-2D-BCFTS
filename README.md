# Learning Ising CFT Characters from Modular Bootstrap Constraints

This project demonstrates how to reconstruct the **Ising model conformal characters**  
\(\chi_1(t), \chi_\varepsilon(t), \chi_\sigma(t)\)  
using only **modular S-transformation (Cardy) constraints**, combined with IR asymptotics and a single normalization anchor.

The neural network learns the energy and spin characters with high precision, matching the exact theta/eta expressions across the full interval  
\( t \in [10^{-1}, 5] \).

---

## Features

- Accurate PyTorch implementations of **Jacobi θ₂, θ₃, θ₄** and **Dedekind η** in float64.
- Correct nome conventions:
  - \( q_\theta = e^{-\pi t} \) for θ-functions,
  - \( q_\eta = e^{-2\pi t} \) for η.
- Exact computation of the identity character:
  \[
  \chi_1(t) = \frac{\sqrt{\theta_3} + \sqrt{\theta_4}}{2\sqrt{\eta}}.
  \]
- Neural network learns only the smooth correction functions  
  \(u_\varepsilon(s), u_\sigma(s)\), where \(s = \log t\).
- Enforces all three **Ising modular S-transform constraints**.
- Uses smoothness regularization and IR asymptotic regularization.
- Uses a single anchor point to fix the overall normalization.
- Validates χ₁ against a **50-digit mpmath reference**.
- Generates high-quality plots comparing learned vs. exact characters.

---

## Background

For characters \(\chi_i(\tau)\) of any 2D CFT, modular invariance implies:
\[
\chi_i(-1/\tau) = \sum_j S_{ij}\,\chi_j(\tau).
\]

Setting \(\tau = i t\) makes this a relation between values at \(t\) and \(1/t\).

The **Ising model** has three characters and the modular S-matrix:
\[
S = 
\begin{pmatrix}
 \tfrac12 & \tfrac12 & \tfrac1{\sqrt2} \\
 \tfrac12 & \tfrac12 & -\tfrac1{\sqrt2} \\
 \tfrac1{\sqrt2} & -\tfrac1{\sqrt2} & 0
\end{pmatrix}.
\]

Thus:
\[
\begin{aligned}
\chi_1(t) &= \tfrac12 \chi_1(1/t) + \tfrac12 \chi_\varepsilon(1/t) + \tfrac1{\sqrt2}\chi_\sigma(1/t), \\
\chi_\varepsilon(t) &= \tfrac12 \chi_1(1/t) + \tfrac12 \chi_\varepsilon(1/t) - \tfrac1{\sqrt2}\chi_\sigma(1/t), \\
\chi_\sigma(t) &= \tfrac1{\sqrt2}\chi_1(1/t) - \tfrac1{\sqrt2}\chi_\varepsilon(1/t).
\end{aligned}
\]

These equations define the residuals minimized during training.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <this-repo>
cd <this-repo>
pip install torch numpy mpmath matplotlib