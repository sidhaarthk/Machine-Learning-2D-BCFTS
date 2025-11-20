import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#  Global setup
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = math.pi
sqrt5 = math.sqrt(5.0)

# t-domain
T_MIN, T_MAX = 2e-1, 7
S_MIN, S_MAX = math.log(T_MIN), math.log(T_MAX)


# ============================================================
#  CharacterNet: Lee–Yang χ_1, χ_φ
# ============================================================

class CharacterNet(nn.Module):
    r"""
    Lee–Yang minimal model M(5,2), c = -22/5.
    Two characters:

      χ_1(t)  = exp(a_1_fixed * t + u_1(s))
      χ_φ(t)  = exp(a_φ       * t + u_φ(s))

    with s = log t, t = e^s.
    """

    def __init__(self, hidden=64):
        super().__init__()

        # neural u(s) (2 components: 1, φ)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

        # FIXED exponent for χ_1 (identity)
        # a(h) = 2π (c/24 - h). For Lee–Yang: c = -22/5, h_1 = 0
        # c/24 = -11/60 → a_1 = 2π * (-11/60) = -11π/30
        self.a1_fixed = -11.0 * pi / 30.0

        # Trainable exponent for χ_φ (do NOT fix it)
        # (You could initialise near π/30, but keep it trainable.)
        self.a_phi = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))

    def forward(self, s):
        """
        Input: s of shape (batch, 1)
        Returns:
            chi1, chi_phi, u1, u_phi  (all shape (batch,))
        """
        t = torch.exp(s).squeeze(-1)  # (batch,)
        u = self.net(s)               # (batch, 2)
        u1, u_phi = u[:, 0], u[:, 1]

        chi1   = torch.exp(self.a1_fixed * t + u1)
        chi_phi = torch.exp(self.a_phi      * t + u_phi)

        return chi1, chi_phi, u1, u_phi


model = CharacterNet(hidden=64).to(device)


# ============================================================
#  Rogers–Ramanujan functions and exact Lee–Yang characters
# ============================================================

# Truncation for q-series
N_RR = 50  # you can increase if you want higher precision

def rogers_ramanujan_G(q, N=N_RR):
    """
    G(q) = sum_{n=0}^∞ q^{n^2} / (q;q)_n
    where (q;q)_n = Π_{k=1}^n (1 - q^k)
    q: tensor (batch,)
    """
    G = torch.zeros_like(q)
    poch = torch.ones_like(q)  # (q;q)_0 = 1

    for n in range(0, N+1):
        if n > 0:
            poch = poch * (1.0 - q**n)
        G = G + q**(n*n) / poch

    return G


def rogers_ramanujan_H(q, N=N_RR):
    """
    H(q) = sum_{n=0}^∞ q^{n(n+1)} / (q;q)_n
    q: tensor (batch,)
    """
    H = torch.zeros_like(q)
    poch = torch.ones_like(q)  # (q;q)_0 = 1

    for n in range(0, N+1):
        if n > 0:
            poch = poch * (1.0 - q**n)
        H = H + q**(n*(n+1)) / poch

    return H


def chi1_exact(t):
    """
    χ_1(q) = q^{11/60} H(q),  q = e^{-2π t}
    """
    q = torch.exp(-2.0 * pi * t)
    H = rogers_ramanujan_H(q)
    return q**(11.0/60.0) * H


def chi_phi_exact(t):
    """
    χ_φ(q) = q^{-1/60} G(q),  q = e^{-2π t}
    """
    q = torch.exp(-2.0 * pi * t)
    G = rogers_ramanujan_G(q)
    return q**(-1.0/60.0) * G


# ============================================================
#  Cardy residuals: Lee–Yang S-matrix (2×2)
# ============================================================

# Modular S-matrix for Lee–Yang (ordering: 1, φ):
# S = (2/√5) * [ [ -sin(2π/5),  sin(π/5) ],
#                [  sin(π/5),   sin(2π/5) ] ]
S11 = -2.0 * math.sin(2.0*pi/5.0) / sqrt5
S12 =  2.0 * math.sin(    pi/5.0) / sqrt5
S21 = S12
S22 =  2.0 * math.sin(2.0*pi/5.0) / sqrt5

def cardy_residuals(s):
    """
    Enforce χ_i(t) ≈ Σ_j S_{ij} χ_j(1/t),
    with t = e^s,  1/t = e^{-s}.
    """
    chi1_t, chi_phi_t, _, _ = model(s)
    chi1_inv, chi_phi_inv, _, _ = model(-s)

    R1 = S11 * chi1_inv + S12 * chi_phi_inv - chi1_t
    R2 = S21 * chi1_inv + S22 * chi_phi_inv - chi_phi_t

    return R1, R2


def cardy_loss(batch_size=256):
    s = torch.empty(batch_size, 1, device=device).uniform_(S_MIN, S_MAX)
    R1, R2 = cardy_residuals(s)
    return (R1**2 + R2**2).mean()


# ============================================================
#  Regularisation
# ============================================================

def smoothness_regularisation(batch_size=256, delta=0.05):
    s = torch.empty(batch_size, 1, device=device).uniform_(S_MIN, S_MAX)
    s_plus = torch.clamp(s + delta, S_MIN, S_MAX)
    _, _, u1, u_phi = model(s)
    _, _, u1p, u_phip = model(s_plus)
    return ((u1p - u1)**2 + (u_phip - u_phi)**2).mean()


def asymptotic_regularisation(batch_size=256, band=0.4):
    """
    Penalise large u's near t = T_MAX (s close to S_MAX).
    """
    s = torch.empty(batch_size, 1, device=device).uniform_(S_MIN, S_MAX)
    mask = s[:, 0] > (S_MAX - band)
    if not mask.any():
        return torch.tensor(0.0, device=device)
    _, _, u1, u_phi = model(s[mask])
    return (u1**2 + u_phi**2).mean()


# ============================================================
#  ANCHOR term (Lee–Yang, Rogers–Ramanujan)
# ============================================================

def anchor_loss(t0=2.0, weight=10.0):
    """
    Pins χ_1, χ_φ at t = t0 to their exact Lee–Yang values
    (Rogers–Ramanujan expressions).
    """
    t0 = torch.tensor([t0], dtype=torch.float64, device=device)
    s0 = torch.log(t0).unsqueeze(-1)

    chi1_nn, chi_phi_nn, _, _ = model(s0)

    chi1_ex  = chi1_exact(t0)
    chi_phi_ex = chi_phi_exact(t0)

    mse = ((chi1_nn - chi1_ex)**2 +
           (chi_phi_nn - chi_phi_ex)**2).mean()

    return weight * mse


# ============================================================
#  Training loop
# ============================================================

def train(num_steps=20000, batch_size=256, lr_initial=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr_initial)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=num_steps, eta_min=1e-9
    )

    for step in range(1, num_steps + 1):
        Lc = cardy_loss(batch_size)
        Ls = smoothness_regularisation(batch_size)
        La = asymptotic_regularisation(batch_size)
        L_anchor = anchor_loss(t0=2.0, weight=20.0)

        # you can re-enable Ls if you want
        loss = Lc + 0*Ls + 1e-2 * La + L_anchor

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 500 == 0:
            print(
                f"step {step:5d}  "
                f"Lc={Lc.item():.3e}  La={La.item():.3e}  Ls={Ls.item():.3e}  "
                f"Anch={L_anchor.item():.3e}  "
                f"a_phi={model.a_phi.item():.4f}"
            )


# ============================================================
#  Plot NN vs exact (Lee–Yang)
# ============================================================

def plot_vs_exact():
    model.eval()
    with torch.no_grad():
        t = torch.logspace(
            math.log10(T_MIN),
            math.log10(T_MAX),
            300,
            device=device
        )
        s = torch.log(t).unsqueeze(-1)

        chi1_nn, chi_phi_nn, _, _ = model(s)

        chi1_ex  = chi1_exact(t).squeeze()
        chi_phi_ex = chi_phi_exact(t).squeeze()

    t_np = t.cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].loglog(t_np, chi1_ex.cpu().numpy(), label="Exact")
    ax[0].loglog(t_np, chi1_nn.cpu().numpy(), "--", label="NN")
    ax[0].set_title(r"$\chi_1$ (Lee\text{–}Yang)")
    ax[0].legend()
    ax[0].grid(True, ls="--", alpha=0.4)

    ax[1].loglog(t_np, chi_phi_ex.cpu().numpy(), label=r"Exact")
    ax[1].loglog(t_np, chi_phi_nn.cpu().numpy(), "--", label="NN")
    ax[1].set_title(r"$\chi_\phi$ (Lee\text{–}Yang)")
    ax[1].legend()
    ax[1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("Training Lee–Yang model: ONE fixed asymptotic (identity) + ANCHOR…")
    train(num_steps=1000000)

    print("Plotting NN vs exact Lee–Yang characters…")
    plot_vs_exact()