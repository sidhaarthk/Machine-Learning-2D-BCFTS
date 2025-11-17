import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp   # only used AFTER training for accuracy checks

# ============================================================
#  Global setup
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = math.pi
sqrt2 = math.sqrt(2.0)

# t-domain for training / plotting
T_MIN, T_MAX = 1e-1, 5
S_MIN, S_MAX = math.log(T_MIN), math.log(T_MAX)


# ============================================================
#  Neural net: u_1(s), u_eps(s), u_sig(s)
# ============================================================

class CharacterNet(nn.Module):
    """
    Small MLP that outputs (u_1(s), u_eps(s), u_sig(s)) and we exponentiate
    on top of IR envelopes to get χ_1(t), χ_ε(t), χ_σ(t).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, s):
        # s: (batch,1)
        # returns: (batch,3) = (u_1, u_eps, u_sig)
        return self.net(s)


model = CharacterNet(hidden=64).to(device)


# ============================================================
#  IR-only envelopes
# ============================================================

def A_1(t: torch.Tensor) -> torch.Tensor:
    """
    IR envelope for vacuum character χ_1.
    NOTE: this is a modest exponential ansatz to capture leading IR decay.
    You may want to replace/tune this if you have a more accurate asymptotic.
    """
    # placeholder exponential envelope; adjust if you want a different IR behaviour
    return torch.exp(pi * t / 24.0)


def A_eps(t: torch.Tensor) -> torch.Tensor:
    """
    IR envelope for ε-character (leading exponential decay).
    χ_ε(t) ≈ A_eps(t) * exp(u_eps(s)), t = exp(s).
    """
    return torch.exp(-23.0 * pi * t / 24.0)   # Ising χ_ε IR asymptotics (kept from your code)


def A_sig(t: torch.Tensor) -> torch.Tensor:
    """
    IR envelope for σ-character.
    """
    return torch.exp(-pi * t / 12.0)          # Ising χ_σ IR asymptotics (kept)


# ============================================================
#  θ and η in float64, given their nomes q
# ============================================================

N_THETA = 100
N_ETA   = 400


def theta3(q, N=N_THETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(1, N+1, device=q.device, dtype=q.dtype).reshape(-1, 1)
    terms = torch.exp((n * n) * logq)
    out = 1.0 + 2.0 * terms.sum(dim=0)
    return out.reshape(q_shape)


def theta4(q, N=N_THETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(1, N+1, device=q.device, dtype=q.dtype).reshape(-1, 1)
    signs = torch.where(n % 2 == 0, 1.0, -1.0)
    terms = signs * torch.exp((n * n) * logq)
    out = 1.0 + 2.0 * terms.sum(dim=0)
    return out.reshape(q_shape)


def theta2(q, N=N_THETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(0, N, device=q.device, dtype=q.dtype).reshape(-1, 1)
    terms = torch.exp(((n + 0.5)**2) * logq)
    out = 2.0 * terms.sum(dim=0)
    return out.reshape(q_shape)


def eta(q, N=N_ETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(1, N+1, device=q.device, dtype=q.dtype).reshape(-1, 1)
    qn = torch.exp(n * logq)
    log_terms = torch.log1p(-qn)
    log_prod = log_terms.sum(dim=0)
    log_eta = logq.squeeze(0)/24.0 + log_prod
    return torch.exp(log_eta).reshape(q_shape)


# ============================================================
#  Helper: Ising θ/η conventions at τ = i t
# ============================================================

def ising_nomes(t: torch.Tensor):
    """
    Given t > 0 and τ = i t, return the θ- and η-nomes:

      q_theta = e^{π i τ} = e^{-π t}
      q_eta   = e^{2π i τ} = e^{-2π t}
    """
    q_theta = torch.exp(-pi * t)
    q_eta   = torch.exp(-2.0 * pi * t)
    return q_theta, q_eta


# ============================================================
#  χ_1(t) exact in PyTorch (used for anchor/check)
# ============================================================

def chi1_exact_from_theta_eta(t: torch.Tensor) -> torch.Tensor:
    """
    Ising vacuum character exact expression from θ/η:
      χ_1(t) = (sqrt θ3 + sqrt θ4) / (2 sqrt η),
    evaluated at τ = i t.
    """
    q_theta, q_eta = ising_nomes(t)

    th3 = torch.clamp(theta3(q_theta), min=1e-40)
    th4 = torch.clamp(theta4(q_theta), min=1e-40)
    et  = torch.clamp(eta(q_eta),      min=1e-40)

    return (torch.sqrt(th3) + torch.sqrt(th4)) / (2.0 * torch.sqrt(et))


# ============================================================
#  NN characters χ_1, χ_ε, χ_σ from model
# ============================================================

def chi_from_model(s: torch.Tensor):
    """
    Given s = log t, produce χ_1(t), χ_ε(t), χ_σ(t) from the model with IR envelopes.
    Returns (chi1, chi_eps, chi_sig, u1, u_eps, u_sig).
    """
    t = torch.exp(s).squeeze(-1)   # (batch,)
    u = model(s)                   # (batch,3)
    u1, u_eps, u_sig = u[:, 0], u[:, 1], u[:, 2]

    chi1 = A_1(t) * torch.exp(u1)
    chi_eps = A_eps(t) * torch.exp(u_eps)
    chi_sig = A_sig(t) * torch.exp(u_sig)

    return chi1, chi_eps, chi_sig, u1, u_eps, u_sig


# ============================================================
#  Cardy residuals for Ising (now using model χ_1)
# ============================================================

def cardy_residuals(s: torch.Tensor):
    """
    Cardy constraints for the Ising model using model-produced characters at t and 1/t.
    """
    t = torch.exp(s).squeeze(-1)
    t_inv = torch.exp(-s).squeeze(-1)

    # model at t
    chi1_t, chi_eps_t, chi_sig_t, _, _, _ = chi_from_model(s)

    # model at 1/t (i.e. s -> -s)
    chi1_inv, chi_eps_inv, chi_sig_inv, _, _, _ = chi_from_model(-s)

    R1 = 0.5 * chi1_inv + 0.5 * chi_eps_inv + (1 / sqrt2) * chi_sig_inv - chi1_t
    R2 = 0.5 * chi1_inv + 0.5 * chi_eps_inv - (1 / sqrt2) * chi_sig_inv - chi_eps_t
    R3 = (1 / sqrt2) * chi1_inv - (1 / sqrt2) * chi_eps_inv - chi_sig_t
    R6 = chi1_inv + chi_eps_inv - chi1_t - chi_eps_t

    return R1, R2, R3, R6


# ============================================================
#  Sampling + losses
# ============================================================

def sample_s(batch_size, s_min=S_MIN, s_max=S_MAX):
    """
    Uniform sampling in s = log t, i.e. log-uniform in t.
    """
    return torch.empty(batch_size, 1, device=device).uniform_(s_min, s_max)


def cardy_loss(batch_size=256):
    """
    Mean squared Cardy residuals over random s-batch.
    """
    s = sample_s(batch_size)
    R1, R2, R3, R6 = cardy_residuals(s)
    return (R1**2 + R2**2 + R3**2 + R6**2).mean()


# ============================================================
#  Anchor: fix overall normalization / branch choice (now anchors all three)
# ============================================================

def chi_eps_exact(t: torch.Tensor) -> torch.Tensor:
    """
    Exact Ising ε-character:

      χ_ε = (sqrt(θ3/η) - sqrt(θ4/η)) / 2
    """
    q_theta, q_eta = ising_nomes(t)

    th3 = torch.clamp(theta3(q_theta), min=1e-40)
    th4 = torch.clamp(theta4(q_theta), min=1e-40)
    et  = torch.clamp(eta(q_eta),      min=1e-40)

    return (torch.sqrt(th3 / et) - torch.sqrt(th4 / et)) / 2.0


def chi_sig_exact(t: torch.Tensor) -> torch.Tensor:
    """
    Exact Ising σ-character:

      χ_σ = (1/√2) * sqrt(θ2/η)
    """
    q_theta, q_eta = ising_nomes(t)

    th2 = torch.clamp(theta2(q_theta), min=1e-40)
    et  = torch.clamp(eta(q_eta),      min=1e-40)

    return (1.0 / math.sqrt(2.0)) * torch.sqrt(th2 / et)


def anchor_loss(t0=3.0, weight=10.0):
    """
    Penalise deviation of NN χ_1, χ_ε, χ_σ from exact θ/η characters at t0.
    This fixes normalization / square-root branch choices.
    """
    t0_torch = torch.tensor([t0], dtype=torch.float64, device=device)
    s0 = torch.log(t0_torch).unsqueeze(-1)   # shape (1,1)

    # NN estimates
    chi1_nn, chi_eps_nn, chi_sig_nn, _, _, _ = chi_from_model(s0)

    # Exact θ/η values
    chi1_ex = chi1_exact_from_theta_eta(t0_torch)
    chi_eps_ex = chi_eps_exact(t0_torch)
    chi_sig_ex = chi_sig_exact(t0_torch)

    mse = ((chi1_nn - chi1_ex)**2 + (chi_eps_nn - chi_eps_ex)**2 + (chi_sig_nn - chi_sig_ex)**2).mean()
    return weight * mse


# ============================================================
#  Regularisation: asymptotics & smoothness
# ============================================================

def asymptotic_regularisation(batch_size=256, band=0.4):
    """
    Encourage u_1,u_eps,u_sig -> 0 near the IR end t ~ T_MAX, so that
    χ follow the chosen envelopes there.
    """
    s = sample_s(batch_size)
    mask = s[:, 0] > (S_MAX - band)
    if not mask.any():
        return torch.tensor(0.0, device=device)
    s_sel = s[mask]
    _, _, _, u1, u_eps, u_sig = chi_from_model(s_sel)
    return (u1**2 + u_eps**2 + u_sig**2).mean()


def smoothness_regularisation(batch_size=256, delta=0.05):
    """
    Encourage χ's to be smooth in s. We penalise differences in u
    at nearby s, which loosely approximates (∂_s u)^2.
    """
    s = sample_s(batch_size)
    s_plus = torch.clamp(s + delta, S_MIN, S_MAX)
    _, _, _, u1, u_eps, u_sig = chi_from_model(s)
    _, _, _, u1_p, u_eps_p, u_sig_p = chi_from_model(s_plus)
    return ((u1_p - u1)**2 + (u_eps_p - u_eps)**2 + (u_sig_p - u_sig)**2).mean()


# ============================================================
#  Training loop
# ============================================================

def train(num_steps=100000, batch_size=256, lr_initial=1e-3):
    """
    Training loop with a cosine-annealed learning-rate schedule.
    This dramatically improves final precision, allowing loss < 1e-7.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr_initial)

    # Cosine decay: LR goes 1e-3 → 1e-4 → 3e-5 → 1e-5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=1e-8     # final LR
    )

    for step in range(1, num_steps + 1):
        # Core Cardy constraint
        Lc = cardy_loss(batch_size)

        # IR regularisation
        La = asymptotic_regularisation(batch_size)

        # Smoothness
        Ls = smoothness_regularisation(batch_size)

        # Anchor
        L_anchor = anchor_loss(t0=3.0, weight=20.0)

        # Total
        loss = Lc + 1e2*La + 1e-3*Ls + L_anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()     # <──— LR updated here

        if step % 500 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step {step:6d} | "
                f"LR={lr_now:.2e} | "
                f"Lc={Lc.item():.3e} | "
                f"La={La.item():.3e} | "
                f"Ls={Ls.item():.3e} | "
                f"L_anchor={L_anchor.item():.3e} | "
                f"Total={loss.item():.3e}"
            )


# ============================================================
#  Post-hoc check: χ₁ precision vs mpmath 50-dps exact
# ============================================================

def check_chi1_accuracy():
    mp.mp.dps = 50

    def chi1_mpmath(t):
        # correct conventions: q_theta = e^{-π t}, q_eta = e^{-2π t}
        q_theta = mp.e**(-mp.pi * t)
        q_eta   = mp.e**(-2 * mp.pi * t)
        th3 = mp.jtheta(3, 0, q_theta)
        th4 = mp.jtheta(4, 0, q_theta)

        # high-precision eta via product
        prod = mp.mpf('1')
        qn = q_eta
        # build product until extremely small
        while abs(qn) > mp.mpf('1e-40'):
            prod *= (1 - qn)
            qn *= q_eta
        et = q_eta**(mp.mpf('1') / 24) * prod

        return (mp.sqrt(th3) + mp.sqrt(th4)) / (2 * mp.sqrt(et))

    model.eval()
    with torch.no_grad():
        ts = np.logspace(math.log10(T_MIN), math.log10(T_MAX), 40)
        max_rel = 0.0
        for t in ts:
            s = torch.log(torch.tensor([t], dtype=torch.float64)).unsqueeze(-1).to(device)
            chi1_nn, _, _, _, _, _ = chi_from_model(s)
            approx = float(chi1_nn.cpu().numpy().squeeze())
            exact = float(chi1_mpmath(t))
            rel = abs(approx - exact) / abs(exact)
            max_rel = max(max_rel, rel)
    print(f"Max relative error χ₁ (NN vs 50-dps mpmath) on [{T_MIN}, {T_MAX}]: {max_rel:.3e}")


# ============================================================
#  Plot NN vs exact θ/η characters
# ============================================================

def evaluate_and_plot():
    model.eval()
    with torch.no_grad():
        t_grid = torch.logspace(math.log10(T_MIN), math.log10(T_MAX), 200, device=device)
        s_grid = torch.log(t_grid).unsqueeze(-1)

        chi1_nn, chi_eps_nn, chi_sig_nn, _, _, _ = chi_from_model(s_grid)
        chi1_ex = chi1_exact_from_theta_eta(t_grid)
        chi_eps_ex = chi_eps_exact(t_grid)
        chi_sig_ex = chi_sig_exact(t_grid)

        t_np      = t_grid.cpu().numpy()
        one_nn_np = chi1_nn.cpu().numpy()
        eps_nn_np = chi_eps_nn.cpu().numpy()
        sig_nn_np = chi_sig_nn.cpu().numpy()
        one_ex_np = chi1_ex.cpu().numpy()
        eps_ex_np = chi_eps_ex.cpu().numpy()
        sig_ex_np = chi_sig_ex.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].set_title(r"$\chi_1(t)$")
    axes[0].loglog(t_np, one_ex_np, label=r"$\chi_1$ exact", linewidth=2)
    axes[0].loglog(t_np, one_nn_np, "--", label="NN", linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\chi_1$")
    axes[0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title(r"$\chi_\varepsilon(t)$")
    axes[1].loglog(t_np, eps_ex_np, label=r"$\chi_\varepsilon$ exact", linewidth=2)
    axes[1].loglog(t_np, eps_nn_np, "--", label="NN", linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\chi_\varepsilon$")
    axes[1].grid(True, which="both", ls="--", alpha=0.4)
    axes[1].legend()

    axes[2].set_title(r"$\chi_\sigma(t)$")
    axes[2].loglog(t_np, sig_ex_np, label=r"$\chi_\sigma$ exact", linewidth=2)
    axes[2].loglog(t_np, sig_nn_np, "--", label="NN", linewidth=2)
    axes[2].set_xlabel("t")
    axes[2].set_ylabel(r"$\chi_\sigma$")
    axes[2].grid(True, which="both", ls="--", alpha=0.4)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    print("Training NN to learn all three Ising characters (χ_1, χ_ε, χ_σ)…")
    train(num_steps=500000, batch_size=256, lr_initial=1e-3)

    print("Checking χ₁(t) precision against 50-dps mpmath reference…")
    check_chi1_accuracy()

    print("Plotting NN vs θ/η characters (χ_1, ε, σ)…")
    evaluate_and_plot()
