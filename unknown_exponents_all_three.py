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
sqrt2 = math.sqrt(2.0)

# t-domain
T_MIN, T_MAX = 2e-1, 5
S_MIN, S_MAX = math.log(T_MIN), math.log(T_MAX)


# ============================================================
#  CharacterNet: u_i(s) + learned exponents for ε, σ
# ============================================================

class CharacterNet(nn.Module):
    """
    χ_1(t)      = exp( a_1_fixed * t + u_1(s) )
    χ_ε(t)      = exp( a_eps      * t + u_eps(s) )
    χ_σ(t)      = exp( a_sig      * t + u_sig(s) )
    """

    def __init__(self, hidden=64):
        super().__init__()

        # neural u(s)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

        # FIXED exponent for χ1
        self.a1_fixed = pi / 24.0

        # Trainable exponents
        self.a_eps = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
        self.a_sig = nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))

    def forward(self, s):
        t = torch.exp(s).squeeze(-1)
        u = self.net(s)
        u1, u_eps, u_sig = u[:,0], u[:,1], u[:,2]

        chi1    = torch.exp(self.a1_fixed * t + u1)
        chi_eps = torch.exp(self.a_eps      * t + u_eps)
        chi_sig = torch.exp(self.a_sig      * t + u_sig)

        return chi1, chi_eps, chi_sig, u1, u_eps, u_sig


model = CharacterNet(hidden=64).to(device)


# ============================================================
#  Exact Ising characters (θ/η)
# ============================================================

N_THETA = 80
N_ETA   = 200

def theta3(q):
    logq = torch.log(q).unsqueeze(0)      # shape (1, batch)
    n = torch.arange(1, N_THETA+1, dtype=q.dtype, device=q.device).unsqueeze(1)
    terms = torch.exp((n**2) * logq)      # shape (N_THETA, batch)
    return 1.0 + 2.0 * terms.sum(dim=0)   # → (batch,)

def theta4(q):
    logq = torch.log(q).unsqueeze(0)
    n = torch.arange(1, N_THETA+1, dtype=q.dtype, device=q.device).unsqueeze(1)
    signs = torch.where(n % 2 == 0, 1.0, -1.0)
    terms = signs * torch.exp((n**2) * logq)
    return 1.0 + 2.0 * terms.sum(dim=0)

def theta2(q):
    logq = torch.log(q).unsqueeze(0)
    n = torch.arange(0, N_THETA, dtype=q.dtype, device=q.device).unsqueeze(1)
    terms = torch.exp(((n + 0.5)**2) * logq)
    return 2.0 * terms.sum(dim=0)

def eta(q):
    logq = torch.log(q).unsqueeze(0)
    n = torch.arange(1, N_ETA+1, dtype=q.dtype, device=q.device).unsqueeze(1)
    terms = torch.log1p(-torch.exp(n*logq))   # shape (N_ETA, batch)
    log_eta = logq/24.0 + terms.sum(dim=0)
    return torch.exp(log_eta)                 # → (batch,)

def ising_nomes(t):
    return torch.exp(-pi*t), torch.exp(-2*pi*t)

def chi1_exact(t):
    q_theta, q_eta = ising_nomes(t)
    th3 = theta3(q_theta); th4 = theta4(q_theta); et = eta(q_eta)
    return (torch.sqrt(th3) + torch.sqrt(th4)) / (2*torch.sqrt(et))

def chi_eps_exact(t):
    q_theta, q_eta = ising_nomes(t)
    th3 = theta3(q_theta); th4 = theta4(q_theta); et = eta(q_eta)
    return (torch.sqrt(th3/et) - torch.sqrt(th4/et)) / 2.0

def chi_sig_exact(t):
    q_theta, q_eta = ising_nomes(t)
    th2 = theta2(q_theta); et = eta(q_eta)
    return (1/math.sqrt(2)) * torch.sqrt(th2/et)


# ============================================================
#  Cardy residuals
# ============================================================

def cardy_residuals(s):
    chi1_t, chi_eps_t, chi_sig_t, _, _, _ = model(s)
    chi1_inv, chi_eps_inv, chi_sig_inv, _, _, _ = model(-s)

    R1 = 0.5*chi1_inv + 0.5*chi_eps_inv + (1/sqrt2)*chi_sig_inv - chi1_t
    R2 = 0.5*chi1_inv + 0.5*chi_eps_inv - (1/sqrt2)*chi_sig_inv - chi_eps_t
    R3 = (1/sqrt2)*chi1_inv - (1/sqrt2)*chi_eps_inv - chi_sig_t
    R6 = chi1_inv + chi_eps_inv - chi1_t - chi_eps_t

    return R1, R2, R3, R6

def cardy_loss(batch_size=256):
    s = torch.empty(batch_size,1,device=device).uniform_(S_MIN,S_MAX)
    R1,R2,R3,R6 = cardy_residuals(s)
    return (R1**2 + R2**2 + R3**2 + R6**2).mean()


# ============================================================
#  Regularisation
# ============================================================

def smoothness_regularisation(batch_size=256, delta=0.05):
    s = torch.empty(batch_size,1,device=device).uniform_(S_MIN,S_MAX)
    s_plus = torch.clamp(s+delta, S_MIN, S_MAX)
    _,_,_,u1,ue,us = model(s)
    _,_,_,u1p,uep,usp = model(s_plus)
    return ((u1p-u1)**2 + (uep-ue)**2 + (usp-us)**2).mean()

def asymptotic_regularisation(batch_size=256, band=0.4):
    s = torch.empty(batch_size,1,device=device).uniform_(S_MIN,S_MAX)
    mask = s[:,0] > (S_MAX-band)
    if not mask.any(): 
        return torch.tensor(0.0,device=device)
    _,_,_,u1,ue,us = model(s[mask])
    return (u1**2 + ue**2 + us**2).mean()


# ============================================================
#  ANCHOR term (new)
# ============================================================

def anchor_loss(t0=2.0, weight=10.0):
    """
    Pins χ1, χ_eps, χ_sig at t=t0 to their exact Ising values.
    Ensures normalization and removes overall rescalings.
    """
    t0 = torch.tensor([t0], dtype=torch.float64, device=device)
    s0 = torch.log(t0).unsqueeze(-1)

    chi1_nn, chi_eps_nn, chi_sig_nn, _, _, _ = model(s0)

    chi1_ex = chi1_exact(t0)
    chi_eps_ex = chi_eps_exact(t0)
    chi_sig_ex = chi_sig_exact(t0)

    mse = ((chi1_nn - chi1_ex)**2 +
           (chi_eps_nn - chi_eps_ex)**2 +
           (chi_sig_nn - chi_sig_ex)**2).mean()

    return weight * mse


# ============================================================
#  Training loop
# ============================================================

def train(num_steps=20000, batch_size=256, lr_initial=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr_initial)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=1e-9)

    for step in range(1,num_steps+1):
        Lc = cardy_loss(batch_size)
        Ls = smoothness_regularisation(batch_size)
        La = asymptotic_regularisation(batch_size)
        L_anchor = anchor_loss(t0=2.0, weight=20.0)

        loss = Lc + 1e-3*Ls + 1e-2*La + L_anchor

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 500 == 0:
            print(f"step {step:5d}  "
                  f"Lc={Lc.item():.3e}  La={La.item():.3e}  Ls={Ls.item():.3e}  Anch={L_anchor.item():.3e}  "
                  f"a_eps={model.a_eps.item():.4f}, a_sig={model.a_sig.item():.4f}")


# ============================================================
#  Plot NN vs exact
# ============================================================

def plot_vs_exact():
    model.eval()
    with torch.no_grad():
        t = torch.logspace(math.log10(T_MIN), math.log10(T_MAX), 300, device=device)
        s = torch.log(t).unsqueeze(-1)

        chi1_nn, chi_eps_nn, chi_sig_nn, _, _, _ = model(s)

        chi1_ex = chi1_exact(t).squeeze()
        chi_eps_ex = chi_eps_exact(t).squeeze()
        chi_sig_ex = chi_sig_exact(t).squeeze()

    t_np = t.cpu().numpy()

    fig, ax = plt.subplots(1,3,figsize=(18,5))

    ax[0].loglog(t_np, chi1_ex.cpu().numpy(), label="Exact")
    ax[0].loglog(t_np, chi1_nn.cpu().numpy(), "--", label="NN")
    ax[0].set_title(r"$\chi_1$")
    ax[0].legend(); ax[0].grid(True,ls="--",alpha=0.4)

    ax[1].loglog(t_np, chi_eps_ex.cpu().numpy(), label="Exact")
    ax[1].loglog(t_np, chi_eps_nn.cpu().numpy(), "--", label="NN")
    ax[1].set_title(r"$\chi_\varepsilon$")
    ax[1].legend(); ax[1].grid(True,ls="--",alpha=0.4)

    ax[2].loglog(t_np, chi_sig_ex.cpu().numpy(), label="Exact")
    ax[2].loglog(t_np, chi_sig_nn.cpu().numpy(), "--", label="NN")
    ax[2].set_title(r"$\chi_\sigma$")
    ax[2].legend(); ax[2].grid(True,ls="--",alpha=0.4)

    plt.tight_layout()
    plt.show()



# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("Training with TWO learned asymptotics + ANCHOR…")
    train(num_steps=500000)

    print("Plotting NN vs exact Ising characters…")
    plot_vs_exact()
