import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# ============================================================
#  Global setup
# ============================================================

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")   # parallelism requires CPU

pi = math.pi
sqrt2 = math.sqrt(2.0)

# t-domain
T_MIN, T_MAX = 2e-1, 5
S_MIN, S_MAX = math.log(T_MIN), math.log(T_MAX)


# ============================================================
#  CharacterNet: u_i(s) + learned exponents for ε, σ
# ============================================================

class CharacterNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 3, dtype=torch.float64),
        )

        self.a1_fixed = pi / 24.0

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


# ============================================================
#  Exact Ising characters (θ/η)
# ============================================================

N_THETA = 80
N_ETA   = 200

def theta3(q):
    logq = torch.log(q).unsqueeze(0)
    n = torch.arange(1, N_THETA+1, dtype=q.dtype, device=q.device).unsqueeze(1)
    terms = torch.exp((n**2) * logq)
    return 1.0 + 2.0 * terms.sum(dim=0)

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
    terms = torch.log1p(-torch.exp(n * logq))
    log_eta = logq/24.0 + terms.sum(dim=0)
    return torch.exp(log_eta)

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

def cardy_residuals(model, s):
    chi1_t, chi_eps_t, chi_sig_t, _, _, _ = model(s)
    chi1_inv, chi_eps_inv, chi_sig_inv, _, _, _ = model(-s)

    R1 = 0.5*chi1_inv + 0.5*chi_eps_inv + (1/sqrt2)*chi_sig_inv - chi1_t
    R2 = 0.5*chi1_inv + 0.5*chi_eps_inv - (1/sqrt2)*chi_sig_inv - chi_eps_t
    R3 = (1/sqrt2)*chi1_inv - (1/sqrt2)*chi_eps_inv - chi_sig_t
    R6 = chi1_inv + chi_eps_inv - chi1_t - chi_eps_t

    return R1, R2, R3, R6

def cardy_loss(model, batch_size=256):
    s = torch.empty(batch_size,1,device=device).uniform_(S_MIN,S_MAX)
    R1,R2,R3,R6 = cardy_residuals(model, s)
    return (R1**2 + R2**2 + R3**2 + R6**2).mean()


# ============================================================
#  Regularisation
# ============================================================

def smoothness_regularisation(model, batch_size=256, delta=0.05):
    s = torch.empty(batch_size,1,device=device).uniform_(S_MIN,S_MAX)
    s_plus = torch.clamp(s+delta, S_MIN, S_MAX)
    _,_,_,u1,ue,us = model(s)
    _,_,_,u1p,uep,usp = model(s_plus)
    return ((u1p-u1)**2 + (uep-ue)**2 + (usp-us)**2).mean()

def asymptotic_regularisation(model, batch_size=256, band=0.4):
    s = torch.empty(batch_size,1,device=device).uniform_(S_MIN,S_MAX)
    mask = s[:,0] > (S_MAX-band)
    if not mask.any():
        return torch.tensor(0.0,device=device)
    _,_,_,u1,ue,us = model(s[mask])
    return (u1**2 + ue**2 + us**2).mean()


# ============================================================
#  Anchor loss
# ============================================================

def anchor_loss(model, t0, weight=20.0):
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
#  Deviation metric
# ============================================================

def avg_deviation(model, n_points=200):
    t = torch.logspace(math.log10(T_MIN), math.log10(T_MAX), n_points, device=device)
    s = torch.log(t).unsqueeze(-1)

    with torch.no_grad():
        chi1_nn, chi_eps_nn, chi_sig_nn, _, _, _ = model(s)

    chi1_ex  = chi1_exact(t)
    chi_eps_ex = chi_eps_exact(t)
    chi_sig_ex = chi_sig_exact(t)

    dev = (
        (chi1_nn - chi1_ex).abs() +
        (chi_eps_nn - chi_eps_ex).abs() +
        (chi_sig_nn - chi_sig_ex).abs()
    ).mean()

    return dev.item()


# ============================================================
#  One Experiment (50,000 epochs)
# ============================================================

def run_single_experiment(anchor_t0, epochs=50000, verbose=False):
    torch.set_default_dtype(torch.float64)

    model = CharacterNet(hidden=64).to(device).double()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-9)

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"t0={anchor_t0:.2f}")

    for step in iterator:
        Lc = cardy_loss(model)
        Ls = smoothness_regularisation(model)
        La = asymptotic_regularisation(model)
        L_anchor = anchor_loss(model, t0=anchor_t0, weight=20.0)

        loss = Lc + 1e-3*Ls + 1e-2*La + L_anchor

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    return avg_deviation(model)



# ============================================================
#  Parallel launcher with progress bar
# ============================================================

def run_experiments_parallel(num_points=200, epochs=50000, n_jobs=-1):
    anchor_points = np.linspace(0.2, 5.0, num_points)

    deviations = Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(t0, epochs=epochs, verbose=False)
        for t0 in tqdm(anchor_points, desc="Dispatching")
    )

    return anchor_points, np.array(deviations)


# ============================================================
#  Histogram
# ============================================================

def plot_scatter(anchor_points, deviations):
    plt.figure(figsize=(8,6))
    plt.scatter(anchor_points, deviations, s=20, alpha=0.7)
    plt.xlabel("Anchor point $t_0$")
    plt.ylabel("Avg deviation")
    plt.title("Deviation vs Anchor point $t_0$")
    plt.grid(True, ls='--', alpha=0.4)
    plt.show()



# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("Running 200 experiments over anchor t0 ∈ [0.2, 5]...")

    anchor_points, deviations = run_experiments_parallel(
        num_points=200,
        epochs=50000
    )
    print("Plotting scatter plot…")
    plot_scatter(anchor_points, deviations)