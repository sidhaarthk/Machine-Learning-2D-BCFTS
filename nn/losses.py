import math
import torch
import numpy as np
import mpmath as mp

from config import device, pi, sqrt2, T_MIN, T_MAX, S_MIN, S_MAX


# --- IR envelopes (same as your original script) -----------------

def A_eps(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-23.0 * pi * t / 24.0)


def A_sig(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-pi * t / 12.0)


# --- theta/eta backend (same as your original) -------------------

N_THETA = 100
N_ETA = 400


def theta3(q, N=N_THETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(1, N + 1, device=q.device, dtype=q.dtype).reshape(-1, 1)
    terms = torch.exp((n * n) * logq)
    out = 1.0 + 2.0 * terms.sum(dim=0)
    return out.reshape(q_shape)


def theta4(q, N=N_THETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(1, N + 1, device=q.device, dtype=q.dtype).reshape(-1, 1)
    signs = torch.where(n % 2 == 0, 1.0, -1.0)
    terms = signs * torch.exp((n * n) * logq)
    out = 1.0 + 2.0 * terms.sum(dim=0)
    return out.reshape(q_shape)


def theta2(q, N=N_THETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(0, N, device=q.device, dtype=q.dtype).reshape(-1, 1)
    terms = torch.exp(((n + 0.5) ** 2) * logq)
    out = 2.0 * terms.sum(dim=0)
    return out.reshape(q_shape)


def eta(q, N=N_ETA):
    q_shape = q.shape
    logq = torch.log(q).reshape(1, -1)
    n = torch.arange(1, N + 1, device=q.device, dtype=q.dtype).reshape(-1, 1)
    qn = torch.exp(n * logq)
    log_terms = torch.log1p(-qn)
    log_prod = log_terms.sum(dim=0)
    log_eta = logq.squeeze(0) / 24.0 + log_prod
    return torch.exp(log_eta).reshape(q_shape)


# --- nomes and exact characters ----------------------------------

def ising_nomes(t: torch.Tensor):
    q_theta = torch.exp(-pi * t)
    q_eta = torch.exp(-2.0 * pi * t)
    return q_theta, q_eta


def chi_1(t: torch.Tensor) -> torch.Tensor:
    q_theta, q_eta = ising_nomes(t)
    th3 = torch.clamp(theta3(q_theta), min=1e-40)
    th4 = torch.clamp(theta4(q_theta), min=1e-40)
    et = torch.clamp(eta(q_eta), min=1e-40)
    return (torch.sqrt(th3) + torch.sqrt(th4)) / (2.0 * torch.sqrt(et))


def chi_eps_exact(t: torch.Tensor) -> torch.Tensor:
    q_theta, q_eta = ising_nomes(t)
    th3 = torch.clamp(theta3(q_theta), min=1e-40)
    th4 = torch.clamp(theta4(q_theta), min=1e-40)
    et = torch.clamp(eta(q_eta), min=1e-40)
    return (torch.sqrt(th3 / et) - torch.sqrt(th4 / et)) / 2.0


def chi_sig_exact(t: torch.Tensor) -> torch.Tensor:
    q_theta, q_eta = ising_nomes(t)
    th2 = torch.clamp(theta2(q_theta), min=1e-40)
    et = torch.clamp(eta(q_eta), min=1e-40)
    return (1.0 / math.sqrt(2.0)) * torch.sqrt(th2 / et)


# --- NN characters from net (only eps, sig) ----------------------

def chi_eps_sig_from_net(net, s: torch.Tensor):
    """
    Given s = log t, return:
      chi_eps(t), chi_sig(t), u_eps(s), u_sig(s)
    using envelopes A_eps, A_sig and NN outputs.
    """
    t = torch.exp(s).squeeze(-1)
    u = net(s)  # (batch, 2)
    u_eps, u_sig = u[:, 0], u[:, 1]

    chi_eps = A_eps(t) * torch.exp(u_eps)
    chi_sig = A_sig(t) * torch.exp(u_sig)

    return chi_eps, chi_sig, u_eps, u_sig


# --- Cardy residuals and loss -----------------------------------

def sample_s(batch_size, s_min=S_MIN, s_max=S_MAX):
    return torch.empty(batch_size, 1, device=device).uniform_(s_min, s_max)


def cardy_residuals(net, s: torch.Tensor):
    t = torch.exp(s).squeeze(-1)
    t_inv = torch.exp(-s).squeeze(-1)

    chi_eps_t, chi_sig_t, _, _ = chi_eps_sig_from_net(net, s)
    chi1_t = chi_1(t)

    chi_eps_inv, chi_sig_inv, _, _ = chi_eps_sig_from_net(net, -s)
    chi1_inv = chi_1(t_inv)

    R1 = 0.5 * chi1_inv + 0.5 * chi_eps_inv + (1 / sqrt2) * chi_sig_inv - chi1_t
    R2 = 0.5 * chi1_inv + 0.5 * chi_eps_inv - (1 / sqrt2) * chi_sig_inv - chi_eps_t
    R3 = (1 / sqrt2) * chi1_inv - (1 / sqrt2) * chi_eps_inv - chi_sig_t
    R6 = chi1_inv + chi_eps_inv - chi1_t - chi_eps_t

    return R1, R2, R3, R6


def cardy_loss(net, batch_size=256):
    s = sample_s(batch_size)
    R1, R2, R3, R6 = cardy_residuals(net, s)
    return (R1**2 + R2**2 + R3**2 + R6**2).mean()


# --- Regularisation ---------------------------------------------

def anchor_loss(net, t0=3.0, weight=10.0):
    t0_torch = torch.tensor([t0], dtype=torch.float64, device=device)
    s0 = torch.log(t0_torch).unsqueeze(-1)

    chi_eps_nn, chi_sig_nn, _, _ = chi_eps_sig_from_net(net, s0)

    chi_eps_ex = chi_eps_exact(t0_torch)
    chi_sig_ex = chi_sig_exact(t0_torch)

    mse = ((chi_eps_nn - chi_eps_ex)**2 +
           (chi_sig_nn - chi_sig_ex)**2).mean()
    return weight * mse


def asymptotic_regularisation(net, batch_size=256, band=0.4):
    s = sample_s(batch_size)
    mask = s[:, 0] > (S_MAX - band)
    if not mask.any():
        return torch.tensor(0.0, device=device)
    s_sel = s[mask]
    _, _, u_eps, u_sig = chi_eps_sig_from_net(net, s_sel)
    return (u_eps**2 + u_sig**2).mean()


def smoothness_regularisation(net, batch_size=256, delta=0.05):
    s = sample_s(batch_size)
    s_plus = torch.clamp(s + delta, S_MIN, S_MAX)
    _, _, u_eps, u_sig = chi_eps_sig_from_net(net, s)
    _, _, u_eps_p, u_sig_p = chi_eps_sig_from_net(net, s_plus)
    return ((u_eps_p - u_eps)**2 + (u_sig_p - u_sig)**2).mean()


# --- χ₁ accuracy check (same as your original) -------------------

def check_chi1_accuracy():
    mp.mp.dps = 50

    def chi1_mpmath(t):
        q_theta = mp.e**(-mp.pi * t)
        q_eta = mp.e**(-2 * mp.pi * t)
        th3 = mp.jtheta(3, 0, q_theta)
        th4 = mp.jtheta(4, 0, q_theta)
        prod = mp.mpf('1')
        qn = q_eta
        while abs(qn) > mp.mpf('1e-40'):
            prod *= (1 - qn)
            qn *= q_eta
        et = q_eta**(mp.mpf('1') / 24) * prod
        return (mp.sqrt(th3) + mp.sqrt(th4)) / (2 * mp.sqrt(et))

    ts = np.logspace(math.log10(T_MIN), math.log10(T_MAX), 40)
    max_rel = 0.0
    for t in ts:
        t_torch = torch.tensor([t], dtype=torch.float64)
        approx = chi_1(t_torch).item()
        exact = float(chi1_mpmath(t))
        rel = abs(approx - exact) / abs(exact)
        max_rel = max(max_rel, rel)
    print(f"Max relative error χ₁ on [{T_MIN}, {T_MAX}]: {max_rel:.3e}")