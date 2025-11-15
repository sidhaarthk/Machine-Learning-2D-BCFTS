import torch

# Number of terms for theta and eta expansions
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