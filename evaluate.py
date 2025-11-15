import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from config import device, T_MIN, T_MAX
from nn.character_net import CharacterNet
from nn.losses import (
    chi_eps_sig_from_net,
    chi_eps_exact,
    chi_sig_exact,
    check_chi1_accuracy,
)


def evaluate_and_plot():

    # Load trained network
    net = CharacterNet(hidden=64).to(device)
    net.load_state_dict(torch.load("ising_net.pt", map_location=device))
    net.eval()

    # Sample t-grid
    t_grid = torch.logspace(math.log10(T_MIN), math.log10(T_MAX), 200, device=device)
    s_grid = torch.log(t_grid).unsqueeze(-1)

    # NN characters
    with torch.no_grad():
        chi_eps_nn, chi_sig_nn, _, _ = chi_eps_sig_from_net(net, s_grid)

    # Exact characters
    chi_eps_ex = chi_eps_exact(t_grid)
    chi_sig_ex = chi_sig_exact(t_grid)

    # Convert for plotting
    t_np = t_grid.cpu().numpy()
    eps_nn_np = chi_eps_nn.cpu().numpy()
    sig_nn_np = chi_sig_nn.cpu().numpy()
    eps_ex_np = chi_eps_ex.cpu().numpy()
    sig_ex_np = chi_sig_ex.cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title(r"$\chi_\varepsilon(t)$")
    axes[0].loglog(t_np, eps_ex_np, label="exact", linewidth=2)
    axes[0].loglog(t_np, eps_nn_np, "--", label="NN", linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title(r"$\chi_\sigma(t)$")
    axes[1].loglog(t_np, sig_ex_np, label="exact", linewidth=2)
    axes[1].loglog(t_np, sig_nn_np, "--", label="NN", linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].grid(True, which="both", ls="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Checking χ₁ precision...")
    check_chi1_accuracy()

    print("Plotting NN vs exact χ_ε, χ_σ ...")
    evaluate_and_plot()