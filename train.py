import torch
import torch.optim as optim

from config import device
from nn.character_net import CharacterNet
from nn.losses import (
    cardy_loss,
    asymptotic_regularisation,
    smoothness_regularisation,
    anchor_loss,
    check_chi1_accuracy,
)
from nn.losses import chi_eps_exact, chi_sig_exact  # optional reuse
from nn.losses import chi_eps_sig_from_net          # optional reuse


def train(num_steps=10000, batch_size=256, lr=1e-3):
    net = CharacterNet(hidden=64).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for step in range(1, num_steps + 1):
        Lc = cardy_loss(net, batch_size)
        La = asymptotic_regularisation(net, batch_size)
        Ls = smoothness_regularisation(net, batch_size)
        L_anchor = anchor_loss(net, t0=3.0, weight=20.0)

        loss = Lc + 1e-1 * La + 1e-3 * Ls + L_anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(
                f"step {step:5d} | "
                f"Lc={Lc.item():.3e} | "
                f"La={La.item():.3e} | "
                f"Ls={Ls.item():.3e} | "
                f"L_anchor={L_anchor.item():.3e} | "
                f"Total={loss.item():.3e}"
            )

    torch.save(net.state_dict(), "ising_net.pt")
    print("Saved trained model to ising_net.pt")
    return net


if __name__ == "__main__":
    print("Training NN on Cardy conditions with θ/η χ₁…")
    net = train()
    print("Checking χ₁(t) precision…")
    check_chi1_accuracy()