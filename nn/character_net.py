import torch.nn as nn


class CharacterNet(nn.Module):
    """
    Small MLP that outputs (u_eps(s), u_sig(s)).
    We exponentiate on top of IR envelopes to get chi_eps(t), chi_sig(t).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s):
        # s: (batch, 1)
        # returns: (batch, 2) = (u_eps, u_sig)
        return self.net(s)