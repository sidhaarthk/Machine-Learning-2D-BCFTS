import math
import torch

from .rcft_base import RCFTModel
from .theta_eta import theta2, theta3, theta4, eta


pi = math.pi
sqrt2 = math.sqrt(2.0)


class IsingModel(RCFTModel):
    """
    The Ising minimal model M(4,3), c = 1/2.
    
    Characters (in standard order):
      χ1, χ_ε, χ_σ

    The interface matches RCFTModel for use with any NN + Cardy pipeline.
    """

    def __init__(self):
        super().__init__()
        self._names = ["chi1", "chi_eps", "chi_sig"]

    # ---------------------------------------------------------
    # Model metadata
    # ---------------------------------------------------------
    def num_characters(self):
        return 3

    def character_names(self):
        return self._names

    # ---------------------------------------------------------
    #  nomes for τ = i t
    # ---------------------------------------------------------
    def nomes(self, t):
        """
        Return (q_theta, q_eta) for:
          q_theta = exp(-π t)  (θ2, θ3, θ4)
          q_eta   = exp(-2π t) (η)
        """
        q_theta = torch.exp(-pi * t)
        q_eta = torch.exp(-2.0 * pi * t)
        return q_theta, q_eta

    # ---------------------------------------------------------
    #  Exact θ/η characters
    # ---------------------------------------------------------
    def exact_characters(self, t: torch.Tensor):
        """
        Return list of exact Ising characters:
          [chi1(t), chi_eps(t), chi_sig(t)]
        """

        q_theta, q_eta = self.nomes(t)

        th2 = theta2(q_theta)
        th3 = theta3(q_theta)
        th4 = theta4(q_theta)
        et = eta(q_eta)

        sqrt = torch.sqrt

        chi1 = (sqrt(th3) + sqrt(th4)) / (2 * sqrt(et))

        chi_eps = (sqrt(th3 / et) - sqrt(th4 / et)) / 2

        chi_sig = (1 / sqrt2) * sqrt(th2 / et)

        return [chi1, chi_eps, chi_sig]

    # ---------------------------------------------------------
    #  IR envelopes A_i(t)
    # ---------------------------------------------------------
    def IR_envelopes(self, t: torch.Tensor):
        """
        Provide IR asymptotic envelopes:
            χ_i(t) ≈ A_i(t) * exp(u_i(log t)).
        
        For Ising:
            χ_ε ~ exp(-23π t / 24)
            χ_σ ~ exp(-π t / 12)
        
        χ1 is not learned by NN but included for Cardy constraints.
        """

        A1 = torch.exp(-pi * t / 12.0)                # mild, for Cardy
        A_eps = torch.exp(-23.0 * pi * t / 24.0)      # heavy ε sector
        A_sig = torch.exp(-pi * t / 12.0)

        return [A1, A_eps, A_sig]