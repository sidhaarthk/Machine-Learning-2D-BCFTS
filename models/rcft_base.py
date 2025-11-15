import torch


class RCFTModel:
    """
    Abstract base class for rational CFT models.
    Minimal interface required by the training and evaluation scripts.
    """

    # -----------------------------
    # MUST be implemented in child:
    # -----------------------------
    def num_characters(self):
        """
        How many characters χ_i(t) this model has.
        """
        raise NotImplementedError

    def character_names(self):
        """
        List of string names, e.g. ["chi1", "chi_eps", "chi_sig"]
        """
        raise NotImplementedError

    def exact_characters(self, t: torch.Tensor):
        """
        Return list [chi_1(t), chi_2(t), ..., chi_n(t)]
        using theta/eta formulas.
        """
        raise NotImplementedError

    def IR_envelopes(self, t: torch.Tensor):
        """
        Return list [A1(t), A2(t), ...] smoothing factors
        for approximate χ_i(t) ≈ A_i(t) * exp(u_i(s)).
        """
        raise NotImplementedError