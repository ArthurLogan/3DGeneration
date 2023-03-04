import torch
from torch import nn
import numpy as np


# kl-regularization block loss
class RegularizeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """minimize kl-divergence compact distribution & N(0,I)"""
        if mu is None or logvar is None:
            return 0
        assert len(mu.shape) == 3 and len(logvar.shape) == 3
        var = logvar.exp()
        loss = 0.5 * ((mu ** 2) + var - logvar).mean()
        return loss


# just for debugging
class Assert:
    @classmethod
    def check(cls, xs):
        for x in xs:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x)
            cls._check_inf(x)
            cls._check_nan(x)

    @classmethod
    def _check_inf(cls, x):
        assert torch.isinf(x).int().sum() == 0, "inf in tensor."
    
    @classmethod
    def _check_nan(cls, x):
        assert torch.isnan(x).int().sum() == 0, "nan in tensor."

