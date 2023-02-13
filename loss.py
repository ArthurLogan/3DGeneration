import torch
from torch import nn


class RegularizeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """minimize kl-divergence compact distribution & N(0,I)"""
        assert len(mu.shape) == 3 and len(logvar.shape) == 3
        var = logvar.exp()
        loss = 0.5 * ((mu ** 2) + var - logvar).mean()
        return loss
