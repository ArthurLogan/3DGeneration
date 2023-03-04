import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

from dgl.geometry import farthest_point_sampler


# dictionary
class Dict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


# position embedding
class Embedding(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.proj = nn.Linear(3, out_ch, bias=False)
        self.initialize()
    
    def initialize(self):
        """initialize weight"""
        init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x):
        """[B, M, 3] -> [B, M, C]"""
        assert len(x.shape) == 3
        y = self.proj(x)
        return y


# attention block
class Attention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj_q = nn.Linear(in_ch, in_ch)
        self.proj_k = nn.Linear(in_ch, in_ch)
        self.proj_v = nn.Linear(in_ch, in_ch)
        self.proj = nn.Linear(in_ch, in_ch)
        self.norm = nn.LayerNorm(in_ch)
        self.initialize()
    
    def initialize(self):
        """initialize weight & bias"""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)
    
    def forward(self, x, y=None):
        """cross attention query [B, M, C] -> [B, M, C]"""
        assert len(x.shape) == 3

        # if y is None, degenerate to self attention
        # else cross attention
        if y is None:
            y = x
        assert len(y.shape) == 3 and x.shape[2] == y.shape[2]
        
        B, M, C = x.shape
        _, N, _ = y.shape
        q = self.proj_q(x)
        k = self.proj_k(y)
        v = self.proj_v(y)

        k = k.permute(0, 2, 1).view(B, C, N)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, M, N]
        w = F.softmax(w, dim=-1)

        h = torch.bmm(w, v)
        assert list(h.shape) == [B, M, C]
        h = self.proj(h)

        x = x + h
        x = self.norm(x)
        return x


# kl regularization block
class VAE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ch = out_ch
        self.proj_u = nn.Linear(in_ch, out_ch)
        self.proj_v = nn.Linear(in_ch, out_ch)
        self.proj = nn.Linear(out_ch, in_ch)
        self.initialize()
    
    def initialize(self):
        """initialize weight & bias"""
        for module in [self.proj_u, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)

    def encode(self, x):
        """vae encoder"""
        mu = self.proj_u(x)
        log_var = self.proj_v(x)
        return mu, log_var

    def decode(self, x):
        """vae decoder"""
        out = self.proj(x)
        return out

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        """[B, M, C] -> [B, M, C_0] -> [B, M, C]"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


# shape autoencoder pipeline
class ShapeAutoEncoder(nn.Module):
    def __init__(self, features, channels, layers, reg):
        """
        features: number of latent features
        channels: positional encoding dimension
        layers: number of attention layer in decoder
        reg: if use kl-regularization
        """
        super().__init__()
        self.features = features
        self.channels = channels
        self.layers = layers
        self.reg = reg

        # position encoding
        self.embedder = Embedding(channels)

        # encoder cross attention
        self.encoder = Attention(channels)

        # kl regularizer
        self.regularizer = VAE(channels, 32)

        # decoder
        self.decoder = nn.Sequential(*[Attention(channels) for _ in range(layers)])

        # interpolate
        self.radial_basis_func = Attention(channels)
        self.transform = nn.Sequential(nn.Linear(channels, 1), nn.Tanh())
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.transform[0].weight)
        init.zeros_(self.transform[0].bias)

    def forward(self, x, q, device):
        """input point cloud samples [B, M, 3]"""
        # partial cross attention
        batch = x.shape[0]
        idxs = farthest_point_sampler(x, self.features).to(device)

        embed = self.embedder(x.to(device))
        pnts = embed[np.array([[i] * self.features for i in range(batch)]), idxs]

        features = self.encoder(pnts, embed)
        assert list(features.shape) == [batch, self.features, self.channels]

        # kl-regularization block
        mu, logvar = None, None
        if self.reg:
            features, mu, logvar = self.regularizer(features)

        # decoder
        features = self.decoder(features)

        q_embed = self.embedder(q.to(device))
        q_output = self.radial_basis_func(q_embed, features)
        occupancy = self.transform(q_output)
        occupancy = 0.5 * occupancy + 0.5

        res_dict = {
            "regularize_mu": mu,
            "regularize_var": logvar,
            "occupancy": occupancy.view(batch, -1)
        }
        
        return res_dict


if __name__ == '__main__':
    device = torch.device('cuda:3')
    net = ShapeAutoEncoder(features=128, channels=512, layers=8, reg=True).to(device)

    B, M, N = 80, 1024, 2048
    epoch = 10
    for i in range(epoch):
        x = torch.randn((B, M, 3)).clamp(0, 1)
        q = torch.randn((B, N, 3)).clamp(0, 1)
        res = net(x, q, device)
