import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from dgl.geometry import farthest_point_sampler


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
        """[B, 3] -> [B, C]"""
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
        """cross attention, dimention unchange"""
        assert len(x.shape) == 2

        # y is None, degenerate to self attention
        if y is None:
            y = x
        assert len(y.shape) == 2 and x.shape[1] == y.shape[1]
        
        B, C = x.shape
        D, _ = y.shape
        q = self.proj_q(x)
        k = self.proj_k(y)
        v = self.proj_v(y)

        w = torch.mm(q, k.T) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, D]
        w = F.softmax(w, dim=-1)

        h = torch.mm(w, v)
        assert list(h.shape) == [B, C]
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
        """[B, C] -> [B, C_0] -> [B, C]"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


# shape autoencoder pipeline
class SAE(nn.Module):
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
        self.radial_basis_func = nn.Sequential(
            Attention(channels),
            nn.Linear(channels, 1),
            nn.Tanh()
        )

    def forward(self, x, q):
        """input point cloud samples [B, 3]"""
        embed = self.embedder(x)

        # partial cross attention
        idxs = farthest_point_sampler(x)
        pnts = embed[idxs]
        features = self.encoder(pnts, embed)

        # kl-regularization block
        features, mu, logvar = self.regularizer(features)

        # decoder
        features = self.decoder(features)

        q_embed = self.embedder(q)
        occupancy = self.radial_basis_func(q_embed, features)

        ret_dict = {
            "regularize_mu": mu,
            "regularize_var": logvar,
            "occupancy": occupancy
        }
        
        return ret_dict
