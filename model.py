import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

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
        return out, z, mu, log_var


# shape autoencoder pipeline
class ShapeAutoEncoder(nn.Module):
    def __init__(self, features, channels, layers, reg_channels):
        """
        features: number of latent features
        channels: positional encoding dimension
        layers: number of attention layer in decoder
        reg_channels: kl-regularization latent channels
        """
        super().__init__()
        self.features = features
        self.channels = channels
        self.layers = layers

        # position encoding
        self.embedder = Embedding(channels)

        # encoder cross attention
        self.encoder = Attention(channels)

        # kl regularizer
        self.regularizer = VAE(channels, reg_channels)

        # decoder
        self.decoder = nn.Sequential(*[Attention(channels) for _ in range(layers)])

        # interpolate
        self.radial_basis_func = Attention(channels)
        self.transform = nn.Sequential(nn.Linear(channels, 1), nn.Tanh())
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.transform[0].weight)
        init.zeros_(self.transform[0].bias)

    def encode(self, x):
        """input point cloud samples [B, M, 3] encode to [B, N, C]"""
        # partial cross attention
        batch = x.shape[0]
        idxs = farthest_point_sampler(x, self.features)

        embed = self.embedder(x)
        pnts = embed[np.array([[i] * self.features for i in range(batch)]), idxs]

        # encoder
        features = self.encoder(pnts, embed)
        assert list(features.shape) == [batch, self.features, self.channels]

        # kl-regularization block
        features, latents, mu, logvar = self.regularizer(features)
        return features, latents, mu, logvar

    def decode(self, feat, q):
        """input encoded features [B, N, C] decode to  occupancies [B, M]"""
        # decoder
        features = self.decoder(feat)

        # cross attention
        q_embed = self.embedder(q)
        q_output = self.radial_basis_func(q_embed, features)
        occupancy = self.transform(q_output)
        occupancy = 0.5 * occupancy + 0.5
        return occupancy

    def forward(self, x, q):
        """input point cloud samples [B, M, 3]"""
        features, _, mu, logvar = self.encode(x)
        occupancy = self.decode(features, q)
        res_dict = {
            "regularize_mu": mu,
            "regularize_var": logvar,
            "occupancy": occupancy.view(x.shape[0], -1)
        }
        return res_dict
    

# shape diffusion processor
class ShapeDenoiser(nn.Module):
    def __init__(self, channels, layers):
        self.denoiser = nn.Sequential(*[Attention(channels) for _ in range(layers * 2)])
    
    def forward(self, x, cond=None):
        """[B, M, C] -> [B, M, C]"""
        y = x
        for i, module in enumerate(self.denoiser):
            if i % 2 == 0:
                y = module(y)
            else:
                y = module(y, cond)
        return y


if __name__ == '__main__':
    net = ShapeAutoEncoder(features=128, channels=512, layers=8, reg=True, reg_channels=32).cuda()
    for key, val in net.state_dict().items():
        print(f"{key}: {val.shape}")
