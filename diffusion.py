import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm


def extract(v, t, shape):
    """extract coefficients at timesteps"""
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(shape) - 1))


class DiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """gaussian diffusion trainer"""
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for reverse process
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0):
        """[B, M, C] -> [1] from feature to noise"""
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction="none")
        return loss


class DiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta, timesteps, skip_type):
        """gaussian diffusion sampler"""
        super().__init__()
        assert 0 < timesteps <= T, "timesteps out of range"

        self.model = model
        self.T = T

        if skip_type == 'uniform':
            skip = T // timesteps
            self.seq = list(range(0, T, skip))
        elif skip_type == 'quad':
            seq = np.linspace(0, np.sqrt(T * 0.8), timesteps) ** 2
            self.seq = [int(s) for s in list(seq)]
        
        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)[self.seq]
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:-1]
        self.register_buffer('betas', 1.0 - alphas_bar / alphas_bar_prev)

        c1 = eta * torch.sqrt(self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))
        c2 = torch.sqrt(1.0 - alphas_bar_prev - c1 ** 2)
        self.register_buffer('coeff1', torch.sqrt(1.0 / (1.0 - self.betas)))
        self.register_buffer('coeff2', self.coeff1 * torch.sqrt(1.0 - alphas_bar) - c2)
        self.register_buffer('posterior_var', c1 ** 2)

    def predict_xt_prev_mean_from_eps(self, x_t, i, eps):
        """predict x_{t-1} in reverse process"""
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, i, x_t.shape) * x_t -
            extract(self.coeff2, i, x_t.shape) * eps
        )
    
    def p_mean_variance(self, x_t, i, t):
        """predict mean & variance for sampling"""
        var = extract(self.posterior_var, i, x_t.shape)
        eps = self.model(x_t, t)
        mean = self.predict_xt_prev_mean_from_eps(x_t, i, eps=eps)
        return mean, var

    def forward(self, x_T):
        """[B, M, C] -> [B, M, C] from noise to feature"""
        B = x_T.shape[0]
        x_t = x_T
        with tqdm(reversed(list(enumerate(self.seq)))) as tqdmSampler:
            for i, time_step in tqdmSampler:
                t = (torch.ones(B) * time_step).to(x_t.device).long()
                i = (torch.ones(B) * i).to(x_t.device).long()
                mean, var = self.p_mean_variance(x_t, i, t)
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
                tqdmSampler.set_postfix(ordered_dict={
                    "time": time_step
                })
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
