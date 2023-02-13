import torch
from torch import nn
import numpy as np

# Check nn.MultiheadAttention
# a = torch.randn(32, 1, 512)
# b = torch.randn(128, 1, 512)

# layer = nn.MultiheadAttention(512, 8, batch_first=True)

# output, weights = layer(a, b, b)

# print(output.shape, weights.shape)


# Check Farthest Point Sampler
# from dgl.geometry import farthest_point_sampler
# from time import time

# x = torch.randn((1, 80000, 3))

# start_time = time()
# idx = farthest_point_sampler(x, 512)
# end_time = time()
# print(idx.shape, end_time - start_time)


# Check Torch Exp
# x = torch.randn((2, 3))
# print(x)
# x = x.exp()
# print(x)


# Check Scheduler
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Linear(3, 3)
    
#     def forward(self, x):
#         return self.net(x)


# net = Net()

# optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0, last_epoch=-1)

# for i in range(100):
#     x = torch.randn((5, 3))
#     y = net(x)
#     optimizer.step()

#     lr = scheduler.get_lr()[0]
#     print(lr - 1e-3 * 0.5 * (1 + np.cos(i / 100 * np.pi)))
#     scheduler.step()


# Check nn.Linear
# B, M, C = 16, 100, 10
# a = torch.randn((B, M, 3))
# layer = nn.Linear(3, C, bias=False)
# b = layer(a)
# print(b.shape)

# Check Print Format
a = 0.1
b = 2
print(f'{b: d}, {a: .2f}')

