import torch
from torch import nn

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
x = torch.randn((2, 3))
print(x)
x = x.exp()
print(x)
