import torch
from torch import nn

class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True)
        # Add latent_dim attribute for compatibility with LatentToModulation model
        self.latent_dim = size

    def forward(self, x):
        return x + self.bias

class OneHotProjector(nn.Module):
    def __init__(self, nb_series):
        super().__init__()
        self.projection =  nn.Linear(in_features=nb_series, 
                                        out_features=1, 
                                        bias=True)

    def forward(self, OneHot):
        return self.projection(OneHot)
   