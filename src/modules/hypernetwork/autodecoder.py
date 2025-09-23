from typing import Any, Dict, Optional

import numpy as np

import torch
from torch import nn

class LatentToModulation(nn.Module):
    
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self,
        latent_dim: int,
        num_modulations: int,
        dim_hidden: int,
        num_layers: int = 1,
        activation: nn.Module = nn.SiLU,
        mip_encode: bool = False,
        apply_znorm: bool = True,
        target_znorm: Optional[float] = None
    ) -> None:

        super().__init__()

        n = 2 if mip_encode else 1
        latent_dim *= n

        self.latent_dim      = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden      = dim_hidden
        self.num_layers      = num_layers
        self.activation      = activation
        
        self.encode_mip      = mip_encode
        self.apply_znorm     = apply_znorm
        self.target_znorm    = target_znorm if isinstance(target_znorm, float) else 1.0

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)

        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def l2_normalizer(self, z: torch.Tensor) -> torch.Tensor:
        """ Normalize z [..., latent_dim] to the unit ball."""
        norm    = torch.norm(z, p=2, dim=-1, keepdim=True)
        z_out = z / (norm + 1e-8)
        return z_out
    
    def mip_encoder(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """ Encode the latent vector with sines and cosines."""
        # https://github.com/JJGO/hyperlight/blob/main/hyperlight/hypernet/encoding.py
        
        if torch.norm(z, p=2, dim=-1, keepdim=True).sum()==0:
            return torch.cat([z, z], dim=-1)
        
        z = torch.special.expit(z)
        scaled_z = torch.pi * z / 2
        
        return torch.cat([torch.cos(scaled_z), torch.sin(scaled_z)], dim=-1)

    def forward(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:
        
        # latent: [..., latent_dim]
        if self.encode_mip:
            latent = self.mip_encoder(latent)
        
        if self.apply_znorm:
            latent = self.l2_normalizer(latent) * self.target_znorm

        out = self.net(latent) # [..., num_modulations]

        return out  # [..., num_modulations]

class LayerWiseLatentToModulation(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        dim_hidden: int,
        num_layers: int
    ) -> None:

        super().__init__()

        self.latent_dim = latent_dim
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.net_list = nn.ModuleList([nn.Linear(latent_dim, dim_hidden) for l in num_layers])

    def forward(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:

        # latent must be (batch_size, latent_dim, num_layers) shape

        out = []

        for l in range(latent.shape[-1]):
            out.append(self.net_list[l](latent[..., l]))

        return torch.stack(out, dim=1) # [batch_size, num_layers, hidden_dim]