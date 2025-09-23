from typing import Dict, Any

import torch
import torch.nn as nn

class NeRFEncoding(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""

    def __init__(
        self,
        num_freq: int,
        min_freq_log2: int,
        max_freq_log2: int,
        log_sampling: bool = True,
        include_input: bool = True,
        input_dim: int = 3,
        base_freq: int = 2,
    ) -> None:
        
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """

        super().__init__()

        self.num_freq      = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling  = log_sampling
        self.include_input = include_input
        self.out_dim       = 0
        self.base_freq     = base_freq

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = self.base_freq ** torch.linspace(
                min_freq_log2, max_freq_log2, steps=num_freq
            ) # [num_freq,]
        else:
            self.bands = self.base_freq * torch.arange(
                min_freq_log2, num_freq, 1
                )

        self.bands = self.bands.to(dtype=torch.float32) # [num_freq,]

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(
        self,
        coords: torch.Tensor,
        with_batch: bool = True
    ) -> torch.Tensor:
        
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        
        if with_batch:
            N = coords.shape[0]
            winded = (coords[...,None, :] * self.bands[None,None,:,None]).reshape(
                N, coords.shape[1], coords.shape[-1] * self.num_freq)
            encoded = torch.cat([torch.sin(winded*2*torch.pi), torch.cos(winded*2*torch.pi)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)

        else:
            N = coords.shape[0]
            winded = (coords[:, None] * self.bands[None, :, None]).reshape(
                N, coords.shape[1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded*2*torch.pi), torch.cos(winded*2*torch.pi)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """A human readable name for the given wisp module."""
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
        }