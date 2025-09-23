import torch
import torch.nn as nn

from src.modules.freq_embedding import NeRFEncoding, GaussianEncoding

class FourierFeatures(nn.Module):
    
    """
    INR for a single instance
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        num_frequencies = 8,
        width: int = 256,
        depth: int = 5,
        frequency_embedding: str = "nerf",
        include_input: bool = True,
        scale: int = 5,
        log_sampling: bool = False,
        min_frequencies: int = 0,
        max_frequencies: int = 32,
        base_frequency: float = 1.25
    ) -> None:
        
        super().__init__()

        self.frequency_embedding = frequency_embedding.lower() # type of frequency embedding (NeRF or Gaussian)
        self.include_input = include_input                     # whether to add coordinate (time) to freq embedding

        # (time) coordinate embedding with NeRF:
        if self.frequency_embedding == "nerf":
            self.embedding = NeRFEncoding(
                num_frequencies,
                min_frequencies,
                max_frequencies,
                log_sampling=log_sampling,
                include_input=include_input,
                input_dim=input_dim,
                base_freq=base_frequency,
            )
            self.in_channels = [self.embedding.out_dim] + [width] * (depth - 1)

        # or (time) coordinate embedding with Gaussian encoding:
        elif self.frequency_embedding == "gaussian":
            self.scale = scale
            self.embedding = GaussianEncoding(
                embedding_size=num_frequencies * 2, scale=scale, dims=input_dim
            )
            embed_dim = (
                num_frequencies * 2 + input_dim
                if include_input
                else num_frequencies * 2
            )
            self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [output_dim]
        
        # INR layers (linear):
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )

        # corresponding INR activations:
        self.activations = nn.ModuleList(
            [nn.ReLU() for k in range(depth-1)]
        )

        self.depth      = depth # depth of the network
        self.hidden_dim = width # hidden dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], axis=-1)

        for idx, l in enumerate(self.layers[:-1]):
            position = self.activations[idx](l(position))

        out = self.layers[-1](position)

        return out

if __name__ == "__main__":

    model = FourierFeatures()
    X = torch.rand(1, 256, 1)
    out = model(X)

