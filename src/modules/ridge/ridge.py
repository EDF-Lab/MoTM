from typing import Tuple

from einops import rearrange, repeat
import torch
import torch.nn as nn

class RidgeRegressor(nn.Module):

    def __init__(
        self,
        lambda_regu: float = 0.
    ) -> None:
        super().__init__()
        self.lambda_regu = lambda_regu.reshape(-1,1) if isinstance(lambda_regu,torch.Tensor) else lambda_regu

    def get_weights(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        share_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit ridge regression and return weight

        Args:
            X (torch.Tensor): `(batch_size, L, d)` tensor of observed features
            Y (torch.Tensor): `(batch_size, L, 1)` tensor of observed outputs
        
        Returns:
            A tuple of weights and biases tensors of respective shapes `(batch_size, d, 1)` and `(batch_size, 1, 1)`.
        """
        if share_weights:
            bs = X.shape[0]
            X  = rearrange(X, 'b T d -> 1 (b T) d')
            Y = rearrange(Y, 'b T 1 -> 1 (b T) 1')
        batch_size, context_size, mixture_dim = X.shape
        ones = torch.ones(batch_size, context_size, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if context_size >= mixture_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(self.lambda_regu)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(self.lambda_regu)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))
        
        if share_weights:
            weights = repeat(weights, '1 d 1 -> b d 1', b=bs)
        #w has shape (N, H, 1): b has shape (N, 1, 1)
        return weights[:, :-1], weights[:, -1:]