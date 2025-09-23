import torch
import torch.nn as nn
from torch.nn import functional


mse_fn = torch.nn.MSELoss()

per_element_mse_fn   = torch.nn.MSELoss(reduction="none")
per_element_huber_fn = torch.nn.HuberLoss(reduction="none")

def batch_pointwise_fn(
    x1: torch.Tensor,
    x2: torch.Tensor,
    loss_name: str = 'MSE'
) -> torch.Tensor:
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).
    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).
        loss_name (str): loss name, MSE or Huber
    Returns:
        MSE tensor of shape (batch_size,).
    """

    if loss_name.lower() == 'mse':
        per_element_mse = per_element_mse_fn(x1, x2)
    elif loss_name.lower() == 'huber':
        per_element_mse = per_element_huber_fn(x1, x2)

    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)

class SmoothPinballLossTorch(nn.Module):
    """
    Smoth version of the pinball loss function.

    Parameters
    ----------
    quantiles : torch.tensor
    alpha : int
        Smoothing rate.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(
        self,
        quantiles: torch.Tensor,
        alpha: float=0.01
    ) -> None:
        super(SmoothPinballLossTorch,self).__init__()
        self.pred = None
        self.targes = None
        self.quantiles = quantiles
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        q_error = self.quantiles * error
        beta = 1 / self.alpha
        soft_error = functional.softplus(-error, beta)

        losses = q_error + soft_error

        return losses

class Batch_SmoothPinballLossTorch(nn.Module):

    def __init__(self, quantiles: torch.Tensor) -> None:
        super(Batch_SmoothPinballLossTorch, self).__init__()

        self.pred = None
        self.targes = None
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        per_element_smooth_pinball = SmoothPinballLossTorch(self.quantiles)(pred, target)

        return per_element_smooth_pinball.view(pred.shape[0], -1).mean(dim=1)


