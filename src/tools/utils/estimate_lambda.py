import logging
from collections import defaultdict

from typing import Dict, Union, Optional, List, Any, Tuple
from pathlib import Path

import pandas as pd
import numpy as np 

from einops import rearrange
import torch
from torch.utils.data import DataLoader

from src.modules.inr import ModulatedFourierFeatures
from src.data.scaler import CustomStandardScaler
from src.metalearning.metalearning import outer_step

class LearnableRidgeRegressor(torch.nn.Module):

    def __init__(self, lambda_init: torch.Tensor) -> None:
        super().__init__()
        self.set_lambda(lambda_init)
    
    def set_lambda(self, lambda_init: torch.Tensor) -> None:
        self._lambda = torch.nn.Parameter(lambda_init.reshape(-1,1))

    def forward(self, x_target: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lambda_rr =  torch.nn.functional.softplus(self._lambda)
        W,b = self.get_weights(x,y, lambda_rr)
        yhat = torch.matmul(x_target, W) + b
        return yhat

    def get_weights(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        lambda_rr: torch.Tensor
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        batch_size, context_size, mixture_dim = X.shape
        ones = torch.ones(batch_size, context_size, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if context_size >= mixture_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(lambda_rr)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(lambda_rr)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))
        return weights[:, :-1], weights[:, -1:]
        
def fit_lambda(
    H_context: torch.Tensor,
    series_c: torch.Tensor,
    H_val: torch.Tensor,
    series_val: torch.Tensor,
    lambda_init: float,
    rr_lr: float = 1.0,
    rr_steps: int = 5,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Estimate a ridge regularization coefficient by gradient descent minimizing the loss on a validation split.

    Args:
        H_context (torch.Tensor): tensor of context features of shape `(B, T, d)`
        series_c (torch.Tensor): target values on the context grid, shape `(B, T, 1)`
        H_val (torch.Tensor): tensor of validation features of shape `(B', T', d)`
        series_val (torch.Tensor): target values on the validation grid, shape `(B', T', 1)`
        lambda_init (float): inital value of the ridge reg coeff
        rr_lr (flaot): learning rate
        rr_steps (int): number of gradient steps
        device (torch.device): torch device

    Returns:
        The estimated regularization coefficient as a torch Tensor
    """
    
    lambda_rr = torch.tensor(lambda_init, dtype=torch.float).to(device)

    for _ in range(rr_steps):
        lambda_rr = fit_lambda_step(
            H_context  = H_context,
            series_c   = series_c,
            H_val      = H_val,
            series_val = series_val,
            lambda_rr  = lambda_rr,
            rr_lr      = rr_lr,
            device     = device
        )
    
    unscaled_lambda_rr = torch.nn.functional.softplus(lambda_rr)

    return unscaled_lambda_rr

def fit_lambda_step(
    H_context: torch.Tensor,
    series_c: torch.Tensor,
    H_val: torch.Tensor,
    series_val: torch.Tensor,
    lambda_rr: torch.Tensor,
    rr_lr: float = 1.0,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Apply one step of gradient descent to learn a ridge regularization coefficient.

    Args:
        H_context (torch.Tensor): tensor of context features of shape `(B, T, d)`
        series_c (torch.Tensor): target values on the context grid, shape `(B, T, 1)`
        H_val (torch.Tensor): tensor of validation features of shape `(B', T', d)`
        series_val (torch.Tensor): target values on the validation grid, shape `(B', T', 1)`
        lambda_rr (float): last value of the ridge reg coeff
        rr_lr (flaot): learning rate
        device (torch.device): torch device

    Returns:
        The updated ridge reg coeff.
    """
    
    with torch.enable_grad():

        # instantiate ridge regressor:
        local_rr = LearnableRidgeRegressor(lambda_rr).to(device)

        # fit on context and predict on validation:
        reco = local_rr(H_val, H_context, series_c)

        # get validation loss:
        loss = torch.nn.MSELoss(reduction='none')(reco, series_val).mean() * len(reco)

        # compute grad:
        grad = torch.autograd.grad(loss, local_rr._lambda)[0]
        
        # update lambda:
        lambda_rr = lambda_rr - rr_lr * grad

    # print('Fit lambda,', torch.nn.functional.softplus(lambda_rr), loss)
    
    return lambda_rr

def crossval_lambda_fn(
    list_inr: List[ModulatedFourierFeatures],
    test_loader: DataLoader,
    dict_params: Dict[str, List[Union[int, float, str, bool]]],
    path_results_exp: Path,
    inr_last_layers: int = 1,
    lambda_ridge_init: float = 0.1,
    rr_lr: float = 1.0,
    rr_steps: int = 5,
    val_frac: float = 0.2
) -> List[torch.Tensor]:
    """
    Main routine to learn a ridge regularization to adapt a mixture of TimeFlow models.
    Solved by minimizing the reconstruction loss on a held-out validation set with gradient descent.

    Args:
        list_inr (List[ModulatedFourierFeatures]): list of INR networks forming the base
        test_loader (DataLoader): test dataloader
        dict_params (Dict[str, Any]): dict with keys inner_steps, inner_lr, loss_types and corresp. values as lists
        path_results_exp (Path): *to delete or use* path where to export results, not used
        inr_last_layers (int): number of last INR layers to use as features
        lambda_ridge_init (float): initialization of the ridge regularization coeff
        rr_lr (flaot): learning rate
        rr_steps (int): number of gradient steps
        val_frac (float): fraction of context points to use as the held-out validation set
    
    Returns:
        A list of estimated ridge reg coeff, one estimation per batch in test_loader.
    """

    list_inner_steps = dict_params.get('inner_steps')
    list_inner_lr    = dict_params.get('inner_lr')
    list_loss_types  = dict_params.get('loss_type')

    M = len(list_inr)
    assert len(list_inner_steps) == M
    assert len(list_inner_lr) == M
    assert len(list_loss_types) == M

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move model to device:
    for inr in list_inr:
        inr.to(device)
    
    list_lambdas = []
    for substep, batch in enumerate(test_loader):

        series_c    = batch.get('context_features').to(device)    # [N, T_in, 1] (context values)
        coords_c    = batch.get('context_coordinates').to(device) # [N, T_in, 1] (context grid coordinates)
        modulations = batch.get('modulations').to(device)         # [N, h]    (modulations)

        c_permute = torch.randperm(series_c.shape[1]).to(device)
        c_train   = c_permute[:int((1-val_frac)*len(c_permute))]
        c_val     = c_permute[int((1-val_frac)*len(c_permute)):]

        series_c_val = series_c[:,c_val].clone()
        coords_c_val = coords_c[:,c_val].clone()

        series_c = series_c[:,c_train].clone()
        coords_c = coords_c[:,c_train].clone()

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-7)
        scaler.fit(series_c)
        series_c = scaler.transform(series_c)
        series_c_val = scaler.transform(series_c_val)

        # alloc some space:
        list_hidden_context = []
        list_hidden_context_val = []

        for inr, inner_steps, inner_lr, loss_type in zip(
            list_inr, list_inner_steps, list_inner_lr, list_loss_types
        ):
            
            inr.eval()

            # perform outer + inner loops:
            # (inner steps hidden in outer loop but the gradient descent of the outer loop is not performed)
            outputs = outer_step(
                inr,
                context_coordinates     = coords_c,
                context_features        = series_c,
                target_coordinates      = coords_c,
                target_features         = None,
                inner_steps             = inner_steps,
                inner_lr                = inner_lr,
                is_train                = False,
                return_reconstructions  = False,
                gradient_checkpointing  = False,
                loss_type               = loss_type,
                modulations             = torch.zeros_like(modulations),
                use_target_for_training = False
            )
    
            # get reconstructed features on the target grid:
            modulations = outputs['modulations']

            # modulated_forward on both INR 
            hidden_context     = inr.modulated_forward_mixture(coords_c, modulations, n_last_layers = inr_last_layers) # (N, T_in, nlayers, H_1)
            hidden_context_val = inr.modulated_forward_mixture(coords_c_val, modulations, n_last_layers = inr_last_layers)

            # flatten intermediate activations:
            hidden_context     = rearrange(hidden_context, 'b T n d -> b T (n d)')   # [N, T_in, nlayers x hidden_dim]
            hidden_context_val = rearrange(hidden_context_val, 'b T n d -> b T (n d)')   # [N, T_in, nlayers x hidden_dim]
            
            list_hidden_context.append( hidden_context )
            list_hidden_context_val.append( hidden_context_val )
        
        # build context features for ridge adaptation:
        H_context = torch.cat( list_hidden_context, dim=2 )           # (N, T_in, H_1 + H_2)
        H_context_val = torch.cat( list_hidden_context_val, dim=2 )   # (N, T_in, H_1 + H_2)

        # mu_c          = H_context.mean(dim=1, keepdim=True)
        # sigma_c       = H_context.std(dim=1, keepdim=True) + 1e-6
        # H_context     = (H_context - mu_c) / sigma_c
        # H_context_val = (H_context_val - mu_c) / sigma_c

        crossval_lambda = fit_lambda(
            H_context, series_c, H_context_val, series_c_val,
            lambda_ridge_init, rr_lr, rr_steps,
            device
        )
        # lambda_ridge_init = crossval_lambda.item()

        list_lambdas.append( crossval_lambda )
        
    # all_lambdas = torch.cat(list_lambdas)

    # return all_lambdas.max()
    return list_lambdas