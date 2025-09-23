from typing import Dict, Any, Union

import torch

import src.metalearning.losses as losses
from src.modules.inr import ModulatedFourierFeatures

def inner_loop(
    func_rep: ModulatedFourierFeatures,
    modulations: torch.Tensor,
    coordinates: torch.Tensor,
    features: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
    is_train: bool = False,
    gradient_checkpointing: bool = False,
    loss_type: str = "mse",
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Performs multiple gradient descent steps to refine the modulation parameters.

    Args:
        func_rep (ModulatedFourierFeatures): The implicit neural representation model.
        modulations (torch.Tensor): Initial modulation parameters of shape [*, h].
        coordinates (torch.Tensor): Input coordinates of shape [*, T, 1].
        features (torch.Tensor): Target features of shape [*, T, 1].
        inner_steps (int): Number of inner optimization steps.
        inner_lr (float): Learning rate for inner optimization.
        is_train (bool, optional): Whether in training mode. Defaults to False.
        gradient_checkpointing (bool, optional): Enables gradient checkpointing to save memory. Defaults to False.
        loss_type (str, optional): Type of loss function ('mse' or 'pinball'). Defaults to 'mse'.
        quantile (float, optional): Quantile parameter for pinball loss. Defaults to 0.5.

    Returns:
        torch.Tensor: Updated modulation parameters of shape [*, h].
    """
    fitted_modulations = modulations
    
    for _ in range(inner_steps):
        fitted_modulations = inner_loop_step(
            func_rep,
            fitted_modulations,
            coordinates,
            features,
            inner_lr,
            is_train,
            gradient_checkpointing,
            loss_type,
            quantile,
        )
    
    return fitted_modulations

def inner_loop_step(
    func_rep: ModulatedFourierFeatures,
    modulations: torch.Tensor,
    coordinates: torch.Tensor,
    features: torch.Tensor,
    inner_lr: float,
    is_train: bool = False,
    gradient_checkpointing: bool = False,
    loss_type: str = "mse",
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Performs a single gradient descent step on the modulation parameters.

    Args:
        func_rep (ModulatedFourierFeatures): The implicit neural representation model.
        modulations (torch.Tensor): Current modulation parameters of shape [*, h].
        coordinates (torch.Tensor): Input coordinates of shape [*, T, 1].
        features (torch.Tensor): Target features of shape [*, T, 1].
        inner_lr (float): Learning rate for inner step optimization.
        is_train (bool, optional): Whether in training mode. Defaults to False.
        gradient_checkpointing (bool, optional): Enables gradient checkpointing. Defaults to False.
        loss_type (str, optional): Type of loss function ('mse' or 'pinball'). Defaults to 'mse'.
        quantile (float, optional): Quantile parameter for pinball loss. Defaults to 0.5.

    Returns:
        torch.Tensor: Updated modulation parameters of shape [*, h].
    """
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = features.shape[0]
    
    # get loss function:
    if loss_type.lower() == 'mse':
        loss_fn = lambda x1, x2: losses.batch_pointwise_fn(x1,x2, loss_name='MSE')
    elif loss_type.lower() == 'huber':
        loss_fn = lambda x1, x2: losses.batch_pointwise_fn(x1,x2, loss_name='Huber')
    else:
        loss_fn = losses.Batch_SmoothPinballLossTorch(quantile)
    
    with torch.enable_grad():
        recon_features = func_rep.modulated_forward(coordinates, modulations)
        loss = loss_fn(recon_features, features).mean() * batch_size
        grad = torch.autograd.grad(loss, modulations, create_graph=is_train and not detach)[0]

    return modulations - inner_lr * grad

def outer_step(
    func_rep: ModulatedFourierFeatures,
    context_coordinates: torch.Tensor,
    context_features: torch.Tensor,
    target_coordinates: torch.Tensor,
    target_features: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
    is_train: bool = False,
    return_reconstructions: bool = False,
    gradient_checkpointing: bool = False,
    loss_type: str = "mse",
    quantile: float = 0.5,
    modulations: torch.Tensor = None,
    use_target_for_training: bool = False
) -> Dict[str, Any]:
    """
    Performs an outer optimization step by refining the modulation parameters and computing reconstruction loss.

    Args:
        func_rep (ModulatedFourierFeatures): The implicit neural representation model.
        context_coordinates (torch.Tensor): Context coordinates of shape [*, T_in, 1].
        context_features (torch.Tensor): Context features of shape [*, T_in, 1].
        target_coordinates (torch.Tensor): Target coordinates of shape [*, T, 1].
        target_features (torch.Tensor): Target features of shape [*, T, 1]. Can be None.
        inner_steps (int): Number of inner loop optimization steps.
        inner_lr (float): Learning rate for inner optimization.
        is_train (bool, optional): Whether in training mode. Defaults to False.
        return_reconstructions (bool, optional): Whether to return reconstructions. Defaults to False.
        gradient_checkpointing (bool, optional): Enables gradient checkpointing. Defaults to False.
        loss_type (str, optional): Type of loss function ('mse' or 'pinball'). Defaults to 'mse'.
        quantile (float, optional): Quantile parameter for pinball loss. Defaults to 0.5.
        modulations (torch.Tensor, optional): Initial modulation parameters. Defaults to None.
        use_target_for_training (bool, optional): Whether to use separate target data. Defaults to False.

    Returns:
        Dict[str, Any]: Dictionary containing loss, modulations and optionally reconstructions.
    """
    # get loss function:
    if loss_type.lower() == 'mse':
        loss_fn = lambda x1, x2: losses.batch_pointwise_fn(x1,x2, loss_name='MSE')
    elif loss_type.lower() == 'huber':
        loss_fn = lambda x1, x2: losses.batch_pointwise_fn(x1,x2, loss_name='Huber')
    else:
        loss_fn = losses.Batch_SmoothPinballLossTorch(quantile)
    
    func_rep.zero_grad()
    
    modulations = modulations.requires_grad_()

    context_coordinates = context_coordinates.clone() # [*, T, 1]
    context_features    = context_features.clone()    # [*, T, 1]
    
    # run inner loop (adapt modulations to local context):
    modulations = inner_loop(
        func_rep,
        modulations,
        context_coordinates,
        context_features,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
        quantile,
    )
    
    loss_t = None
    # outer step on the context + target coordinates:
    with torch.set_grad_enabled(is_train):

        # get loss on context grid:
        yhat_c = func_rep.modulated_forward(context_coordinates, modulations)
        loss_c = loss_fn(context_features, yhat_c).mean()

        # get loss on target grid if required:
        if use_target_for_training:
            yhat_t = func_rep.modulated_forward(target_coordinates, modulations)
            loss_t = loss_fn(yhat_t, target_features).mean() if isinstance(target_features, torch.Tensor) else None

    # store outputs in a dict:    
    outputs = {
        "loss_c"      : loss_c,
        "loss_t"      : loss_t,
        "modulations" : modulations,
        "yhat_c"      : None if not return_reconstructions else yhat_c[-1] if "multiscale" in loss_type else yhat_c,
        "yhat_t"      : None if (not return_reconstructions or not use_target_for_training) else yhat_t[-1] if "multiscale" in loss_type else yhat_t
    }
    
    return outputs
