import logging

from typing import Union
from omegaconf import DictConfig
from pathlib import Path

import numpy as np

import torch
from torch import linalg as LA
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.scaler import CustomStandardScaler
from src.modules.inr import ModulatedFourierFeatures, ModulatedWaveletINR
from src.tools.utils.scheduler import get_scheduler
from src.tools.utils.mask import std_mask
from src.tools.utils.plot import plot_losses, plot_loss_steps
from src.metalearning.metalearning import outer_step

def train_fn(
    inr: Union[ModulatedFourierFeatures, ModulatedWaveletINR],
    train_loader: DataLoader,
    cfg: DictConfig,
    path_results_exp: Path
) -> float:
    """
    Main function for training an INR model.

    Parameters:
        inr (ModulatedFourierFeatures): INR network to train
        train_loader (DataLoader): train dataloader
        cfg (DictConfig): yaml config (meta file with all dependencies)
        path_results_exp (Path): path where to export losses & checkpoints

    Returns:
        mae_loss (float): train MAE loss
    """

    logger = logging.getLogger(__name__)

    # get config:
    cfg_optim = cfg.optim
    cfg_train = cfg.trainer

    # build dir to export losses and checkpoints:
    save_loss_path = path_results_exp / 'loss'
    save_ckpt_path = path_results_exp / 'ckpt'
    save_loss_path.mkdir(exist_ok=True)
    save_ckpt_path.mkdir(exist_ok=True)

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() and cfg_train.accelerator=='gpu' else 'cpu')

    # move model to device:
    inr = inr.to(device)
    
    # define optimizer:
    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters(), "lr": cfg_optim.lr_inr, "weight_decay": 0.},
        ],
        lr=cfg_optim.lr_inr,
        weight_decay=0,
    )

    # define scheduler:
    scheduler = get_scheduler(cfg_optim.scheduler, optimizer)

    # get trade-off coeff in loss function (outer step):
    use_target_for_training = cfg_optim.use_target_for_training
    lambda_target           = cfg_optim.get('lambda_target', 0.0)
    if not use_target_for_training:
        lambda_target  = 0.0
        lambda_context = 1.0
    else:
        lambda_context = 1.0 - lambda_target
    assert lambda_target >= 0
    assert lambda_context >= 0
    logger.info('[Training] use target grid for training: {}, lambda = {:}'.format(use_target_for_training, lambda_target))

    best_loss = np.inf
    best_fit  = np.inf

    l_epochs   = []
    l_mse_loss = []
    l_mae_loss = []
    l_steps    = []

    modulations_norm = []

    global_steps     = 0
    max_global_steps = cfg_train.get('max_steps', 0)
    logger.info('[Training] Max number of epochs {:,d} / steps {:,d}'.format(cfg_train.max_epochs, 0 if max_global_steps is None else max_global_steps))
    if (max_global_steps is None) or (max_global_steps <=0):
        max_global_steps = np.inf
        schedule_on_steps = False
    else:
        schedule_on_steps = True

    # iterate through the epochs (one step = one pass of the train loader):
    patience = 0

    for step in range(cfg_train.max_epochs):

        fit_train_mae, fit_train_mse = 0, 0
        inference_train_mae          = 0

        ntrain = 0

        # iterate through the train dataloader:
        for substep, batch in enumerate(train_loader):
            
            inr.train()
            
            series_c    = batch.get('context_features').to(device)    # [N, T_in, 1] (context values)
            series_t    = batch.get('target_features').to(device)    # [N, T, 1] (target values)
            
            # z-normalize series (fit on context, transform on context + target):
            scaler = CustomStandardScaler(dim=1,epsilon=1e-7)
            scaler.fit(series_c)
            series_c = scaler.transform(series_c)
            series_t = scaler.transform(series_t)

            # remove series constant at 0:
            mask_flat   = (scaler.std > 1e-6).squeeze((1,2))
            series_c    = series_c[mask_flat]
            series_t    = series_t[mask_flat]

            # load the rest of the batch:
            modulations = batch.get('modulations').to(device)[mask_flat]         # [N, h]    (modulations)
            coords_c    = batch.get('context_coordinates').to(device)[mask_flat] # [N, T_in, 1] (context grid coordinates)
            coords_t    = batch.get('target_coordinates').to(device)[mask_flat]  # [N, T, 1] (target grid coordinates)
            mask        = batch.get('is_missing_mask').to(device)[mask_flat]     # [N, T, 1] (where the samples are not observed)

            # if apply_mask_during_training, remove samples with very small std:
            if cfg_optim.apply_mask_during_training:

                _, mask_c = std_mask(series_c, threshold=8.0, return_mask=True)
                _, mask_t = std_mask(series_t, threshold=8.0, return_mask=True)
                mask_std  = torch.logical_and(mask_c, mask_t)
                
                series_c, coords_c, series_t, coords_t, modulations = series_c[mask_std], coords_c[mask_std], series_t[mask_std], coords_t[mask_std], modulations[mask_std]
                mask = mask[mask_std]
                
            n_samples = series_c.shape[0]
            if n_samples ==0:
                continue

            # perform outer + inner loops:
            # (inner steps hidden in outer loop but the gradient descent of the outer loop is not performed)
            outputs = outer_step(
                inr,
                context_coordinates     = coords_c,
                context_features        = series_c,
                target_coordinates      = coords_t,
                target_features         = series_t,
                inner_steps             = cfg_optim.inner_steps,
                inner_lr                = cfg_optim.lr_code,
                is_train                = True,
                return_reconstructions  = True,
                gradient_checkpointing  = False,
                loss_type               = cfg_optim.get('loss_type','mse'),
                modulations             = torch.zeros_like(modulations),
                use_target_for_training = use_target_for_training
            )

            # compute the loss function (outer step):
            loss = lambda_context * outputs["loss_c"]
            if use_target_for_training:
                loss += lambda_target * outputs['loss_t']

            # do gradient descent wrt all parameters (outer step optimization):
            optimizer.zero_grad()
            loss.backward(create_graph=False)

            # gradient clipping:
            nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            # if isinstance(inr, ModulatedFourierFeatures):
            #     nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            # else:
            #     # torch.clamp()
            #     nn.utils.clip_grad_value_([p for p in inr.parameters() if p.requires_grad and p.grad.dtype != torch.cfloat], clip_value=1.0)
            #     # print( [p.grad.dtype == torch.cfloat for p in inr.parameters() if p.requires_grad][:3] )
            #     # print( [p for p in inr.parameters() if p.requires_grad][:3] )

            # optmizer step:
            optimizer.step()

            # detach loss:
            loss = loss.cpu().detach()
            
            # get MAE on context grid only:
            with torch.set_grad_enabled(False):
                loss_samples_mae = torch.nn.L1Loss()(outputs.get('yhat_c'), series_c).cpu()
                fit_train_mae += loss_samples_mae.item() * n_samples

            # get MAE on target grid only:
            with torch.set_grad_enabled(False):
                target_reco = outputs['yhat_t'] if cfg_optim.use_target_for_training else inr.modulated_forward(coords_t, outputs['modulations'])
                loss_mae_inference = torch.abs(target_reco - series_t)[mask].mean()

            # compute norms:
            mod_norm = LA.vector_norm(outputs['modulations'], dim=1).mean().cpu().detach()
            modulations_norm.append(mod_norm.item())

            # store step loss:
            l_steps.append(loss.item())

            # update total loss:
            fit_train_mse       += loss.item() * n_samples
            inference_train_mae += loss_mae_inference.cpu().detach().item() * n_samples
            
            ntrain       += n_samples
            global_steps += 1
            if global_steps > max_global_steps:
                break

            if global_steps % 500 == 0:
                plot_loss_steps(
                    l_steps,
                    filename='loss_step_{}'.format(cfg.data.name.lower()),
                    save_path=save_loss_path
                )
                plot_loss_steps(
                    modulations_norm,
                    filename='modulations_step_{}'.format(cfg.data.name.lower()),
                    save_path=save_loss_path
                )

            if schedule_on_steps:
                scheduler.step()

        if ntrain == 0:
            continue
        
        mse_loss = fit_train_mse / (ntrain)
        mae_loss = fit_train_mae / (ntrain)

        inference_train_mae /= ntrain

        l_epochs.append(step)
        l_mse_loss.append(mse_loss)
        l_mae_loss.append(mae_loss)

        if (cfg_train.max_epochs // 200 == 0) or step % ( cfg_train.max_epochs // 200 ) == 0:
            logger.info('[Training] Epoch {:03d}, MAE on context is {:.3f}, MAE on target values is {:.3f}'.format(step,mae_loss,inference_train_mae))

        if (step % cfg.callbacks.plot_freq == 0) or (step == cfg_train.max_epochs - 1):
            plot_losses(
                l_epochs,
                l_mse_loss,
                l_mae_loss,
                filename='loss_epoch_{}'.format(cfg.data.name.lower()),
                save_path=save_loss_path
            )

        if not schedule_on_steps:
            scheduler.step()

        if mse_loss < best_fit:
            best_fit = mse_loss
            torch.save(
                {
                    "data": cfg.data,
                    "cfg_inr": cfg.inr,
                    "epoch": cfg_train.max_epochs,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "train_loss": mse_loss,
                    "global_step": global_steps
                },
                save_ckpt_path / 'best_fit.pt'
            )

        if inference_train_mae < best_loss:
            best_loss = inference_train_mae

            if cfg_train.enable_ckpt:
                torch.save(
                    {
                        "data": cfg.data,
                        "cfg_inr": cfg.inr,
                        "epoch": cfg_train.max_epochs,
                        "inr": inr.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "train_loss": mse_loss,
                        "global_step": global_steps
                    },
                    save_ckpt_path / 'best.pt'
                )

            patience = 0

        else:
            if cfg_optim.get('patience', -1) > 0:
                patience += 1
                logger.info(f"[Training] Patience is {patience}")

        if (cfg_optim.get('patience', -1) > 0) and (patience >= cfg_optim.get('patience', -1)):
            break

        if global_steps > max_global_steps:
            break
            

    logger.info('[Training] end of training after {:,d} steps'.format(global_steps))
    return mae_loss

