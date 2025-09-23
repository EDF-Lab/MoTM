import logging
from collections import defaultdict

from typing import Dict, Union, Optional, List, Any
from pathlib import Path

import pandas as pd
import numpy as np 

from einops import rearrange
import torch
from torch.utils.data import DataLoader

from src.modules.inr import ModulatedFourierFeatures
from src.data.scaler import CustomStandardScaler
from src.metalearning.metalearning import outer_step
from src.tools.utils.plot import make_imputation_mixture_plots
from src.tools.utils.estimate_lambda import crossval_lambda_fn
from src.modules.ridge.ridge import RidgeRegressor
from src.baselines import interpolate_series_torch, impute_by_offset

def infer_fn_mixture(
    list_inr: List[ModulatedFourierFeatures],
    test_loader: DataLoader,
    dict_params: Dict[str, List[Union[int, float, str, bool]]],
    path_results_exp: Path,
    inr_last_layers: int = 1,
    lambda_ridge: float = 0.1,
    share_ridge_weights: bool = False,
    learn_lambda: bool = False,
    baseline_offset: int = 24,
    plot_imputation: bool = True,
    nb_plots: int = 3,
    export_outputs: bool = False,
    max_points_to_plot: int = 335,
    is_block_setting: bool = False,
    run_baselines: bool = True
) -> Dict[str, Union[float, torch.Tensor]]:
    
    """
    Main function for running mixture of INR at inference.

    Parameters:
        list_inr (List[ModulatedFourierFeatures]): list of INR networks to evaluate
        test_loader (DataLoader): test dataloader
        dict_params (Dict[str, Any]): dict with keys inner_steps, inner_lr, loss_types and corresp. values as lists
        path_results_exp (Path): path where to export results
        inr_last_layers (int): number of last INR layers to use as features
        lambda_ridge (float): ridge regularization
        share_ridge_weights (bool): whether to fit one ridge per sample (False) or one per batch (True)
        learn_lambda (bool): whether to estimate lambda from context data
        baseline_offset (int): number of steps for the offset baseline
        plot_imputation (bool): make imputation plots or not
        nb_plots (int): number of samples to plot
        export_outputs (bool): if True, export reconstruction, grids etc.
        max_points_to_plot (int): max number of grid points for reconstruction plots
        is_block_setting (bool): if True, shade imputation block in plots
        run_baselines (bool): whether to estimate the Linear and Repeat baselines

    Returns:
        results (dict): dict of outputs with following items:
            - mae_on_missing_values (float): MAE on all target timesteps with NaN (if ground truth provided)
            - mae_on_observed_values (float): MAE on all target timesteps with no NaN
    """

    logger = logging.getLogger(__name__)

    if learn_lambda:
        list_fit_lambda = crossval_lambda_fn(
            list_inr          = list_inr,
            test_loader       = test_loader,
            dict_params       = dict_params,
            path_results_exp  = path_results_exp,
            inr_last_layers   = inr_last_layers,
            lambda_ridge_init = lambda_ridge,
            rr_lr             = 1.0,
            rr_steps          = 10,
            val_frac          = 0.5
        )
        lambda_ridge = torch.cat(list_fit_lambda).max().item()
        logger.info('[Inference] Learning lambda (max aggregation): {:.3f}'.format(lambda_ridge))
        logger.info('[Inference] FYI lambda min: {:.3f} - mean: {:.3f}'.format(
            torch.cat(list_fit_lambda).min().item(),
            torch.cat(list_fit_lambda).mean().item()
        ))
    
    list_inner_steps = dict_params.get('inner_steps')
    list_inner_lr    = dict_params.get('inner_lr')
    list_loss_types  = dict_params.get('loss_type')

    M = len(list_inr)
    assert len(list_inner_steps) == M
    assert len(list_inner_lr) == M
    assert len(list_loss_types) == M

    list_names = [str(i+1) for i in range(M)]
    if run_baselines:
        list_names_all = list_names + ['Ridge', 'EM', 'Naive', 'Linear', 'Offset']
    else:
        list_names_all = list_names + ['Ridge', 'EM', 'Naive']

    infer_path = Path(path_results_exp) / 'inference'
    infer_path.mkdir(exist_ok=True)

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move model to device:
    for inr in list_inr:
        inr.to(device)

    results   = defaultdict(list)
    materials = defaultdict(list)

    # instantiate ridge:
    ridge_adaptive_weights = RidgeRegressor(lambda_regu=lambda_ridge).to(device)
    logger.info('[Inference] Ridge regression with reg lambda = {:.2f}'.format(lambda_ridge))

    chunks = 0    
    # iterate through the train dataloader:
    for substep, batch in enumerate(test_loader):

        series_c    = batch.get('context_features').to(device)    # [N, T_in, 1] (context values)
        series_t    = batch.get('target_features').to(device)     # [N, T, 1] (target values)

        # check if series_t (ground truth) exists
        series_t_exists = isinstance(series_t, torch.Tensor)

        for inr in list_inr:
            inr.to(device)

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-7)
        scaler.fit(series_c)
        series_c = scaler.transform(series_c)
        
        # load the rest of the batch:
        modulations = batch.get('modulations').to(device)         # [N, h]    (modulations)
        coords_c    = batch.get('context_coordinates').to(device) # [N, T_in, 1] (context grid coordinates)
        coords_t    = batch.get('target_coordinates').to(device)  # [N, T, 1] (target grid coordinates)
        mask        = batch.get('is_missing_mask').to(device)     # [N, T, 1] (where the samples are not observed)

        # alloc some space:
        list_hidden_context = []
        list_hidden_target  = []
        list_interpo        = []
        list_losses_context = []
        list_pred_context   = []

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
                target_coordinates      = coords_t,
                target_features         = None,
                inner_steps             = inner_steps,
                inner_lr                = inner_lr,
                is_train                = False,
                return_reconstructions  = True,
                gradient_checkpointing  = False,
                loss_type               = loss_type,
                modulations             = torch.zeros_like(modulations),
                use_target_for_training = True
            )

            # get loss function:
            loss_fn = torch.nn.L1Loss(reduction="none")
            
            # get reconstructed features on the target grid:
            interpo     = outputs['yhat_t']
            modulations = outputs['modulations']
            
            list_interpo.append(interpo)

            ################
            # Mixture INR  #
            ################

            # Utiliser ici les inr et les modulations, il faut coder le modulated_forward_mixture
            # Voir le code du ridge

            # modulated_forward on both INR 
            hidden_context = inr.modulated_forward_mixture(coords_c, modulations, n_last_layers = inr_last_layers) # (N, T_in, nlayers, H_1)

            # Make prediction on target grid 
            hidden_target = inr.modulated_forward_mixture(coords_t, modulations, n_last_layers = inr_last_layers)

            #  
            yhat_context = inr.layers[-1](hidden_context[...,-1,:])    # (N, T_in, 1)
            l_context    = loss_fn( series_c, yhat_context )           # (N, T_in, 1)

            # flatten intermediate activations:
            hidden_context = rearrange(hidden_context, 'b T n d -> b T (n d)')   # [N, T_in, nlayers x hidden_dim]
            hidden_target  = rearrange(hidden_target, 'b T n d -> b T (n d)')

            list_hidden_context.append( hidden_context )
            list_hidden_target.append( hidden_target )
            list_losses_context.append( -l_context )
            list_pred_context.append( yhat_context )

        # (a) mixture 1 = ridge on hidden context:

        # build context features for ridge adaptation:
        H_context = torch.cat( list_hidden_context, dim=2 )   # (N, T_in, H_1 + H_2)

        # fit the Ridge:
        W, b = ridge_adaptive_weights.get_weights(X=H_context, Y=series_c, share_weights = share_ridge_weights)

        # Build target features for ridge:
        H_target = torch.cat( list_hidden_target, dim=2 )

        # predict:
        interpo_mixture = torch.matmul(H_target, W) + b

        # (b) mixture 2 = weighted sum of predictions:

        L_context = torch.cat( list_losses_context, dim=-1 )         # (N, T_in, 2)
        weights_context = torch.nn.Softmax(dim=2)(L_context)         # (N, T_in, 2)
        weights_context = weights_context.mean(dim=1,keepdims=True)  # (N, 1, 2)

        Y_target   = torch.cat( list_interpo, dim=-1 )                    # (N, T_out, 2)
        interpo_em = ( Y_target * weights_context ).sum(-1,keepdims=True) # (N, T_out, 1)

        # (c) baseline 3 = ridge on model outputs:

        # fit ridge on model outputs:
        Y_context   = torch.cat( list_pred_context, dim=-1 )                    # (N, T_in, 2)
        W, b = ridge_adaptive_weights.get_weights(X=Y_context, Y=series_c)
        interpo_ridge_naive = torch.matmul(Y_target, W) + b

        list_interpo_2 = [interpo_mixture, interpo_em, interpo_ridge_naive]

        chunks += interpo_mixture.shape[0]

        ################
        # save results #
        ################

        # Compute normalize score 
        if series_t_exists:
            series_t = series_t.to(device)
            series_t = scaler.transform(series_t)
            
            if run_baselines:
                # linear interpolation:
                interpo_linear = interpolate_series_torch( series_t.squeeze(), mask.squeeze() )
                list_interpo_2.append(interpo_linear)
                
                # offset:
                interpo_offset = impute_by_offset( series_t.squeeze(), mask.squeeze(), step=baseline_offset )
                list_interpo_2.append(interpo_offset)

            list_interpo_inv = []
            for interpo, model_name in zip(
                list_interpo + list_interpo_2,
                list_names_all
            ):
                error_normalize             = torch.abs(interpo - series_t)
                mae_on_missing_values_norm  = error_normalize[mask]
                mae_on_observed_values_norm = error_normalize[~mask]
                mse_on_missing_values_norm  = error_normalize[mask]**2
                mse_on_observed_values_norm = error_normalize[~mask]**2
                materials['interpo_{}_norm'.format(model_name)].append(interpo.detach().cpu().squeeze())

                # undo z-normalization (back to data space):
                list_interpo_inv.append( scaler.inv_transform(interpo) )

                if len(mae_on_missing_values_norm) > 0:
                    results['mae_on_missing_values_norm_{}'.format(model_name)].append(mae_on_missing_values_norm.detach().cpu())
                    results['mse_on_missing_values_norm_{}'.format(model_name)].append(mse_on_missing_values_norm.detach().cpu())
                if len(mae_on_observed_values_norm) > 0:
                    results['mae_on_observed_values_norm_{}'.format(model_name)].append(mae_on_observed_values_norm.detach().cpu())
                    results['mse_on_observed_values_norm_{}'.format(model_name)].append(mse_on_observed_values_norm.detach().cpu())
            
        materials['series_c_norm'].append(series_c.detach().cpu().squeeze())
        materials['series_t_norm'].append(series_t.detach().cpu().squeeze())

        # compute errors on missing values (if ground truth is provided):
        if series_t_exists:
            series_t = scaler.inv_transform(series_t).to(device) # [N, T, 1]
            mask     = mask.to(device)                           # [N, T, 1]

            for interpo, model_name in zip(
                list_interpo_inv,
                list_names_all
            ):
                error_normalize        = torch.abs(interpo - series_t)
                mae_on_missing_values  = error_normalize[mask]
                mae_on_observed_values = error_normalize[~mask]
                mse_on_missing_values  = error_normalize[mask]**2
                mse_on_observed_values = error_normalize[~mask]**2
                materials['interpo_{}'.format(model_name)].append(interpo.detach().cpu().squeeze())

                if len(mae_on_missing_values) > 0:
                    results['mae_on_missing_values_{}'.format(model_name)].append(mae_on_missing_values.detach().cpu())
                    results['mse_on_missing_values_{}'.format(model_name)].append(mse_on_missing_values.detach().cpu())
                if len(mae_on_observed_values) > 0:
                    results['mae_on_observed_values_{}'.format(model_name)].append(mae_on_observed_values.detach().cpu())
                    results['mse_on_observed_values_{}'.format(model_name)].append(mse_on_observed_values.detach().cpu())
            
            # results['gt'].append(series_t)

                if plot_imputation:
                    materials['interpo_{}'.format(model_name)].append(interpo.detach().cpu().squeeze())
            
            materials['coords_c'].append(coords_c.detach().cpu().squeeze())
            materials['series_c'].append(scaler.inv_transform(series_c.detach()).cpu().squeeze())
            materials['coords_t'].append(coords_t.detach().cpu().squeeze())
            materials['mask'].append(mask.detach().cpu().squeeze())

            if series_t_exists:
                materials['series_t'].append(series_t.detach().cpu().squeeze())

    # agregate all results:
    results = {key:torch.cat(val,dim=0).cpu().detach() if len(val)>0 else val for key,val in results.items()}
    results['chunks'] = chunks
    results = {key:val.nanmean().cpu().detach().item() if ('mae' in key or 'mse' in key) and len(val)>0 else val for key,val in results.items()}

    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)
    logger.info('[Inference] end of inference loop')

    # export metrics:
    df = pd.DataFrame(
        {
            'chunks': results['chunks'],
            'Model': [name if len(name)>1 else 'TimeFlow_'+str(name) for name in list_names_all],
            'MAE on Missing Values': [results.get('mae_on_missing_values_{}'.format(model_name)) for model_name in list_names_all],
            'MAE on Observed Values': [results.get('mae_on_observed_values_{}'.format(model_name)) for model_name in list_names_all],
            'MAE on Missing Values (norm)': [results.get('mae_on_missing_values_norm_{}'.format(model_name)) for model_name in list_names_all],
            'MAE on Observed Values (norm)': [results.get('mae_on_observed_values_norm_{}'.format(model_name)) for model_name in list_names_all],
        }
    )
    df.set_index('Model', inplace=True)
    df.to_csv(save_dir_metrics / 'mae_mixture.csv')

    df = pd.DataFrame(
        {
            'chunks': results['chunks'],
            'Model': [name if len(name)>1 else 'TimeFlow_'+str(name) for name in list_names_all],
            'MSE on Missing Values': [results.get('mse_on_missing_values_{}'.format(model_name)) for model_name in list_names_all],
            'MSE on Observed Values': [results.get('mse_on_observed_values_{}'.format(model_name)) for model_name in list_names_all],
            'MSE on Missing Values (norm)': [results.get('mse_on_missing_values_norm_{}'.format(model_name)) for model_name in list_names_all],
            'MSE on Observed Values (norm)': [results.get('mse_on_observed_values_norm_{}'.format(model_name)) for model_name in list_names_all],
        }
    )
    df.set_index('Model', inplace=True)
    df.to_csv(save_dir_metrics / 'mse_mixture.csv')
    
    # concat batches outputs
    for key in materials:
        # stack batches together
        materials[key] = torch.cat(materials[key], dim=0) 

    if plot_imputation:
        make_imputation_mixture_plots(
            materials, 
            nb_plots, 
            infer_path, 
            series_t_exists,
            max_points_to_plot,
            is_blockwise=is_block_setting
        )

    # if export_outputs:
    #     save_dir_materials = infer_path / 'materials'
    #     save_dir_materials.mkdir(exist_ok=True)
    #     torch.save(materials, save_dir_materials / 'materials.pt')

    return results
