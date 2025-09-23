from collections import defaultdict

from typing import Dict, Union
from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.modules.inr import ModulatedFourierFeatures
from src.data.scaler import CustomStandardScaler
from src.metalearning.metalearning import outer_step
from src.tools.utils.plot import make_imputation_plots

def infer_fn(
    inr: ModulatedFourierFeatures,
    test_loader: DataLoader,
    inner_steps: int,
    inner_lr: float,
    path_results_exp: Path,
    loss_type: str = 'mse', 
    plot_imputation: bool = True,
    nb_plots: int = 3,
    export_outputs: bool = False,
    max_points_to_plot: int = 335
) -> Dict[str, Union[float, torch.Tensor]]:
    
    """
    Main function for running an INR model at inference.

    Parameters:
        inr (ModulatedFourierFeatures): INR network to evaluate
        test_loader (DataLoader): test dataloader
        inner_steps (int): number of steps in the inner loop
        inner_lr (float): learning rate of the inner loop
        path_results_exp (Path): path where to export results
        plot_imputation (bool): make imputation plots or not
        nb_plots (int): number of samples to plot
        export_outputs (bool): if True, export reconstruction, grids etc.

    Returns:
        results (dict): dict of outputs with following items:
            - mae_on_missing_values (float): MAE on all target timesteps with NaN (if ground truth provided)
            - mae_on_observed_values (float): MAE on all target timesteps with no NaN
            - reco (torch.Tensor): estimated features on the target grid
            - gt (torch.Tensor): ground truth features on the target grid, if provided
    """

    infer_path = path_results_exp #/ 'inference'
    infer_path.mkdir(exist_ok=True)

    # get device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move model to device:
    inr = inr.to(device)

    results   = defaultdict(list)
    materials = defaultdict(list)

    # iterate through the train dataloader:
    for substep, batch in enumerate(test_loader):

        series_c    = batch.get('context_features').to(device)    # [N, T_in, 1] (context values)
        series_t    = batch.get('target_features').to(device)     # [N, T, 1] (target values)

        # check if series_t (ground truth) exists
        series_t_exists = isinstance(series_t, torch.Tensor)

        inr.eval()

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-7)
        scaler.fit(series_c)
        series_c = scaler.transform(series_c)

        mask_flat   = (scaler.std > 0).squeeze((1,2))
        series_c    = series_c[mask_flat]
        series_t    = series_t[mask_flat]

        if series_c.shape[0] == 0:
            continue

        # load the rest of the batch:
        modulations = batch.get('modulations').to(device)[mask_flat]         # [N, h]    (modulations)
        coords_c    = batch.get('context_coordinates').to(device)[mask_flat] # [N, T_in, 1] (context grid coordinates)
        coords_t    = batch.get('target_coordinates').to(device)[mask_flat]  # [N, T, 1] (target grid coordinates)
        mask        = batch.get('is_missing_mask').to(device)[mask_flat]     # [N, T, 1] (where the samples are not observed)

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

        # get reconstructed features on the target grid:
        interpo = outputs['yhat_t']

        # Compute normalize score 
        if series_t_exists:
            series_t                    = series_t.to(device)
            series_t                    = scaler.transform(series_t)
            error_normalize             = torch.abs(interpo - series_t)
            mae_on_missing_values_norm  = error_normalize[mask]
            mae_on_observed_values_norm = error_normalize[~mask]

        # undo z-normalization (back to data space):
        interpo = scaler.inv_transform(interpo)
        results['reco'].append(interpo)

        # compute errors on missing values (if ground truth is provided):
        if series_t_exists:
            series_t = scaler.inv_transform(series_t).to(device) # [N, T, 1]
            mask     = mask.to(device)                           # [N, T, 1]
            error    = torch.abs(interpo - series_t)
            
            mae_on_missing_values  = error[mask]
            mae_on_observed_values = error[~mask]

            # Denormalize error
            if len(mae_on_missing_values) > 0:
                results['mae_on_missing_values'].append(mae_on_missing_values)
            if len(mae_on_observed_values) >0:
                results['mae_on_observed_values'].append(mae_on_observed_values)

            # Normalize error 
            if len(mae_on_missing_values_norm) > 0:
                results['mae_on_missing_values_norm'].append(mae_on_missing_values_norm)
            if len(mae_on_observed_values_norm) >0:
                results['mae_on_observed_values_norm'].append(mae_on_observed_values_norm)      

            results['gt'].append(series_t)

        if plot_imputation:
            materials['coords_c'].append(coords_c.detach().cpu().squeeze())
            materials['series_c'].append(scaler.inv_transform(series_c.detach()).cpu().squeeze())
            materials['coords_t'].append(coords_t.detach().cpu().squeeze())
            materials['interpo'].append(interpo.detach().cpu().squeeze())
            materials['mask'].append(mask.cpu().squeeze())

            if series_t_exists:
                materials['series_t'].append(series_t.detach().cpu().squeeze())

    # agregate all results:
    results = {key:torch.cat(val,dim=0).cpu().detach() if len(val)>0 else val for key,val in results.items()}
    results = {key:val.mean().cpu().detach().item() if 'mae' in key and len(val)>0 else val for key,val in results.items()}

    save_dir_metrics = infer_path / 'metrics'
    save_dir_metrics.mkdir(exist_ok=True)

    # export metrics:
    pd.DataFrame({
        'Model': ['TimeFlow'],
        'MAE on Missing Values' : [results.get('mae_on_missing_values')],
        'MAE on Observed Values': [results.get('mae_on_observed_values')],
        'MAE on Missing Values (norm)' : [results.get('mae_on_missing_values_norm')],
        'MAE on Observed Values (norm)': [results.get('mae_on_observed_values_norm')]
    }).to_csv(save_dir_metrics / 'mae.csv')
    
    # concat batches outputs
    for key in materials:
        # stack batches together
        materials[key] = torch.cat(materials[key], dim=0) 

    if plot_imputation:
        make_imputation_plots(
            materials, 
            nb_plots, 
            infer_path, 
            series_t_exists,
            max_points_to_plot
        )

    if export_outputs:
        save_dir_materials = infer_path / 'materials'
        save_dir_materials.mkdir(exist_ok=True)
        torch.save(materials, save_dir_materials / 'materials.pt')

    return results

