from typing import List, Dict, Optional 
from pathlib import Path

import torch 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def plot_losses(
    l_epochs: List[int],
    l_mse_loss: List[float],
    l_mae_loss: List[float],
    filename: Optional[str] = 'training_loss',
    save_path: Optional[Path] = Path('./')
) -> None:

    fig, ax = plt.subplots(1,1,figsize=(12,3)) # (20,5)
    ax.yaxis.grid(True)
    ax.semilogy(l_epochs, l_mse_loss, label='MSE')
    ax.semilogy(l_epochs, l_mae_loss, label='MAE')
    ax.legend()
    fig.tight_layout()
    fig.savefig( save_path / '{}.png'.format(filename) )
    plt.close(fig)


def plot_loss_steps(
    step_loss: List[float],
    filename: Optional[str] = 'training_loss_step',
    save_path: Optional[Path] = Path('./')
) -> None:
    
    fig,ax = plt.subplots(1,1,figsize=(12,3),sharey=True)
    ax.yaxis.grid(True)

    # step_loss = np.asarray(step_loss)

    ax.loglog(step_loss, label='values',color='tab:gray', lw=0.75)
    try:
        M = 16
        running_average = np.convolve(step_loss, np.ones(M)/M, mode='valid')        
        ax.fill_between(np.arange(len(step_loss))[M-1:], 0, running_average, color='tab:blue', alpha=0.1)
        ax.loglog(np.arange(len(step_loss))[M-1:], running_average, color='tab:blue', lw=1,alpha=0.85)
    except:
        pass
    ax.set_xlabel('Steps')
    ax.legend()
    fig.tight_layout()
    fig.savefig( save_path / '{}.png'.format(filename) )
    plt.close(fig)


def make_imputation_plots(
    materials: Dict[str, torch.Tensor],
    nb_plots: int,
    save_dir: Path,
    gt_exists: bool,
    max_points: int = -1
):

    """
    Generates and saves time series imputation plots.

    Args:
        materials (Dict[str, torch.Tensor]): Data for plotting:
            - 'coords_c', 'series_c': Context points (known values).
            - 'coords_t', 'series_t': Full time series (including ground truth).
            - 'interpo': Imputed values by TimeFlow.
            - 'mask': Boolean mask (True for imputed values).
        nb_plots (int): Number of samples to plot.
        save_dir (Path): Directory for saving plots.
        gt_exists (bool): Whether ground truth is available.
        max_points (int): how many time points to plot (will plot all if <0)

    Output:
        Saves `nb_plots` as PNGs in `save_dir/inference_plots/`.
    """    

    save_dir_plots = save_dir / 'inference_plots'
    save_dir_plots.mkdir(exist_ok=True)

    samples_to_plot = torch.randint(materials['interpo'].shape[0], (nb_plots,))
    max_points      = min(max_points, materials['interpo'].shape[1]) if max_points > 0 else materials['interpo'].shape[1]

    # set matplotlib params:
    matplotlib.rc('font', **{'size':18})
    matplotlib.rc('lines', **{'linewidth':2.5, 'linestyle': '-', 'markersize': 5})

    for s in samples_to_plot:
        
        # get target coordinates (ordered from 0 to 1 with no repeat):
        coords_t    = np.array(materials['coords_t'][s])

        # limit target to max_points samples for clarity:
        time_mask_t    = coords_t < coords_t[max_points-1]
        coords_t       = coords_t[time_mask_t]
        imputed_values = np.array(materials['interpo'][s])[time_mask_t]

        # get target mask (True if value is missing, False if observed and part of context):
        mask = np.array(materials['mask'][s])[time_mask_t]

        # init figure:
        plt.figure(figsize=(20,5))

        # plot Ground Truth if accessible:
        if gt_exists:
            plt.plot(
                coords_t,
                materials['series_t'][s][time_mask_t],
                color="tab:green",
                lw=3,
                label='Ground Truth'
            )

        # plot context points (context = not is missing in target):
        coords_context = coords_t[~mask]
        values_context = materials['series_t'][s][time_mask_t][~mask]

        plt.plot(
            coords_context,
            values_context,
            'o',
            markersize=2,
            color="tab:red",
            label='Context'
        )

        # make TimeFlow interpo plot:
        coords_imputed = coords_t[mask]
        values_imputed = imputed_values[mask]

        if len(coords_imputed) > 0:
            # plot TimeFlow at all timesteps (as a line):
            plt.plot(
                coords_t,
                imputed_values,
                color="tab:blue",
                label='TimeFlow'
            )

            # scatter plot TimeFlow at non-context:
            plt.scatter(
                coords_imputed,
                values_imputed,
                marker='o',
                color="tab:blue"
            )

        plt.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.075),
            fancybox=True, 
            shadow=True, 
            ncol=3
        )
        # plt.axhline(0,color='tab:gray', lw=1)
    
        plt.tight_layout()
        plt.savefig(
            f'{save_dir_plots}/sample_{s}.png',
            dpi=100,
            bbox_inches='tight'
        )



def make_imputation_mixture_plots(
    materials: Dict[str, torch.Tensor],
    nb_plots: int,
    save_dir: Path,
    gt_exists: bool,
    max_points: int = -1,
    model_name: str = 'Ridge',
    is_blockwise: bool = False
):

    """
    Generates and saves time series imputation plots.

    Args:
        materials (Dict[str, torch.Tensor]): Data for plotting:
            - 'coords_c', 'series_c': Context points (known values).
            - 'coords_t', 'series_t': Full time series (including ground truth).
            - 'interpo': Imputed values by TimeFlow.
            - 'mask': Boolean mask (True for imputed values).
        nb_plots (int): Number of samples to plot.
        save_dir (Path): Directory for saving plots.
        gt_exists (bool): Whether ground truth is available.
        max_points (int): how many time points to plot (will plot all if <0)

    Output:
        Saves `nb_plots` as PNGs in `save_dir/inference_plots/`.
    """    

    save_dir_plots = save_dir / 'inference_plots_{}'.format(model_name.lower())
    save_dir_plots.mkdir(exist_ok=True)

    samples_to_plot = torch.randint(materials['interpo_{}_norm'.format(model_name)].shape[0], (nb_plots,))
    max_points      = min(max_points, materials['interpo_{}_norm'.format(model_name)].shape[1]) if max_points > 0 else materials['interpo_{}_norm'.format(model_name)].shape[1]

    # set matplotlib params:
    matplotlib.rc('font', **{'size':18})
    matplotlib.rc('lines', **{'linewidth':1.5, 'linestyle': '-', 'markersize': 4})

    for s in samples_to_plot:
        
        # get target coordinates (ordered from 0 to 1 with no repeat):
        coords_t    = np.array(materials['coords_t'][s])

        # limit target to max_points samples for clarity:
        time_mask_t    = coords_t < coords_t[max_points-1]
        coords_t       = coords_t[time_mask_t]
        imputed_values = np.array(materials['interpo_{}_norm'.format(model_name)][s])[time_mask_t]

        # get target mask (True if value is missing, False if observed and part of context):
        mask = np.array(materials['mask'][s])[time_mask_t]
        
        # get start and end indices of NA blocks:
        block_indices = np.where(np.diff(mask))[0] if is_blockwise else []

        # init figure:
        plt.figure(figsize=(12,4))

        # plot Ground Truth if accessible:
        if gt_exists:
            plt.plot(
                coords_t,
                materials['series_t_norm'][s][time_mask_t],
                color="tab:green",
                lw=1.75,
                label='Ground Truth'
            )

        # plot context points (context = not is missing in target):
        coords_context = coords_t[~mask]
        values_context = materials['series_t_norm'][s][time_mask_t][~mask]

        plt.plot(
            coords_context,
            values_context,
            'o',
            markersize=2.5,
            color="tab:red",
            label='Context'
        )

        # make TimeFlow interpo plot:
        coords_imputed = coords_t[mask]
        values_imputed = imputed_values[mask]

        if len(coords_imputed) > 0:
            for i,idx in enumerate(block_indices[::2]):
                plt.axvline(coords_t[idx],ls='--',color='k',lw=.5)
                end_point = coords_t[block_indices[1::2][i]] if i < len(block_indices[1::2]) else coords_t[-1]
                plt.axvline(end_point,ls='--',color='k',lw=0.5)
                plt.axvspan(coords_t[idx], end_point, facecolor='tab:gray',alpha=0.1)


            # plot TimeFlow at all timesteps (as a line):
            plt.plot(
                coords_t,
                imputed_values,
                lw=1.75,
                color="tab:blue",
                label='MoTM'
            )

            # plt.plot(
            #     coords_imputed,
            #     values_imputed,
            #     color="tab:blue",
            #     label='MoTM'
            # )


            # Initialiser une série avec des NaN
            # values_plot = np.full_like(coords_t, np.nan, dtype=np.float32)

            # # Insérer les valeurs imputées aux bons indices
            # for coord, value in zip(coords_imputed, values_imputed):
            #     idx = np.where(coords_t == coord)[0]
            #     if len(idx) > 0:
            #         values_plot[idx[0]] = value

            # # Tracer sans relier les points non consécutifs
            # plt.plot(
            #     coords_t,
            #     values_plot,
            #     color="tab:blue",
            #     label='MoTM'
            # )


            # scatter plot TimeFlow at non-context:
            # plt.scatter(
            #     coords_imputed,
            #     values_imputed,
            #     marker='o',
            #     color="tab:blue",
            #     s=6
            # )

        if 'covariates' in materials:
            plt.plot(
                materials['coords_cov'][s],
                materials['covariates'][s],
                lw=1.25,
                color="tab:gray",
                alpha=0.5,
                label='Covariate'
        )

        plt.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.075),
            fancybox=True, 
            shadow=True, 
            ncol=3 + int( 'covariates' in materials )
        )
        plt.axhline(0,color='tab:gray', lw=1)
    
        plt.tight_layout()
        plt.savefig(
            f'{save_dir_plots}/sample_{s}.png',
            dpi=300,
            bbox_inches='tight'
        )