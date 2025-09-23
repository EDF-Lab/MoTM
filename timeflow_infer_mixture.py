import logging

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

from src.data.constants import get_latent_norm, nb_timesteps_per_day
from src.data.utils import PrepareImputationData
from src.data.dataloader import TimeSeriesImputationDataLoader
from src.tools.inference_mixture import infer_fn_mixture

@hydra.main(version_base=None, config_path= "config", config_name="inference")
def run(cfg: DictConfig):

    # get loggers and output directory:
    logger           = logging.getLogger(__name__)
    output_dir       = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info('[Config] Logs, output files, metrics, plots etc. will be located here: {}\n'.format(output_dir))

    # define device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    list_model_path = cfg.list_model_path

    # get model ckpt path:
    model_path = [Path(val) for val in list_model_path]

    # get path to checkpoint:
    # pt_path = [val / 'ckpt' / 'best_fit.pt' if (val / 'ckpt' / 'best_fit.pt').exists() else val / 'ckpt' / 'best.pt' for val in model_path]
    pt_path = [val / 'ckpt' / 'best.pt' for val in model_path]

    # load saved checkpoint:
    if not all([val.exists() for val in pt_path]):
        logger.info('[Inference] Failed to find ckpt, stop here\n')
        return
    else:
        list_inr_training_input = [torch.load(val, map_location=device) for val in pt_path]
        logger.info('[Inference] Mixture of {:d} base models'.format(len(list_model_path)))
    
    # retrieve actual INR and optim (for the inner loop) config:
    list_cfg_inr = [OmegaConf.create(val.get('cfg_inr')) for val in list_inr_training_input]

    # get saved cfg:
    list_saved_cfg = list( [OmegaConf.load( Path(val) / ".hydra" / 'config.yaml' ) for val in model_path] )

    # 1/3 PREPARE DATA (LOAD, PREPROCESS, BUILD DATALOADERS)
    
    path_train = Path(cfg.data.importation.dir) / cfg.data.importation.train_file

    test_file = cfg.data.importation.test_file
    path_test = Path(cfg.data.importation.dir) / test_file if (test_file is not None) else test_file

    gt_file = cfg.data.importation.test_gt_file
    path_gt = Path(cfg.data.importation.dir) / gt_file if (gt_file is not None) else gt_file

    sampling_freq = cfg.data.freq

    series = PrepareImputationData(
        path_train       = path_train,
        path_test        = path_test,
        path_gt_test     = path_gt,
        is_train         = False,
        train_test_split = False,
        seed             = list_saved_cfg[0].seed,
    )
    
    _, data_te = series.extract_data()

    # build train dataloader:
    test_loader = TimeSeriesImputationDataLoader(
        X             = data_te['values'],
        grid          = data_te['coords'],
        latent_dim    = list_cfg_inr[0].latent_dim, # XXX
        ground_truths = data_te['gt'],
        batch_size    = cfg.data.batch_size,
        num_workers   = cfg.data.num_workers,
        test_mode     = True
    )
    
    for batch in test_loader:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                logger.info('[Data Prep] Batch key `{}` of shape {}'.format(key, val.shape))
            else:
                logger.info('[Data Prep] Batch key `{}` of type {}'.format(key,type(val)))
        break

    logger.info('[Data Prep] End of Data Preparation\n')

    # 2/3 PREPARE AND LOAD MODEL

    list_inrs, list_inner_steps, list_inner_lr, list_loss_types = [], [], [], []
    for saved_cfg, cfg_inr, inr_training_input in zip(list_saved_cfg, list_cfg_inr, list_inr_training_input):

        inner_steps = saved_cfg.optim.inner_steps #
        inner_lr    = saved_cfg.optim.lr_code     # TODO allow overwrite from .sh / retrieve value from saved yaml
        loss_type   = saved_cfg.optim.loss_type

        list_inner_steps.append(inner_steps)
        list_inner_lr.append(inner_lr)
        list_loss_types.append(loss_type)

        # instantiate INRs model:
        inr = instantiate( cfg_inr ).to(device)
        num_params = sum(p.numel() for p in inr.parameters())
        logger.info('[Model Prep] Number of parameters: {:,d}'.format(num_params))

        # load model checkpoint:
        inr.load_state_dict(inr_training_input.get('inr'))

        inr.eval()
        list_inrs.append(inr)

    # 3/3 INFERENCE LOOP

    path_results_dir = output_dir

    path_results_exp_mixture         = Path(path_results_dir) / cfg.data.name # (f'{path_results_dir}/{cfg.data.name}')
    path_results_exp_mixture_setting = path_results_exp_mixture / Path( cfg.data.importation.test_file ).stem
    path_results_exp_mixture_setting.mkdir(parents=True, exist_ok=True)

    # inference:
    dict_params = {
        'inner_steps'  : list_inner_steps,
        'inner_lr'     : list_inner_lr,
        'loss_type'    : list_loss_types
    }

    lambda_ridge       = cfg.get('lambda_ridge', 0.1)
    inr_last_layers    = cfg.get('inr_last_layers', 1)
    if inr_last_layers > cfg_inr.depth:
        logger.info('[Model Prep] *Warning* INR depth is {} but required {} feature layers, will threshold at {}'.format(cfg_inr.depth,inr_last_layers,cfg_inr.depth))
        inr_last_layers = cfg_inr.depth
    share_ridge_weights = cfg.get('share_ridge_weights', False)
    max_points_to_plot = min(2016, nb_timesteps_per_day(sampling_freq) * cfg.task.window_len_in_days)

    logger.info('[Inference] Start inference loop...')
    _ = infer_fn_mixture(
        list_inrs,
        test_loader         = test_loader,
        dict_params         = dict_params,
        path_results_exp    = path_results_exp_mixture_setting,
        inr_last_layers     = inr_last_layers,
        lambda_ridge        = lambda_ridge,
        share_ridge_weights = share_ridge_weights,
        learn_lambda        = cfg.get('learn_lambda', False),
        baseline_offset     = nb_timesteps_per_day(sampling_freq),
        plot_imputation     = cfg.task.plot_imputation,
        nb_plots            = cfg.task.nb_plots,
        export_outputs      = cfg.task.export_outputs,
        max_points_to_plot  = max_points_to_plot,
        is_block_setting    = 'block' in str( cfg.data.importation.test_file ),
        run_baselines       = cfg.task.get('run_baselines', True)
    )

    if (Path(output_dir) / 'timeflow_infer.log').exists():
        (Path(output_dir) / 'timeflow_infer.log').rename(model_path[0] / 'timeflow_mixture_infer.log')
    
    logger.info('[LAST INFO] End of experiment\n')

    return

if __name__ == '__main__':
    run()

