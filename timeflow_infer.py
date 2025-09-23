import logging

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

from src.data.constants import nb_timesteps_per_day
from src.data.utils import PrepareImputationData
from src.data.dataloader import TimeSeriesImputationDataLoader
from src.tools.inference import infer_fn

@hydra.main(version_base=None, config_path= "config", config_name="inference")
def run(cfg: DictConfig):

    # get loggers and output directory:
    logger           = logging.getLogger(__name__)
    output_dir       = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info('[Config] Logs, output files, metrics, plots etc. will be located here: {}\n'.format(output_dir))

    # define device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get model ckpt path:
    model_path = Path( cfg.model_path ) if cfg.model_path is not None else Path(output_dir)

    # get path to checkpoint:
    pt_path = model_path / 'ckpt' / 'best.pt'

    # load saved checkpoint:
    if not pt_path.exists():
        logger.info('[Inference] Failed to find ckpt at {}, stop here\n'.format(pt_path))
        return
    else:
        inr_training_input = torch.load(pt_path, map_location=device)

    # retrieve actual INR and optim (for the inner loop) config:
    cfg_inr = OmegaConf.create(inr_training_input.get('cfg_inr'))

    # get saved cfg:
    saved_cfg = OmegaConf.load( Path(model_path) / ".hydra" / 'config.yaml' )

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
        seed             = saved_cfg.seed
    )
    
    _, data_te = series.extract_data()

    # build train dataloader:
    test_loader = TimeSeriesImputationDataLoader(
        X             = data_te['values'],
        grid          = data_te['coords'],
        latent_dim    = cfg_inr.latent_dim,
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

    inner_steps = saved_cfg.optim.inner_steps #
    inner_lr    = saved_cfg.optim.lr_code     # TODO allow overwrite from .sh / retrieve value from saved yaml
    loss_type   = saved_cfg.optim.loss_type

    # instantiate INR model:
    inr = instantiate( cfg_inr ).to(device)
    num_params = sum(p.numel() for p in inr.parameters())
    logger.info('[Model Prep] Number of parameters: {:,d}'.format(num_params))

    # load model checkpoint:
    inr.load_state_dict(inr_training_input.get('inr'))
    
    # 3/3 INFERENCE LOOP

    inr.eval()

    # prepare output path:
    imputation_setting     = Path( cfg.data.importation.test_file ).stem
    path_inference_setting = model_path / 'inference' / imputation_setting
    path_inference_setting.mkdir(parents=True, exist_ok=True)

    # export inference cfg:
    OmegaConf.save(cfg, path_inference_setting / 'config.yaml')

    # set some plot settings:
    max_points_to_plot = min(2016, nb_timesteps_per_day(sampling_freq) * cfg.task.window_len_in_days)

    # inference:
    logger.info('[Inference] Start inference loop...')
    
    infer_results_dict = infer_fn(
        inr                = inr,
        test_loader        = test_loader,
        inner_steps        = inner_steps,
        inner_lr           = inner_lr,
        path_results_exp   = path_inference_setting,
        loss_type          = loss_type,
        plot_imputation    = cfg.task.plot_imputation,
        nb_plots           = cfg.task.nb_plots,
        export_outputs     = cfg.task.export_outputs,
        max_points_to_plot = max_points_to_plot
    )

    mae_on_missing_values  = infer_results_dict.get('mae_on_missing_values')
    mae_on_observed_values = infer_results_dict.get('mae_on_observed_values')

    logger.info('[Inference] done, Test MAE loss on Missing Values: ' + ('{:.5f}'.format(mae_on_missing_values) if mae_on_missing_values is not None else 'None'))
    logger.info('[Inference] done, Test MAE loss on Observed Values: {:.5f}\n'.format(mae_on_observed_values))

    if (Path(output_dir) / 'timeflow_infer.log').exists():
        (Path(output_dir) / 'timeflow_infer.log').rename( model_path / 'inference' / 'timeflow_infer.log')
    
    logger.info('[LAST INFO] End of experiment\n')

    return

if __name__ == '__main__':
    run()

