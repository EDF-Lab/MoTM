import logging

from pathlib import Path
from time import time

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

import torch

from src.data.constants import nb_timesteps_per_day, get_latent_norm
from src.data.utils import PrepareImputationData
from src.data.dataloader import TimeSeriesImputationDataLoader
from src.modules.inr import ModulatedFourierFeatures, ModulatedWaveletINR, WaveletINR
from src.tools.trainer import train_fn

@hydra.main(version_base=None, config_path="config", config_name="imputation")
def run(cfg : DictConfig):

    # get logger:
    logger = logging.getLogger(__name__)

    # get ouput directory created by hydra:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    path_results_exp = Path( output_dir )
    logger.info('[Config] Logs, output files, metrics, plots etc. will be located here: {}\n'.format(output_dir))

    # 1/3 PREPARE DATA (LOAD, PREPROCESS, BUILD DATALOADERS)

    path_train = Path(cfg.data.importation.dir) / cfg.data.importation.train_file

    window_len_in_days = cfg.task.get('window_len_in_days', 28)
    sampling_freq      = cfg.data.freq
    train_grid_length  = int( nb_timesteps_per_day(sampling_freq) * window_len_in_days )
    OmegaConf.update(cfg.task,'grid_length', train_grid_length, force_add=True)

    # load data:
    series = PrepareImputationData(
        path_train       = path_train,
        is_train         = True,
        train_test_split = cfg.data.train_test.sample.get('split', False),
        grid_length      = train_grid_length,
        seed             = cfg.seed,
        **{
            'test_at_random' : cfg.data.train_test.sample.get('random_split', True),
            'test_size'      : cfg.data.train_test.sample.get('test_size', 0.1),
            'test_split_idx' : cfg.data.train_test.sample.get('split_idx')
        }
    )
    data_tr, _ = series.extract_data()

    # if provided data contain NaN, we force not adding missing values during training:
    if torch.isnan( data_tr['values'] ).any().item():
        OmegaConf.update(cfg.task,'number_of_missing_block', [0])
        OmegaConf.update(cfg.task,'missing_block_size', [0])
        OmegaConf.update(cfg.task,'pointwise_observed_ratio', [1.])
        logger.info('[Data Prep] Training data has NaNs, they are kept as such for training')
    else:
        logger.info('[Data Prep] Training data does not have NaNs, will add some (blocks and pointwise) during training -- see `config.task`')
    
    # build train dataloader:
    train_loader = TimeSeriesImputationDataLoader(
        X           = data_tr['values'],
        grid        = data_tr['coords'],
        latent_dim  = cfg.inr.latent_dim,
        task_cfg    = cfg.task,
        batch_size  = cfg.data.batch_size,
        num_workers = cfg.data.num_workers,
        test_mode   = False,
        weight      = cfg.data.get('weight', 1.0),
        freq        = cfg.data.get('freq')
    )
    
    for batch in train_loader:
        logger.info('[Data Prep] Train dataloader of len {:d}'.format(len(train_loader)))
        for key, val in batch.items():
            logger.info('[Data Prep] Batch key `{}` of shape {}'.format(key, val.shape))
        break

    logger.info('[Data Prep] End of Data Preparation\n')
    
    # 2/3 PREPARE MODEL

    normalize_z  = cfg.inr.apply_znorm
    target_znorm = get_latent_norm( cfg.optim.lr_code, cfg.optim.inner_steps ) if normalize_z else None
    OmegaConf.update(cfg.inr, 'target_znorm', target_znorm)

    inr = instantiate(cfg.inr)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.trainer.accelerator=='gpu' else 'cpu')
    inr    = inr.to(device)

    num_params = sum(p.numel() for p in inr.parameters())
    logger.info('[Model Prep] Number of parameters: {:,d}'.format(num_params))
    logger.info('[Model Prep] Model architecture:')
    logger.info(inr)
    logger.info('[Model Prep] End of Model Preparation\n')

    # 3/3 TRAIN LOOP

    # train:
    logger.info('[Training] Start training...')

    t0 = time()
    mae_loss = train_fn(
        inr              = inr,
        train_loader     = train_loader,
        cfg              = cfg,
        path_results_exp = path_results_exp
    )
    t1 = time()
    logger.info('[Training] Training completed, {:,d} epochs in {:.2f}min'.format(cfg.trainer.max_epochs,(t1-t0)/60))
    logger.info('[Training] Final MAE loss: {:.5f}\n'.format(mae_loss))

    logger.info('[LAST INFO] End of experiment\n')

    return mae_loss

if __name__ == '__main__':
    run()
