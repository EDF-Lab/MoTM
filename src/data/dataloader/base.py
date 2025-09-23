import logging
from typing import Tuple, Dict, List, Any, Union, Optional
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.constants import nb_timesteps_per_day

def TimeSeriesImputationDataLoader(
    X: torch.Tensor, 
    grid: torch.Tensor,
    latent_dim: int,
    task_cfg: Optional[DictConfig] = None,
    ground_truths: Optional[torch.Tensor] = None,
    batch_size: int = 4, 
    num_workers: int = 4, 
    test_mode: bool = False,
    weight: float = 1.0,
    freq: str = '1H'
) -> DataLoader:
    """
    Creates a DataLoader for time series imputation.

    Args:
        X (torch.Tensor): Input tensor of shape [N, T, 1].
        grid (torch.Tensor): Grid tensor of shape [N, T, 1].
        latent_dim (int): Latent dimension size.
        task_cfg (DictConfig): Configuration dictionary.
        ground_truths (torch.Tensor, optional): Ground Truth tensor (no NaN) of shape [N, T, 1]
        batch_size (int, optional): Batch size. Defaults to 4.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
        test_mode (bool, optional): Whether to operate in test mode. Defaults to False.
        weight (float): weight for sampling the dataset
        freq (str): sampling freq (pandas datetime format)

    Returns:
        DataLoader: A DataLoader instance.
    """
    logger  = logging.getLogger(__name__)
    
    if test_mode:
        dataset = ImputationDatasetTest(X,grid, latent_dim=latent_dim, ground_truths=ground_truths)
    else:
        dataset = ImputationDatasetTrain(
            X,
            grid,
            latent_dim = latent_dim,
            task_cfg   = task_cfg,
            weight     = weight,
            freq       = freq
        )
    logger.info('[Data Prep] Prepared TS data, number of samples = {:,d}'.format(len(dataset)))

    return DataLoader(
        dataset, 
        batch_size  = batch_size, 
        shuffle     = not test_mode, 
        num_workers = num_workers,
        collate_fn  = lambda batch: collate_fn(batch, dataset, train=not test_mode),
        drop_last   = not test_mode
    )


class ImputationDataset(Dataset):
    """
    Base Class for time series imputation models.
    """
    def __init__(
        self,
        values: torch.Tensor,
        grids: torch.Tensor,
        latent_dim: int,
        ground_truths: Optional[torch.Tensor] = None,
        weight: float = 1.0
    ) -> None:
        """
        Initializes the training dataset.
        """
        self.raw_values = values # shape [N, T, 1] (may have NaNs)
        self.raw_grids  = grids  # shape [N, T, 1]
        self.modulations    = torch.zeros((values.shape[0], latent_dim))
        self.has_nans       = torch.isnan(self.raw_values).any().item()
        self.ground_truths  = ground_truths
        self.weight = weight

    def total_len(self) -> int:
        return len(self.raw_values)
    
    def __len__(self) -> int:
        return int( self.total_len() * self.weight )
    
    def complete_nans(self, is_test: bool = False, X: Optional[torch.Tensor] = None, grid: Optional[torch.Tensor] = None) -> None:
        """
        Fills missing values (NaNs) in the dataset (replaced by observed values at random instants of the same series).
        """

        if not self.has_nans:
            self.values_with_replacement = self.raw_values.clone()
            self.grids_with_replacement  = self.raw_grids.clone()
        else:
            
            if is_test:
                torch.manual_seed(42)
            else:
                if self.raw_grids.shape[1] < self.raw_values.shape[1]:
                    return
            
            # prepare tensors:
            mask_observed = ~torch.isnan(self.raw_values[..., 0]) # [N, T]
            X_filled      = self.raw_values[..., 0].clone() if X is None else X.clone()        # [N, T]
            partial_grid  = self.raw_grids[..., 0].clone() if grid is None else grid.clone()   # [N, T]

            # loop through samples:
            for i in range(X_filled.shape[0]):

                # time indices with no NaNs in raw data:
                valid_indices   = torch.where(mask_observed[i])[0]

                # time indices corresponding to NaNs in raw data:
                invalid_indices = torch.where(~mask_observed[i])[0]

                # replace NaNs by any other observed value:
                if valid_indices.numel() > 0 and invalid_indices.numel() > 0:  
                    replacements = valid_indices[torch.randint(0, valid_indices.numel(), (invalid_indices.numel(),))]
                    X_filled[i, invalid_indices]     = self.raw_values[i, replacements, 0]
                    partial_grid[i, invalid_indices] = self.raw_grids[i, replacements, 0]   
            
            self.values_with_replacement = X_filled.unsqueeze(-1)      # [N, T, 1] (with duplicates)
            self.grids_with_replacement  = partial_grid.unsqueeze(-1)  # [N, T, 1] (with duplicates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get boolean mask of missing values time indices:
        mask = ~torch.isin(self.raw_grids[idx, ...], self.grids_with_replacement[idx, ...])

        return {
            'context_features': self.values_with_replacement[idx, ...],   # has duplicates
            'target_features': self.ground_truths[idx, ...] if self.ground_truths is not None else None, # has no duplicates (but nans)
            'context_coordinates': self.grids_with_replacement[idx, ...], # has duplicates
            'target_coordinates': self.raw_grids[idx, ...],               # has no duplicates (but nans maybe)
            'modulations': self.modulations[idx, ...],
            'is_missing_mask': mask
        }


class ImputationDatasetTrain(ImputationDataset):
    """
    Dataset for training time series imputation models.
    """
    def __init__(
        self,
        values: torch.Tensor,
        grids: torch.Tensor,
        latent_dim: int,
        task_cfg: DictConfig,
        weight: float = 1.0,
        freq: str = '1H'
    ) -> None:
        """
        Initializes the training dataset.
        """
        super(ImputationDatasetTrain, self).__init__(values, grids, latent_dim)
        
        self.complete_nans(is_test=False)
        self.task_cfg = task_cfg
        self.grid_len = task_cfg.get('grid_length', self.raw_values.shape[1])
        assert self.grid_len == self.raw_grids.shape[1]
        self.one_day = nb_timesteps_per_day(freq)
        self.set_batch_params()
        self.weight  = weight

    def get_time_slice(self) -> Tuple[int,int]:
        """
        Random slicing of the series in time.
        """

        len_of_interest   = self.grid_len
        total_window_size = self.raw_values.shape[1]

        # sample endpoint (end of look-back + horizon window):
        end_point = torch.randint(len_of_interest, total_window_size, (1,)) if total_window_size > len_of_interest else total_window_size
        
        # get start point
        first_point = end_point - len_of_interest

        return first_point, end_point
    
    def set_batch_params(self) -> None:
        """
        Sets parameters for missing data patterns.
        """
        self.number_of_missing_block  = np.random.choice(self.task_cfg.get('number_of_missing_block', [4, 5, 6]))
        self.missing_block_size       = int( self.one_day * np.random.choice(self.task_cfg.get('missing_block_size', [4, 5, 9, 48, 96])) )
        self.pointwise_observed_ratio = np.random.choice(self.task_cfg.get('pointwise_observed_ratio', [0.85, 0.90, 0.95]))

    def remove_observations(self, sample_values: torch.Tensor, sample_grids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add missing values (by blocks and pointwise) to a single sample.
        """
        # (1/2) apply common missing block parameters (delete chunks of data):
        for _ in range(self.number_of_missing_block):
            start_missing_point = torch.randint(
                0, sample_grids.shape[0] - self.missing_block_size - 1, (1,)
            )
            end_missing_point = start_missing_point + self.missing_block_size

            # apply missing block to values:
            sample_values_left  = sample_values.clone()[:start_missing_point]
            sample_values_right = sample_values.clone()[end_missing_point:]
            sample_values       = torch.cat([sample_values_left, sample_values_right], dim=0)

            # apply missing block to coordinates:
            sample_grids_left  = sample_grids.clone()[:start_missing_point]
            sample_grids_right = sample_grids.clone()[end_missing_point:]
            sample_grids       = torch.cat([sample_grids_left, sample_grids_right], dim=0)

        # if we remove some blocks, we don't want to remove to much of the remaining context:
        if self.number_of_missing_block > 0:
            self.pointwise_observed_ratio = max(self.pointwise_observed_ratio, 0.9)

        # (2/2) apply pointwise missing values:
        self.n_points = int(self.pointwise_observed_ratio * sample_grids.shape[0])
        permutation   = torch.randperm(sample_grids.shape[0])[:self.n_points]
        sample_values = sample_values[permutation, ...] # [T', 1]
        sample_grids  = sample_grids[permutation, ...]  # [T', 1]

        return sample_values, sample_grids
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset.
        """

        # allow lens > total_len:
        idx = idx % self.total_len()

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # sample time slice:
        t1, t2 = self.get_time_slice()

        # fetch index sample:
        sample_values = self.values_with_replacement[idx, t1:t2, ...].clone() # [T, 1] (may have duplicates but no NaN)
        sample_grids  = self.grids_with_replacement[idx, ...].clone()  # [T, 1] (may have duplicates but no NaN)

        # add missing values to data for training:
        sample_values_partial, sample_grids_partial = self.remove_observations(sample_values, sample_grids)

        # build mask of missing instants:
        mask = ~torch.isin(self.raw_grids[idx, ...], sample_grids_partial)

        return {'context_features': sample_values_partial, 
                'target_features': sample_values,
                'context_coordinates': sample_grids_partial,
                'target_coordinates': sample_grids, 
                'modulations': self.modulations[idx, ...], 
                'is_missing_mask': mask}


class ImputationDatasetTestCovar(ImputationDataset):
    """
    Dataset for testing time series imputation models.
    """
    def __init__(
        self,
        values: torch.Tensor,
        grids: torch.Tensor,
        covariates: torch.Tensor,
        covar_grids: torch.Tensor,
        latent_dim: int,
        ground_truths: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initializes the testing dataset.
        """
        super(ImputationDatasetTestCovar, self).__init__(values, grids, latent_dim, ground_truths, weight=1.0)
        
        self.covariates  = covariates  # [N, T, C]
        self.covar_grids = covar_grids # [N, T, C]
        
        self.complete_nans( is_test = True )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get boolean mask of missing values time indices:
        mask = ~torch.isin(self.raw_grids[idx, ...], self.grids_with_replacement[idx, ...])

        return {
            'context_features': self.values_with_replacement[idx, ...],   # has duplicates
            'target_features': self.ground_truths[idx, ...] if self.ground_truths is not None else None, # has no duplicates (but nans)
            'context_coordinates': self.grids_with_replacement[idx, ...], # has duplicates
            'target_coordinates': self.raw_grids[idx, ...],               # has no duplicates (but nans maybe)
            'modulations': self.modulations[idx, ...],
            'is_missing_mask': mask,
            'covariates_coordinates': self.covar_grids[idx, ...],
            'covariates': self.covariates[idx, ...]
        }


class ImputationDatasetTest(ImputationDataset):
    """
    Dataset for testing time series imputation models.
    """
    def __init__(
        self,
        values: torch.Tensor,
        grids: torch.Tensor,
        latent_dim: int,
        ground_truths: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initializes the testing dataset.
        """
        super(ImputationDatasetTest, self).__init__(values, grids, latent_dim, ground_truths, weight=1.0)
        self.complete_nans( is_test = True )


def collate_fn(
    batch: List[Dict[str, Any]],
    dataset: Union[ImputationDatasetTrain,ImputationDatasetTest],
    train: bool = True
) -> Dict[str, Any]:
    """
    Custom collate function for batching time series imputation data.

    Args:
        batch (List[Dict[str, Any]]): List of dictionary samples from the dataset.
        dataset (Type[Dataset]): The dataset class used for data loading.
        train (bool, optional): Whether the batch is for training. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing batched tensors and other batch data.
    """
    
    if train:
        dataset.complete_nans()
        dataset.set_batch_params()

    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    batch_dict = {key: torch.stack(value, dim=0) if isinstance(value[0], torch.Tensor) else value
                  for key, value in batch_dict.items()}

    return batch_dict
