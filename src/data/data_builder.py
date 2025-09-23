# Some imports 
from typing import Dict, Optional, List
from pathlib import Path

import numpy as np 
import pandas as pd 

from einops import rearrange

import torch 
from torch import Tensor


class Extract_data:
    """
    A utility class for loading and preparing time series data for training and testing, 
    including simulation of missing values through masking.

    Attributes
    ----------
    data_tensor : torch.Tensor
        The full dataset as a tensor of shape (n_samples, n_timestamps).
    chunk_size : int
        The number of timestamps per chunk used for training and testing data generation.
    train_data : torch.Tensor
        Tensor containing the concatenated training chunks.
    test_data : dict[str, torch.Tensor]
        Dictionary of test datasets under different missing data scenarios.
    val_data : dict[str, torch.Tensor]
        Dictionary of validation datasets under different missing data scenarios.
    """

    def __init__(
        self,
        data_path: str,
        train_ratio: float,
        test_ratio: float,
        nb_channels: int = 1,
        nsamples: int = -1
    ):
        """
        Loads data from a CSV/parquet file and stores it as a PyTorch tensor.

        Parameters
        ----------
        data_path : str
            Path to the CSV or parquet file. Assumes the file has a 'date' column to drop (csv).
        train_ratio : float
            Ratio of the dataset to be used for training (between 0 and 1).
        test_ratio : float
            Ratio of the dataset to be used for testing (between 0 and 1).
        nb_channels : float
            If > 1, reshape output tensor from `(N, T)` to `(N, nb_channels, T')`.
        """

        data_path   = Path(data_path)
        file_suffix = data_path.suffix[1:]

        # load csv file:
        if file_suffix == 'csv':
            # data = pd.read_csv(data_path, index_col=-1)
            data = pd.read_csv(data_path, index_col=0)
            print(data.head(3))

            col_with_date = data.columns[data.columns.str.contains('date', case=False, regex=False)]
            print(col_with_date)

            # csv prepared by ELN:
            if len(col_with_date) == 1:
                col_date_name = col_with_date[0]
                data_array = data.drop(col_date_name, axis=1).values.transpose()

            # handle synthetic data with no date:
            else:
                data_array = data.values
                if data_array.shape[1] < len(data_array):
                    data_array = data_array.transpose()

        # load gifteval data:
        elif file_suffix == 'parquet':
            
            data = pd.read_parquet(data_path, engine='pyarrow')

            # values are stored in `target`:            
            data_array      = data["target"].values
            is_multivariate = isinstance(data_array[0][0], np.ndarray)
            nvars           = len(data_array[0]) if is_multivariate else 1
            if nvars > 1 :
                data_array = np.concatenate(
                    [
                        data_array[idx][n][:].reshape(1,-1) for idx in range(len(data_array)) for n in range(nvars)
                    ], 
                    axis = 0
                )
            else:

                # some datasets have time series of variable lengths, truncate to shortest len:
                min_len = np.min([len(data_array[idx]) for idx in range(len(data_array))])
                max_len = np.max([len(data_array[idx]) for idx in range(len(data_array))])
                print('Samples min / max len: {} to {}'.format(min_len, max_len))
                print('Irregular len:', min_len < max_len)
                data_array = np.concatenate([data_array[idx][:min_len].reshape((1,-1)) for idx in range(len(data_array))], axis=0)

            self.is_multivariate = is_multivariate

        else:
            raise NotImplementedError('Unknown extension {}, valid extensions are `csv` and `parquet`.'.format(file_suffix) )

        # convert to torch Tensor:
        print('Total number of timesteps: {:,d}'.format(data_array.shape[1]))
        print('Total samples: {:,d}'.format(data_array.shape[0]))
        print(data_array[:2])
        self.data_tensor: Tensor = torch.tensor(data_array, dtype=torch.float32)

        # rearrange to create a channel dim (if multivariate dataset wanted):
        if nb_channels > 1:
            self.data_tensor = rearrange(self.data_tensor, '(m c) T -> m c T', c=nb_channels)
        
        # downsample if needed:
        if nsamples > 0:
            torch.manual_seed(24092025)
            permutation = torch.randperm(len(self.data_tensor))
            self.data_tensor = self.data_tensor[permutation[:min(nsamples, len(self.data_tensor))]]
            print('Total samples after downsampling: {:,d}'.format(self.data_tensor.shape[0]))

        # save end of train (t1) and val (t2) time intervals:
        self.t1 = int( self.data_tensor.shape[-1] * train_ratio )
        self.t2 = int( self.data_tensor.shape[-1] * (1.0 - test_ratio) )

        assert self.t1 <= self.t2

    def make_chunks(
        self,
        nb_timestamps_per_day: int,
        t_start: int,
        t_end: int,
        rm_nan_chunks: bool = False,
        flatness_threshold: float = 1.,
        seed: int = 42
    ) -> torch.Tensor:
        """
        Split long time series into chunks of fixed size with random window sliding.

        Args:
            nb_timestamps_per_day (int): number of time steps in a single day
            t_start (int): starting time index of the time series to chunk
            t_end (int): ending time index of the time series to chunk
            seed (int): random seed
        
        Returns:
            A torch.Tensor of size (*, chunk_size).
        """
        
        l_data    = []
        t_index   = t_start

        torch.manual_seed(seed)
    
        # print('chunks start', t_index, self.chunk_size, t_end)
        while (t_index + self.chunk_size) <= t_end:

            new_chunk = self.data_tensor[..., t_index:t_index + self.chunk_size]

            if rm_nan_chunks:
                nanmask   = (new_chunk.isnan().sum(-1) == 0)
                new_chunk = new_chunk[nanmask]

            nanmean   = torch.nanmean(new_chunk, dim=-1, keepdims=True)
            chunk_std = (new_chunk - nanmean).square().nanmean(dim=-1, keepdim=True).sqrt()
            new_chunk = new_chunk[(chunk_std>0).squeeze(-1)]

            flat_mask = (torch.diff(new_chunk) == 0).sum(-1) / (new_chunk.shape[-1] - 1)
            new_chunk = new_chunk[flat_mask <= flatness_threshold]
            if len(new_chunk) > 0:
                l_data.append( new_chunk )
            random_increment = torch.randint(
                low=int(nb_timestamps_per_day / 2),
                high=int(nb_timestamps_per_day * 2),
                size=(1,)
            )
            t_index += random_increment.item()
        x: Tensor = torch.cat(l_data, dim=0) if len(l_data) > 1 else l_data[0]
        return x
    
    def make_train_data(
        self, 
        nb_timestamps_per_day: int, 
        nb_days_per_chunk: int, 
        make_chunks: bool = True,
        flatness_threshold: float = 1.,
        seed: int = 42
    ) -> None:
        """
        Generates training data chunks from the dataset using sliding windows with random offsets.

        Parameters
        ----------
        nb_timestamps_per_day : int
            Number of time steps in a single day.
        nb_days_per_chunk : int
            Number of days to include in one data chunk.
        train_ratio : float
            Ratio of the dataset to be used for training (between 0 and 1).
        make_chunks : bool
            If False, do not use sliding windows and keep the raw series instead.
        """
        
        nb_timestamps_train = self.t1
        self.chunk_size     = nb_timestamps_per_day * nb_days_per_chunk

        if not make_chunks:
            self.train_data: Tensor = self.data_tensor[..., :self.t1]
            print('train data has nans? ', torch.isnan(self.train_data).sum())
            return
        
        x = self.make_chunks(
            nb_timestamps_per_day,
            t_start = 0,
            t_end   = nb_timestamps_train,
            flatness_threshold = flatness_threshold,
            rm_nan_chunks      = True,
            seed    = seed
        )

        self.train_data: Tensor = x

    def make_test_data(
        self, 
        nb_timestamps_per_day: int, 
        nb_days_per_chunk: int, 
        flatness_threshold: float = 1.,
        seed: int = 43
    ) -> None:
        """
        Generates testing data chunks and simulates different missing data scenarios.

        Parameters
        ----------
        nb_timestamps_per_day : int
            Number of time steps in a single day.
        nb_days_per_chunk : int
            Number of days to include in one data chunk.
        """

        nb_timestamps_test   = self.data_tensor.shape[-1]
        self.chunk_size      = nb_timestamps_per_day * nb_days_per_chunk
        test_days            = (nb_timestamps_test - self.t2) // nb_timestamps_per_day
        has_enough_test_days = test_days >= nb_days_per_chunk
        
        if not has_enough_test_days:
            raise RuntimeError('Data with less than {} weeks per sample, cannot chunk'.format(nb_days_per_chunk))

        self.test_data = dict()

        test_data = self.make_chunks(
            nb_timestamps_per_day,
            t_start = self.t2,
            t_end   = nb_timestamps_test,
            flatness_threshold = flatness_threshold,
            rm_nan_chunks      = True,
            seed    = seed
        )

        self.test_data: Dict[str, Tensor] = {
            "scenario_blocks_missing_1":    self.mask_time_series(test_data, 0.0, 2, nb_timestamps_per_day),
            "scenario_blocks_missing_2":    self.mask_time_series(test_data, 0.0, 4, nb_timestamps_per_day),
            "scenario_pointwise_missing_1": self.mask_time_series(test_data, 0.50, 0, 0),
            "scenario_pointwise_missing_2": self.mask_time_series(test_data, 0.70, 0, 0),
            "scenario_blocks_forecast_1":   self.mask_forecasting_target(test_data, 1, nb_timestamps_per_day),
            "scenario_blocks_forecast_2":   self.mask_forecasting_target(test_data, 4, nb_timestamps_per_day),
            "scenario_blocks_forecast_3":   self.mask_forecasting_target(test_data, 7, nb_timestamps_per_day),
            "ground_truth":                 test_data
        }
        
        return has_enough_test_days
    
    def make_val_data(
        self, 
        nb_timestamps_per_day: int, 
        nb_days_per_chunk: int,
        flatness_threshold: float = -1.,
        rm_nan_chunks      = True,
        seed: int = 44
    ) -> None:
        """
        Generates testing data chunks and simulates different missing data scenarios.

        Parameters
        ----------
        nb_timestamps_per_day : int
            Number of time steps in a single day.
        nb_days_per_chunk : int
            Number of days to include in one data chunk.
        """

        assert self.t1 < self.t2
        

        self.chunk_size = nb_timestamps_per_day * nb_days_per_chunk
        print(self.t1, self.t2, self.chunk_size)

        val_data = self.make_chunks(
            nb_timestamps_per_day,
            t_start = self.t1,
            t_end   = self.t2,
            flatness_threshold=flatness_threshold,
            seed    = seed
        )

        self.val_data: Dict[str, Tensor] = {
            "scenario_blocks_missing_1":    self.mask_time_series(val_data, 0.0, 2, nb_timestamps_per_day),
            "scenario_blocks_missing_2":    self.mask_time_series(val_data, 0.0, 4, nb_timestamps_per_day),
            "scenario_pointwise_missing_1": self.mask_time_series(val_data, 0.50, 0, 0),
            "scenario_pointwise_missing_2": self.mask_time_series(val_data, 0.70, 0, 0),
            "scenario_blocks_forecast_1":   self.mask_forecasting_target(val_data, 1, nb_timestamps_per_day),
            "scenario_blocks_forecast_2":   self.mask_forecasting_target(val_data, 4, nb_timestamps_per_day),
            "scenario_blocks_forecast_3":   self.mask_forecasting_target(val_data, 7, nb_timestamps_per_day),
            "ground_truth":                 val_data
        }

    def mask_time_series(
        self, 
        x: Tensor, 
        missing_pointwise_ratio: float = 0.0, 
        num_blocks: int = 2, 
        block_size: int = 96
    ) -> Tensor:
        """
        Masks parts of a time series tensor with NaNs either pointwise or in contiguous blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_chunks, chunk_length).
        missing_pointwise_ratio : float, optional
            Ratio of time points to randomly mask (default is 0.0).
        num_blocks : int, optional
            Number of contiguous blocks to mask (default is 2).
        block_size : int, optional
            Size of each block to mask (default is 96).

        Returns
        -------
        torch.Tensor
            Masked version of the input tensor with NaNs in place of missing values.
        """

        N    = x.shape[0]
        T    = x.shape[-1]
        mask = torch.ones( (N, T), dtype = torch.bool )

        num_points        = int(missing_pointwise_ratio * T)
        available_indices = np.arange(T)

        for i in range(N):

            if num_points > 0:
                point_indices = np.random.choice(available_indices, min(num_points, T), replace=False)
                mask[i, point_indices] = False

            for _ in range(num_blocks):
                is_not_working = True
                while is_not_working:
                    start = np.random.randint(0, T - block_size + 1)
                    if mask[i, start:start + block_size].all():
                        mask[i, start:start + block_size] = False
                        is_not_working = False

        x_masked = x.clone()

        if x.ndim == 2:
            x_masked[~mask] = float('nan')

        elif x.ndim ==3:
            for c in range(x.shape[1]):
                x_c = x_masked[:,c,:]
                x_c[~mask] = float('nan')
                x_masked[:,c,:] = x_c

        return x_masked
    
    def mask_forecasting_target(
        self,
        x: Tensor,
        num_days: int = 2, 
        day_size: int = 24
    ) -> Tensor:
        """
        Masks the last `horizon` time steps of each time series with NaNs to simulate a forecasting setup.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_chunks, chunk_length).
        horizon : int
            Number of future time steps to mask at the end of each time series.

        Returns
        -------
        torch.Tensor
            Masked version of the input tensor with NaNs in the last `horizon` time steps.
        """
        horizon = num_days * day_size

        if horizon <= 0:
            return x

        T = x.shape[-1]
        if horizon > T:
            raise ValueError(f"Horizon {horizon} is greater than time series length {T}.")

        x_masked = x.clone()
        x_masked[..., -horizon:] = float('nan')
        return x_masked