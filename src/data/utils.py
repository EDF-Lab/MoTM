import logging

from typing import Optional, Tuple, Dict
from pathlib import Path
from omegaconf import DictConfig

import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

from einops import rearrange
import torch

from src.data.constants import nb_timesteps_per_day

class PrepareData():
    """
    Custom mother class for preparing (load, split, etc.) data for the INR.
    
    Args:
        path_train (Path): full path to train data
        path_test (Path): full path to test data (optional)
        seed (int): random seed
    """
    def __init__(
        self,
        path_train: Path,
        path_test: Optional[Path] = None,
        seed: int = 33
    ) -> None:

        # set seed:
        torch.manual_seed(seed)
        
        self.path_train = path_train
        self.path_test  = path_test
        self.seed       = seed
        self.logger     = logging.getLogger(__name__)

        # initialize empty data:
        self.set_train_data()
        self.set_test_data()
    
    def set_train_data(
        self,
        X: Optional[torch.Tensor] = None,
        grid: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store train values and time grids.
        
        Args:
            X (torch.Tensor, optional): tensor of values
            grid (torch.Tensor, optional): tensor of time grids
        """
        
        # store train data: 
        self.train_data = {
            'values': X.clone() if X is not None else X,
            'coords': grid.clone() if grid is not None else grid
        }
        if isinstance(X, torch.Tensor):
            self.logger.info('[Data Prep] Train data (values) of size: {}'.format(X.size()))

    def set_test_data(
        self,
        X: Optional[torch.Tensor] = None,
        grid: Optional[torch.Tensor] = None,
        gt: Optional[torch.Tensor] = None
    ) -> None:
        """
        Store test values and time grids.
        
        Args:
            X (torch.Tensor, optional): if not None, tensor of values
            grid (torch.Tensor, optional): if not None, tensor of time grids
            gt (torch.Tensor, optional): if not None, tensor of ground truth values (no NaNs)
        """

        self.test_data = {
            'values' : X.clone() if X is not None else X,
            'coords' : grid.clone() if grid is not None else grid,
            'gt'     : gt.clone() if gt is not None else gt
        }
        if isinstance(X, torch.Tensor):
            self.logger.info('[Data Prep] Test data (values) of size: {}'.format(X.size()))

    def extract_data(
        self
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return train and test data dict."""
        data_tr = self.train_data
        data_te = self.test_data
        return data_tr, data_te

    def train_test_split(
        self,
        X: torch.Tensor,
        grid: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Split tensors into train and test.
        
        Args:
            X (torch.Tensor): if not None, tensor of values
            grid (torch.Tensor): if not None, tensor of time coordinates
            gt (torch.Tensor): if not None, tensor of ground truth values (no NaNs)
            kwargs: kwargs for train test split

        Returns:
            A tuple of train and test dict with keys `values`, `coords`, `gt` and corresp. data
        """

        random_split = kwargs.get('test_at_random', False)
        test_size    = kwargs.get('test_size', 0.1)
        split_idx    = kwargs.get('test_split_idx', 1)

        gt_tr, gt_te = None, None

        if random_split:
            if isinstance(gt, torch.Tensor):
                X_tr, X_te, grid_tr, grid_te, gt_tr, gt_te = train_test_split( X, grid, gt, test_size = test_size, random_state = self.seed )
            else:
                X_tr, X_te, grid_tr, grid_te = train_test_split( X, grid, test_size = test_size, random_state = self.seed )
        else:
            assert (isinstance(split_idx, int) and split_idx>0) or (isinstance(test_size,float) and test_size<1.0)
            
            split_idx = len(X) - split_idx if isinstance(split_idx, int) else int(len(X) * (1-test_size))
            
            X_tr, grid_tr = X[:split_idx], grid[:split_idx]
            X_te, grid_te = X[split_idx:], grid[split_idx:]
            if isinstance(gt, torch.Tensor):
                gt_tr, gt_te = gt[:split_idx], gt[split_idx:]

        train_data_dict = {
            'values'  : X_tr,
            'coords'  : grid_tr,
            'gt'      : gt_tr
        }
        test_data_dict = {
            'values'  : X_te,
            'coords'  : grid_te,
            'gt'      : gt_te
        }

        return train_data_dict, test_data_dict
        
    def load_ts_data(
        self,
        path_data: Path,
        chunk_size: int = -1
    ) -> torch.Tensor:
        """
        Load univariate time series data. 
        
        Args:
            path_data (Path): path to data to load
            chunk_size (int): if >0, target time dimension of the data
        
        Returns:
            torch.Tensor of shape `[N, T, 1]`
        """

        # get file extension:
        file_extension = path_data.suffix

        # load datasets:
        if file_extension == '.csv':

            df_X   = pd.read_csv(path_data, index_col=0)
            load_X = torch.Tensor(df_X.values).to(torch.float32)
        
        elif file_extension == '.tsv':

            df_X   = pd.read_csv(path_data, sep='\t', index_col=0)
            load_X = torch.Tensor(df_X.values).to(torch.float32)
            
        elif file_extension == '.npy':
            
            np_X   = np.load(path_data)
            load_X = torch.Tensor(np_X).to(torch.float32)
        
        elif file_extension == '.dat':

            np_X   = np.memmap(path_data, dtype='float32', mode='r')
            load_X = torch.Tensor(np_X).to(torch.float32)
        
        elif file_extension == '.pt':            
            
            load_X = torch.load(path_data).to(torch.float32)  
        
        elif len(file_extension) == 0:

            load_X = torch.load(path_data.with_suffix('.pt')).to(torch.float32)  

        else:

            raise NotImplementedError('Data format should be one of: `.pt`, `.csv`, `.tsv`, `.npy`, `.dat`')
        
        # memmap arrays are 1D, need to reshape:
        if load_X.ndim == 1:

            assert chunk_size > 0
            assert (len(load_X) % chunk_size) == 0
            load_X = load_X.reshape((-1, chunk_size, 1))

        elif load_X.ndim == 2:

            load_X = load_X.unsqueeze(-1)

        elif load_X.ndim == 3:

            load_X = rearrange(load_X, 'n c T -> n T c')

        return load_X  # [N, T, 1]

    def make_grid(
        self,
        grid_length: int,
        num_samples: int,
        end_point: float = 1.0,
        start_point: float = 0.0
    ) -> torch.Tensor:
        """
        Build grid of time coordinates for the INR.

        Args:
            grid_length (int): number of time coords
            num_samples (int): number of samples for which to build grid
            end_point (float): last value of the grid
            start_point (float): first value of the grid
        
        Returns:
            torch.Tensor of time coordinates, of shape `(num_samples, grid_length, 1)`
        """

        grid = torch.linspace(start_point, end_point, grid_length).float().unsqueeze(0).repeat_interleave(repeats=num_samples, dim=0)
        grid = grid.unsqueeze(-1) # [N, T, 1]

        return grid
        
class PrepareImputationData(PrepareData):
    """
    Custom class for preparing (load, split, etc.) data for the imputation task.

    Args:
        path_train (Path): full path to train data
        path_test (Path): full path to test data (optional)
        path_gt_test (Path): full path to test ground truth (optional)
        is_train (bool): whether we prepare train or test data
        train_test_split (bool): whether to split data in path_train into train and test data
        grid_length (int): length of the time grid
        seed (int): random seed
        kwargs: additional args for splitting into train and test data
    """
    def __init__(
        self,
        path_train: Optional[Path] = None,
        path_test: Optional[Path] = None,
        path_gt_test: Optional[Path] = None,
        is_train: bool = True,
        train_test_split: bool = False,
        grid_length: Optional[int] = None,
        seed: int = 33,
        **kwargs
    ) -> None:
        super(PrepareImputationData, self).__init__(path_train, path_test, seed)

        #
        self.grid_length = grid_length

        # either the train path or the test path should be specified:
        assert isinstance(path_train, Path) or isinstance(path_test, Path)

        # additional checks if train mode activated:
        if is_train:

            # make sure that a path is provided:
            assert isinstance(path_train, Path)
            mode      = 'train'
            data_path = path_train
            
            # force train_test_split to be False if we provided a test path (path_test is not None):
            train_test_split = (not isinstance(path_test, Path)) and train_test_split

            path_gt_test = None

        # additional checks if test mode activated:
        if not is_train:

            # either we provide a separate test path, or we use the train path with train_test split:
            assert isinstance(path_test, Path) or (isinstance(path_train, Path) and train_test_split)
            mode      = 'test'
            data_path = path_test if isinstance(path_test, Path) else path_train

            # force train_test_split to be False if we provided a test path (path_test is not None):
            train_test_split = not isinstance(path_test, Path)

        # set default test for values and coordinates:
        X_te, grid_te, gt_te = None, None, None

        # (i) load main data (train or test):
        X_tr, grid_tr = self.get_data(data_path)

        # load ground truth data:
        if path_gt_test is not None:
            gt_te, _ = self.get_data(path_gt_test)
            assert gt_te.shape == X_tr.shape
            # assert no NaN ?

        # (ii-a) if we loaded train data but want also to load test data, load test data:
        if is_train and isinstance(path_test, Path):
            X_te, grid_te = self.get_data(path_test)

        # (ii-b) if we loaded train data but want to split into train vs test, split:
        elif train_test_split:
            train_dict, test_dict = self.train_test_split(X_tr, grid_tr, gt=gt_te, **kwargs)
            X_tr, grid_tr        = train_dict['values'], train_dict['coords']
            X_te, grid_te, gt_te = test_dict['values'], test_dict['coords'], test_dict['gt']

        # (iii-c) the only case left in test mode is when we loaded test data in (i) and no train data:
        elif (not is_train):
            X_te, grid_te = X_tr.clone(), grid_tr.clone()

        # store train data:       
        self.set_train_data(
            X    = X_tr if mode == 'train' else None,
            grid = grid_tr if mode == 'train' else None
        )

        # store test data:
        self.set_test_data(
            X    = X_tr if isinstance(path_test, Path) and (not train_test_split) else X_te,
            grid = grid_tr if isinstance(path_test, Path) and (not train_test_split) else grid_te,
            gt   = gt_te
        )

    def get_data(
        self,
        data_path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load observed values and generate time grid.

        Args:
            data_path (Path): full path to data to load

        Returns:
            Values and coordinates as torch.Tensor.
        """
        # load tensor of values:
        X = self.load_ts_data(data_path, self.grid_length).clone() # [N, T, 1]

        grid_length = self.grid_length if isinstance(self.grid_length, int) else X.shape[1]

        # build grid of coordinates:
        grid  = self.make_grid(grid_length = grid_length, num_samples = X.shape[0]) # [N, T, 1]
        
        return X, grid

class PrepareImputationDataWrapper():
    """
    Automatize certain steps in data preparation.

    Args:
        name (str): name of the dataset
        freq (str): sampling frequency of the dataset (pandas datetime style)
        importation (DictConfig): yaml config with path to train test data etc.
        weight (float): weight for dataloader sampling
        seed (int): random seed
    """

    def __init__(
        self,
        name: str,
        freq: str,
        importation: DictConfig,
        weight: float = 1.0,
        seed: int = 42
    ):
        
        self.name        = name
        self.freq        = freq
        self.importation = importation
        self.seed        = seed
        self.weight      = weight
        
    def prepare_data(
        self,
        task_cfg: DictConfig,
        is_train: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        
        path_train = Path(self.importation.dir) / self.importation.train_file

        test_file = self.importation.test_file
        path_test = Path(self.importation.dir) / test_file if (test_file is not None) else test_file

        gt_file = self.importation.test_gt_file
        path_gt = Path(self.importation.dir) / gt_file if (gt_file is not None) else gt_file

        window_len_in_days  = task_cfg.get('window_len_in_days', 28)
        sampling_freq       = self.freq
        train_grid_length   = int( nb_timesteps_per_day(sampling_freq) * window_len_in_days )
        self.train_grid_len = train_grid_length
        
        # load data:
        series = PrepareImputationData(
            path_train       = path_train,
            path_test        = path_test,
            path_gt_test     = path_gt,
            is_train         = is_train,
            train_test_split = False,
            grid_length      = train_grid_length,
            seed             = self.seed,
        )
        if is_train:
            data_, _ = series.extract_data()
        else:
            _, data_ = series.extract_data()

        data_['train_grid_len'] = train_grid_length
        data_['weight'] = self.weight
        data_['freq']   = self.freq

        return data_

class ConcatPrepareImputationData():
    
    def __init__(self, *builders) -> None:
        super().__init__()
        assert len(builders) > 0, "Must provide at least one builder to ConcatPrepareImputationData"
        assert all(
            isinstance(builder, PrepareImputationDataWrapper) for builder in builders
        ), "All builders must be instances of PrepareImputationDataWrapper"
        self.builders = builders
    
    def load_dataset(
        self,
        task_cfg: DictConfig,
        is_train: bool = True
    ):
        return {
            builder.name:builder.prepare_data(task_cfg, is_train) for builder in self.builders
        }