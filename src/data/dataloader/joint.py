from typing import Sequence, Union
import random

import numpy as np

import torch
from torch.utils.data import Sampler

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)

class JointTrainSampler(Sampler):

    def __init__(
        self, 
        batch_size: Union[int, Sequence[int]],
        seq_dataset_len: Sequence[int]
    ) -> None:
        # https://discuss.pytorch.org/t/how-to-concatenate-different-datasets-each-with-different-dimensions/123218/2
        
        self.ntrain     = len(seq_dataset_len)
        self.batch_size = [batch_size] * self.ntrain if isinstance(batch_size, int) else batch_size

        cum_data_len = np.cumsum([0]+list(seq_dataset_len))
        self.indices = [list(range(idx1,idx2)) for idx1, idx2 in zip(cum_data_len[:-1], cum_data_len[1:])]
        
        assert len(self.indices) == self.ntrain
        assert len(self.batch_size) == self.ntrain
    
    def __iter__(self):
        
        for val in self.indices:
            random.shuffle(val)
        
        all_batches = [list(chunk(index, bs)) for index, bs in zip(self.indices, self.batch_size)]
        all_batches = [x for l in all_batches for x in l]
        
        all_batches = [batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return sum([len(index) // bs + int((len(index) % bs) > 0) for index,bs in zip(self.indices, self.batch_size)])
