import abc
from typing import Union
import torch

class AbstractCustomScaler(abc.ABC):

    @abc.abstractmethod
    def fit(
        self,
        X: torch.Tensor
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def inv_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor: ...

class CustomStandardScaler(AbstractCustomScaler):

    def __init__(
        self,
        dim: Union[int, tuple] = 1,
        epsilon: float = 1e-7
    ) -> None:

        self.mean    = None
        self.std     = None 
        self.dim     = dim
        self.epsilon = epsilon
        
    def fit(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
         
        # X has shape (batch_size, grid_size, d) (in general: d=1)

        self.mean = torch.mean(X, dim=self.dim, keepdims=True)
        self.std  = torch.std(X, dim=self.dim, keepdims=True) + self.epsilon

    def transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_transform = (X - self.mean) / self.std

        return X_transform
    
    def inv_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_inv_transform = X * self.std + self.mean

        return X_inv_transform

class CustomMinmaxScaler(AbstractCustomScaler):

    def __init__(
        self,
        dim: Union[int, tuple] = 1,
        epsilon: float = 1e-8
    ) -> None:

        self.xmin    = None
        self.xmax    = None 
        self.dim     = dim
        self.epsilon = epsilon
        
    def fit(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
         
        # X has shape (batch_size, grid_size, d) (in general: d=1)

        self.xmin = torch.min(X, dim=self.dim, keepdims=True)[0]
        self.xmax = torch.max(X, dim=self.dim, keepdims=True)[0]

    def transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_transform = (X - self.xmin) / (self.xmax - self.xmin + self.epsilon)

        return X_transform
    
    def inv_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_inv_transform = (self.xmax - self.xmin + self.epsilon) * X + self.xmin

        return X_inv_transform