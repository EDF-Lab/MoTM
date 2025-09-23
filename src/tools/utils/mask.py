from typing import Union, Tuple
import torch

def std_mask(
    x: torch.Tensor,
    threshold: float = 8.0,
    return_mask: bool = False,
    *args: torch.Tensor
) -> Union[Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor], torch.Tensor]]:
    """
    Filter out somewhat odd samples based on the ratio max/std.

    Parameters:
        x (torch.Tensor): tensor of shape [B, T, 1] to mask, with time as the 2nd dimension
        threshold (float): keep samples below threshold
        return_mask (bool): whether to return the mask or not
        *args (torch.Tensor): extra tensors on which the mask is applied
    
    Returns:
        out (torch.Tensor): output tensor of shape [B_out, T, 1], B_out < B.
        mask (torch.Tensor): if return_mask=True, return the mask
    """
    
    mask = x.max(1)[0] / (1e-7 + x.std(1)) < threshold
    mask = mask[...,0]
    
    out  = [x[mask]]
    for y in args:
        out.append(y[mask])

    if return_mask:
        return tuple(out), mask
    else:
        return tuple(out)