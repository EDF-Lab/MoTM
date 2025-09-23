from typing import Optional

import numpy as np

import torch
    
def interpolate_series_torch(
    series_t: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Interpole linéairement les valeurs manquantes (mask == True)
    dans un tenseur PyTorch de forme (N, T).
    
    Args:
        series_t: torch.Tensor of shape (N, T), avec des valeurs manquantes.
        mask: torch.BoolTensor de même forme, True là où les valeurs sont manquantes.

    Returns:
        series_interp: torch.Tensor de forme (N, T), avec les valeurs interpolées.
    """
    series_np = series_t.cpu().numpy()
    mask_np = mask.cpu().numpy()

    N, T = series_np.shape
    t = np.arange(T)
    series_interp_np = np.copy(series_np)

    for i in range(N):
        observed = ~mask_np[i]
        if np.sum(observed) < 2:
            continue  # Pas assez de points pour interpoler
        series_interp_np[i] = np.interp(t, t[observed], series_np[i][observed])
    
    return torch.tensor(series_interp_np, dtype=series_t.dtype, device=series_t.device).unsqueeze(-1)

def impute_by_offset(
    series_t: torch.Tensor,
    mask: torch.Tensor,
    step: int = 24,
    max_lag: Optional[int] = None
) -> torch.Tensor:
    """
    Impute les valeurs manquantes par valeurs à ±24, ±48, ...,
    sinon par la moyenne des valeurs observées de la série correspondante.

    Args:
        series_t: (N, T) tensor avec NaN aux endroits manquants.
        mask: bool tensor (N, T), True là où la valeur est manquante.
        step: intervalle entre les tentatives (par défaut 24).
        max_lag: distance max testée (si None, on prend T // 2).

    Returns:
        series_filled: (N, T) tensor imputé.
    """
    N, T = series_t.shape
    if max_lag is None:
        max_lag = T // 2

    series_filled = series_t.clone()
    
    for n in range(N):
        # Moyenne des valeurs observées de la série n
        observed_values = series_t[n][~mask[n]]
        mean_value = observed_values.mean() if observed_values.numel() > 0 else torch.tensor(0.0, dtype=series_t.dtype, device=series_t.device)

        for t in range(T):
            if not mask[n, t]:
                continue  # valeur déjà observée
            found = False
            for offset in range(step, max_lag + 1, step):
                for direction in [-1, 1]:
                    neighbor_t = t + direction * offset
                    if 0 <= neighbor_t < T and not mask[n, neighbor_t]:
                        series_filled[n, t] = series_t[n, neighbor_t]
                        found = True
                        break
                if found:
                    break
            if not found:
                series_filled[n, t] = mean_value  # fallback par moyenne
    
    return series_filled.unsqueeze(-1)
