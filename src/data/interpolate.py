import numpy as np

def impute_cst_with_value(data_to_impute):
    """
    Remplit les NaNs dans une série temporelle par une valeur si :
    - Les deux valeurs à gauche et les deux valeurs à droite du segment de NaNs sont identiques.

    Arguments :
    - data_to_impute : ndarray de forme (N, T) avec des NaNs.

    Retourne :
    - data_imputed : ndarray avec les NaNs remplis si les conditions sont remplies.
    - mask : ndarray de même forme que data, True où l'imputation a eu lieu, False sinon.
    """
    data = data_to_impute.copy()
    mask = np.zeros_like(data, dtype=bool)

    for i in range(data.shape[0]):  # Parcourir chaque série temporelle
        series = data[i, :]

        # Identifier les indices des NaNs
        nan_indices = np.isnan(series)

        t = 0
        while t < len(series):
            if nan_indices[t]:
                # Trouver le début et la fin du segment de NaNs consécutifs
                start = t
                while t < len(series) and nan_indices[t]:
                    t += 1
                end = t - 1  # Fin du segment de NaNs consécutifs

                # Vérifier les deux valeurs à gauche et les deux à droite
                if (
                    start >= 2 and end < len(series) - 2 and  # S'assurer qu'on a assez de place
                    not np.isnan(series[start - 2:start]).any() and
                    not np.isnan(series[end + 1:end + 3]).any() and
                    np.all(series[start - 2:start] == series[start - 2]) and
                    np.all(series[end + 1:end + 3] == series[end + 1]) and
                    series[start - 2] == series[end + 1]
                ):
                    # Imputer le segment entier avec la valeur des bords
                    series[start:end + 1] = series[start - 2]
                    mask[i, start:end + 1] = True  # Marquer les NaNs imputés
            else:
                t += 1  # Passer à l'élément suivant si ce n'est pas un NaN

    return data, mask


def impute_single_nan_with_interpolation(data_to_impute):

    data = data_to_impute.copy()
    mask = np.zeros_like(data, dtype=bool)

    for i in range(data.shape[0]):  # Parcourir chaque série temporelle
        series = data[i, :]
        nan_indices = np.isnan(series)
        
        for t in range(1, len(series) - 1):  # Ignorer les bords
            if nan_indices[t]:
                if not nan_indices[t - 1] and not nan_indices[t + 1]:
                    # Interpolation linéaire si trou de taille 1
                    series[t] = (series[t - 1] + series[t + 1]) / 2
                    mask[i, t] = True  # Marquer l'imputation
    
    return data, mask


def apply_imputations(data):

    data_after_first_imputation, mask_after_first_imputation = impute_cst_with_value(data)
    data_imputed, mask_after_second_imputation = impute_single_nan_with_interpolation(data_after_first_imputation)
    combined_mask = mask_after_first_imputation | mask_after_second_imputation
    
    return data_imputed, combined_mask