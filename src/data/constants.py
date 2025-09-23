
def nb_timesteps_per_day(
    freq: str
) -> int:
    """
    Args:
        freq (str): pandas-like sampling rate, e.g. '10T' for 10 minute
    
    Returns:
        The number of timesteps in a day sampled every `freq` instant
    """

    freq_to_minute = {
        '5T'   : 5,
        '10T'  : 10,
        '15T'  : 15,
        '30T'  : 30,
        '1H'   : 60,
        '2H'   : 120,
        '3H'   : 180,
        '6H'   : 360,
    }
    
    assert freq in freq_to_minute.keys()

    sampling_rate_in_minutes = freq_to_minute[freq]

    one_day = 24 * 60 // sampling_rate_in_minutes
    
    return one_day

def get_latent_norm(
    inner_lr: float,
    inner_steps: int
) -> float:
    """
    Return the target norm of the latent code, for numerical stability (empirically set).

    Args:
        inner_lr (float): Learning rate for inner optimization.
        inner_steps (int): Number of inner optimization steps.
    """

    list_inner_lr    = [0.01, 0.05, 0.1, 0.5]
    list_inner_steps = [1, 2, 3, 5]

    assert inner_lr in list_inner_lr
    assert inner_steps in list_inner_steps

    znorm = {
        1: [0.080, 0.188, 0.270, 0.605],
        2: [0.107, 0.220, 0.308, 0.686],
        3: [0.114, 0.227, 0.321, 0.716],
        5: [0.134, 0.260, 0.337, 0.765]
    }

    x = znorm[inner_steps]
    x = x[list_inner_lr.index(inner_lr)]

    return x