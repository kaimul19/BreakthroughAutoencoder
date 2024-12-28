import numpy as np
import random as rd

def inject_signals(data: np.ndarray, 
                   signal_split: dict, 
                   signal_params: np.ndarray, 
                   loading_bar_bool: bool = True):
    """
    """

    # Generate the injection list
    injection_list = generate_injection_list(signal_split, data.shape[1])

