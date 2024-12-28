import numpy as np
import random as rd
import setigen as stg
from numba import njit, prange
from Decorators import TimeMeasure

@TimeMeasure
def generate_injection_list(signal_split: dict, number_slides: int):
    """
    Function to generate a list including the injection type for each data slide

    Parameters:
    - signal_split: dictionary containing the split of the signals (e.g. {"Background": 0.5, "Linear": 0.5})
    - number_slides: number of slides in the data

    Returns:
    - output_dictionary: dictionary containing the injection type for each data slide (e.g. {"Background": 50, "Linear": 50, ...})
    """
    # Initialize the injection list
    keys = np.array(list(signal_split.keys()))

    proportions = np.array(list(signal_split.values()))

    number_each_injection = (proportions * number_slides).astype(int)

    # Check if the number of slides is equal to the sum of the number of each injection
    if np.sum(number_each_injection) != number_slides:
        # If not, add the difference to the first injection
        number_each_injection[0] += number_slides - np.sum(number_each_injection)
    
    # Generate the injection list
    possible_indexes = np.arange(number_slides)

    output_dictionary = {}

    # Loop through the injections
    for i, number in enumerate(number_each_injection):

        # Select the indexes for the current injection
        selected_numbers = np.random.choice(possible_indexes, size=number, replace=False)
        print(f"For key: {keys[i]}, selected numbers: {len(selected_numbers)}")

        # Add the selected indexes to the output dictionary
        output_dictionary[keys[i]] = selected_numbers 

        # Remove the selected indexes from the possible indexes
        possible_indexes = np.setdiff1d(possible_indexes, selected_numbers)
        print(f"Possible indexes: {len(possible_indexes)}")

    return output_dictionary




def inject_signals(data: np.ndarray, 
                   signal_split: dict, 
                   signal_params: np.ndarray, 
                   loading_bar_bool: bool = True):
    """
    """

    # Generate the injection list
    index_dictionary = generate_injection_list(signal_split, data.shape[1])
    keys = list(signal_split.keys())

    # Loop through the injections
    for key in keys:
        indexes = index_dictionary[key]
        for index in indexes:
            data[:, index, :] = add_injection(data[:, index, :], signal_params[key], injection_type = key, loading_bar_bool = loading_bar_bool)

            
if __name__ == "__main__":
    signal_split = {"Background": 0.5, "Linear": 0.5}
    number_slides = 99
    generate_injection_list(signal_split, number_slides)