import numpy as np
import random as rd
import setigen as stg
from numba import njit, prange
from Decorators import TimeMeasure

@TimeMeasure
def generate_injection_list(split: dict, number_slides: int, unequal_split_index: int = 0):
    """
    Function to generate a list including the injection type for each data slide

    Parameters:
    - split: dictionary containing the split of the signals (e.g. {"Background": 0.5, "Linear": 0.5} or {"True": 0.5, "False": 0.5})
    - number_slides: number of slides in the data
    - unequal_split_index: index of the injection type that should be given the extra slide if the number of slides is not divisible by the number of injections

    Returns:
    - output_dictionary: dictionary containing the injection type for each data slide (e.g. {"Background": 50, "Linear": 50, ...} or {"True": 50, "False": 50})
    """
    # Initialize the injection list
    keys = np.array(list(split.keys()))

    proportions = np.array(list(split.values()))

    number_each_injection = (proportions * number_slides).astype(int)

    # Check if the number of slides is equal to the sum of the number of each injection
    if np.sum(number_each_injection) != number_slides:
        # If not, add the difference to the first injection
        number_each_injection[unequal_split_index] += number_slides - np.sum(number_each_injection)
    
    # Generate the injection list
    possible_indexes = np.arange(number_slides)

    output_dictionary = {}

    # Loop through the injections
    for i, number in enumerate(number_each_injection):

        # Select the indexes for the current injection
        selected_numbers = np.random.choice(possible_indexes, size=number, replace=False)
        # print(f"For key: {keys[i]}, selected numbers: {len(selected_numbers)}")

        # Add the selected indexes to the output dictionary
        output_dictionary[keys[i]] = selected_numbers 

        # Remove the selected indexes from the possible indexes
        possible_indexes = np.setdiff1d(possible_indexes, selected_numbers)
        # print(f"Possible indexes: {len(possible_indexes)}")

    return output_dictionary

def inject_signals(data: np.ndarray, 
                   signal_split: dict, 
                   true_false_split: dict,
                   signal_params: np.ndarray, 
                   loading_bar_bool: bool = True,
                   num_workers: int = 20):
    """
    """

    # Generate the injection list
    signal_index_dictionary = generate_injection_list(signal_split, data.shape[0])
    keys = list(signal_split.keys())

    # Loop through the injections
    for key in keys:
        print(f"Injecting {key} signals")
        indexes = signal_index_dictionary[key]
        print(f"{len(indexes)=}")
        data[indexes, :, :, :] = add_injection_type(data[indexes, :, :, :], signal_params, injection_type = key, true_false_split = true_false_split, loading_bar_bool = loading_bar_bool, num_workers = num_workers)

    return data

# @njit(parallel=True)
def threshold_and_normalise_data(data: np.ndarray, theshold_sigma):
    data = np.log(data)  # Shape: (N, 6, 16, 4096)

    # Compute mean and std along the last axis (axis=-1, corresponding to 4096)
    mean = np.mean(data, axis=-1, keepdims=True)  # Shape: (N, 6, 16, 1)
    std = np.std(data, axis=-1, keepdims=True)    # Shape: (N, 6, 16, 1)

    # Create a mask for all entries
    mask = data > mean + 5 * std  # Shape: (N, 6, 16, 4096)

    # Apply the mask to create the altered array
    data = np.where(mask, 1, 0)  # Shape: (N, 6, 16, 4096)

    return data

    """
    """
    if injection_type == "Background":
        return data  # No injection needed
    
    cadences = generate_frames(data)
    if injection_type == "Linear":
        data = add_linear(data, signal_params, loading_bar_bool)
    else:
        raise ValueError(f"Invalid injection type: {injection_type}")

    return data

@njit(parallel=True)
def generate_frames(data):
    """
    """
    cadences = []
    for i in prange(data.shape[0]):
        frame_list = []
        for j in range(6):
            frame = stg.Frame.from_data(df=2.7939677238464355*u.Hz,
                                        dt=18.25361108*u.s,
                                        fch1=0*u.MHz,
                                        ascending = True,
                                        data=data[i, j])
            frame_list.append(frame)
        cadences.append(stg.OrderedCadence(frame_list, order = "ABACAD"))
    return cadences

            
if __name__ == "__main__":
    signal_split = {"Background": 0.5, "Linear": 0.5}
    number_slides = 99
    output_dictionary = generate_injection_list(signal_split, number_slides)
    print(f"{len(output_dictionary["Background"])=}, {len(output_dictionary["Linear"])=}")
    