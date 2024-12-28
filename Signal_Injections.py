import numpy as np
import random as rd


def generate_injection_list(signal_split: dict, number_slides: int):
    """
    Function to generate a list including the injection type for each data slide

    Parameters:
    - signal_split: dictionary containing the split of the signals (e.g. {"Background": 0.5, "Linear": 0.5})
    - number_slides: number of slides in the data

    Returns:
    - injection_array: np.array of the injection type for each data slide in string format (e.g. ["Background", "Linear", "Background", "Linear"])
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

    injection_array = np.empty(number_slides, dtype=str)

    for i, number in enumerate(number_each_injection):
        selected_numbers = np.random.choice(possible_indexes, size=number, replace=False)
        print(f"For key: {keys[i]}, selected numbers: {len(selected_numbers)}")
        injection_array[selected_numbers] = keys[i]
        possible_indexes = np.setdiff1d(possible_indexes, selected_numbers)
        print(f"Possible indexes: {len(possible_indexes)}")

    return injection_array




def inject_signals(data: np.ndarray, 
                   signal_split: dict, 
                   signal_params: np.ndarray, 
                   loading_bar_bool: bool = True):
    """
    """

    # Generate the injection list
    injection_list = generate_injection_list(signal_split, data.shape[1])

            
if __name__ == "__main__":
    signal_split = {"Background": 0.5, "Linear": 0.5}
    number_slides = 99
    generate_injection_list(signal_split, number_slides)