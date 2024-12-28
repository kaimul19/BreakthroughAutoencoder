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





def add_injection_type(data: np.ndarray, signal_params: np.ndarray, injection_type: str, true_false_split: dict, loading_bar_bool: bool = True, num_workers: int = 20):
    """
    Add a specific injection type to the data.

    Parameters:
    - data: Input data array.
    - signal_params: Parameters for the signal injection.
    - injection_type: The type of injection to add.
    - true_false_split: Dictionary specifying indices for True/False signals.
    - loading_bar_bool: If True, display a loading bar.
    - num_workers: Number of workers for parallel processing.

    Returns:
    - Updated data with injections.
    """
    if injection_type == "Background":
        return data  # No injection needed

    # Generate true/false index dictionary
    true_false_index_dictionary = generate_injection_list(true_false_split, data.shape[0], unequal_split_index=1)
    # unified_True_false_list = [""] * data.shape[0]

    # for true_index in true_false_index_dictionary["True"]:
    #     print(f"{true_index=}")
    #     unified_True_false_list[true_index] = "True"

    # for false_index in true_false_index_dictionary["False"]:
    #     print(f"{false_index=}")
    #     unified_True_false_list[false_index] = "False"

    # print(true_false_index_dictionary)
    # Generate frames for each cadence
    cadences = generate_frames(data, repeat(true_false_index_dictionary), max_workers=num_workers)

    if injection_type == "Linear":
        # Parallel processing
        with ThreadPoolExecutor() as executor:
            updated_cadences = list(executor.map(add_linear, cadences, repeat(signal_params)))

    else:
        raise ValueError(f"Invalid injection type: {injection_type}")


    data = return_to_data(updated_cadences)

    return data


def return_to_data(cadences):
    """
    Convert a list of cadences back to the original-shaped NumPy array.

    Parameters:
    - cadences: List of OrderedCadence objects.

    Returns:
    - A NumPy array with the original shape of the data.
    """
    # Define a helper function to extract data from a single cadence
    def extract_cadence_data(cadence):
        return np.array([frame.get_data() for frame in cadence])

    # Use ThreadPoolExecutor for parallel data extraction
    with ThreadPoolExecutor() as executor:
        data_list = list(executor.map(extract_cadence_data, cadences))

    # Stack the list of NumPy arrays into a single array
    data_array = np.stack(data_list, axis=0)
    return data_array



def add_linear(cadence, signal_params):
    """
    Add a linear signal to the "A" observations in the given cadence.

    Parameters:
    - cadence: An OrderedCadence object.
    - signal_params: Parameters for the signal injection (e.g., [f_start, drift_rate, snr]).

    Returns:
    - Updated OrderedCadence object.
    """
    # Add a signal to frames labeled "A"
    cadence.by_label("A").add_signal(stg.constant_path(f_start=cadence[0].get_frequency(index=600),
                               drift_rate=4*u.Hz/u.s),
                           stg.constant_t_profile(level=cadence[0].get_intensity(snr=30)),
                           stg.sinc2_f_profile(width=80*cadence[0].df*u.Hz),
                           stg.constant_bp_profile(level=1),
                           doppler_smearing=True)
    

    # signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(200),
    #                                         drift_rate=2*u.Hz/u.s),
    #                       stg.constant_t_profile(level=1),
    #                       stg.box_f_profile(width=20*u.Hz),
    #                       stg.constant_bp_profile(level=1))
    return cadence








def generate_frames(data, true_false_index_dictionary, max_workers=20):
    """
    Generate frames and cadences in parallel using ProcessPoolExecutor.

    Parameters:
    - data: 3D numpy array with shape (n_cadences, n_frames, frame_data)
    - true_false_index_dictionary: Dictionary specifying True/False indices.
    - max_workers: Number of parallel processes to use.

    Returns:
    - List of OrderedCadence objects.
    """
    cadences = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # No need to multiply the dictionary; it maps 1-to-1 with data
        cadences = list(executor.map(process_into_cadence, data, true_false_index_dictionary))

    return cadences


def process_into_cadence(data_slice, true_false_index_dictionary):
    """
    Process a single cadence of data and generate an OrderedCadence.

    Parameters:
    - data_slice: A 2D numpy array (n_frames, frame_data) for one cadence.
    - true_false_index_dictionary: Dictionary specifying True/False indices.

    Returns:
    - OrderedCadence object.
    """
    # Determine the order based on whether the cadence index is in True or False
    cadence_index = np.where(data_slice == data_slice)[0][0]  # Assuming slice index maps to global index
    if cadence_index in true_false_index_dictionary["True"]:
        order = "ABACAD"
    elif cadence_index in true_false_index_dictionary["False"]:
        order = "AAAAAA"
    else:
        raise ValueError(f"Cadence index {cadence_index} not found in True/False index dictionary.")

    # Create the frames
    frame_array = np.empty(data_slice.shape[0], dtype=stg.Frame)
    for j in range(data_slice.shape[0]):  # Loop over frames in the cadence
        frame = stg.Frame.from_data(
            df=2.7939677238464355 * u.Hz,
            dt=18.25361108 * u.s,
            fch1=0 * u.MHz,
            ascending=True,
            data=data_slice[j]
        )
        frame_array[j] = frame

    # Create an OrderedCadence for this set of frames
    return stg.OrderedCadence(frame_array, order=order)



            
if __name__ == "__main__":
    signal_split = {"Background": 0.5, "Linear": 0.5}
    number_slides = 99
    output_dictionary = generate_injection_list(signal_split, number_slides)
    print(f"{len(output_dictionary["Background"])=}, {len(output_dictionary["Linear"])=}")
    