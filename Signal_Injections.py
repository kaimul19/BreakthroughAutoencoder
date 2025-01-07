import numpy as np
import random as rd
import setigen as stg
from numba import njit, prange
from Decorators import TimeMeasure
from astropy import units as u
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from itertools import repeat

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

    true_false_dictionary = {}


    # Loop through the injections
    for key in keys:
        print(f"Injecting {key} signals")
        indexes = signal_index_dictionary[key]
        print(f"{len(indexes)=}")
        print(f"data[indexes, :, :, :].shape={data[indexes, :, :, :].shape}")

        data[indexes, :, :, :], dict_to_append = add_injection_type(data[indexes, :, :, :], signal_params, injection_type = key, true_false_split = true_false_split, loading_bar_bool = loading_bar_bool, num_workers = num_workers)
        true_false_dictionary = update_dictionary(true_false_dictionary, dict_to_append)
    for i in range(0,100,10):
    # subplot now
        fig, axs = plt.subplots(5, 1)
        for j in range(5):
            axs[j].imshow(data[i, j, :, :], aspect='auto')
        plt.savefig(f"1test{i}.png")
        plt.close()
    data = threshold_and_normalise_data(data, 2)
    print(f"{data.shape=}")

    return data

def update_dictionary(dictionary_to_update, dict_to_append):
    """
    Update a dictionary with another dictionary.

    Parameters:
    - dictionary_to_update: The dictionary to update.
    - dict_to_append: The dictionary to append.

    Returns:
    - The updated dictionary.

    
    """
    for key in dict_to_append:
        if key in dictionary_to_update:
            dictionary_to_update[key].extend(dict_to_append[key])
        else:
            dictionary_to_update[key] = dict_to_append[key]
    return dictionary_to_update

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





def add_injection_type(data: np.ndarray, signal_params: np.ndarray, injection_type: str, true_false_split: dict, loading_bar_bool: bool = True, num_workers: int = 20, bin_width = 4096):
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
    - Updated data array.
    - Dictionary specifying True/False indices.
    """
    if injection_type == "Background":
        true_false_index_dictionary = {}
        return data, true_false_index_dictionary  # No injection needed

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
            print("Injecting linear signals")
            print("True")
            cadences[true_false_index_dictionary["True"]] = list(executor.map(add_linear, 
                                                                              cadences[true_false_index_dictionary["True"]], 
                                                                              repeat(signal_params), 
                                                                              repeat(True), 
                                                                              repeat(bin_width)))
            print("False")
            cadences[true_false_index_dictionary["False"]] = list(executor.map(add_linear, 
                                                                               cadences[true_false_index_dictionary["False"]], 
                                                                               repeat(signal_params), 
                                                                               repeat(False), 
                                                                               repeat(bin_width)))
    else:
        raise ValueError(f"Invalid injection type: {injection_type}")


    data = return_to_data(cadences)

    return data, true_false_index_dictionary


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



def add_linear(cadence, signal_params, true_or_false, bin_width = 4096):
    """
    Add a linear signal to the "A" observations in the given cadence.

    Parameters:
    - cadence: An OrderedCadence object.
    - signal_params: Parameters for the signal injection (e.g., [f_start, drift_rate, snr]).

    Returns:
    - Updated OrderedCadence object.
    """

    print("made it to the funciton")
    print(f"{cadence=}")
    print(f"{cadence.shape=}")
    start = int(rd.random()*(bin_width-1))+1    # Start the line in some random point along frequency
    RandMultiplier = rd.choice([-1, 1])
    if RandMultiplier == -1:
        # print("Negative")
        MaxSlope = -(96/start) * (18.25361108/2.7939677238464355)
        slope = MaxSlope 
        drift = (1/slope)* rd.random()
        
    else:
        # print("positive")
        spaceToEnd = bin_width - start
        MaxSlope = (96/spaceToEnd) * (18.25361108/2.7939677238464355)
        slope = MaxSlope 
        drift = (1/slope)* rd.random()

    # if wide == False:
    #     width = rd.random()*50+abs(drift)*18
    # else:
    #     width = rd.uniform(400,900)+abs(drift)*18



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

    return np.array(cadences)


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
    signal_split = {"Background": 0.0, "Linear": 1}
    number_slides = 100
    output_dictionary = generate_injection_list(signal_split, number_slides)
    # print(f"{len(output_dictionary["Background"])=}, {len(output_dictionary["Linear"])=}")

    true_false_split = {"True": 1, "False": 0}
    output_dictionary2 = generate_injection_list(true_false_split, number_slides)
    # print(f"{output_dictionary2=}")
    # print(f"{output_dictionary2["True"]=}")
    # print(f"{type(output_dictionary2["True"])=}")
    # if 0 in output_dictionary2["True"]:
    #     print("In True")
    # if 0 in output_dictionary2["False"]:
    #     print("In False")
    mask = generate_injection_list(true_false_split, number_slides)

    print(f"{np.sum(mask)=}, {len(mask)=}")

    import os 
    print(f"{os.getcwd()=}")
    data_shape = np.load('HIP13402/shape.npy')

    data_shape = tuple(int(dim) for dim in data_shape)
    data = np.memmap('HIP13402/seperated_raw_data.npy', dtype='float32', mode='c', shape=data_shape)
    print(f"{data[200:300].shape=}=")
    data2 = inject_signals(data[400:500], signal_split = signal_split, true_false_split = true_false_split, signal_params = np.array([1000, 0, 10000.0]), num_workers=20)
    print(f"{data2.shape=}=")
    for i in range(0,100,10):
        # subplot now
        fig, axs = plt.subplots(5, 1)
        for j in range(5):
            axs[j].imshow(data2[i, j, :, :], aspect='auto')
        plt.savefig(f"2test{i}.png")
        plt.close()
    
