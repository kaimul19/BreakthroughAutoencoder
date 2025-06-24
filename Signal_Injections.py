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
from tqdm import tqdm
import pickle
import os


# @TimeMeasure
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

        # Add the selected indexes to the output dictionary
        output_dictionary[keys[i]] = selected_numbers 

        # Remove the selected indexes from the possible indexes
        possible_indexes = np.setdiff1d(possible_indexes, selected_numbers)

    return output_dictionary

@TimeMeasure
def inject_signals(data: np.ndarray, 
                   signal_split: dict, 
                   true_false_split: dict,
                   signal_params: np.ndarray, 
                   loading_bar_bool: bool = True,
                   num_workers: int = 20):
    """
    Function to inject signals into the data.
    Parameters:
    - data: Input data array with shape (N, 6, 16, 4096).
    - signal_split: Dictionary specifying the signal type split (e.g., {"Background": 0.2, "Linear": 0.8}).
    - true_false_split: Dictionary specifying the True/False split (e.g., {"True": 0.3, "False": 0.7}).
    - signal_params: Parameters for the signal injection (e.g., [f_start, drift_rate, snr]).
    - loading_bar_bool: If True, display a loading bar.
    - num_workers: Number of workers for parallel processing.
    Returns:
    - Updated data array with injected signals.
    shape: (N, 6, 16, 4096) where N is the number of cadences

    """

    # Generate the injection list
    signal_index_dictionary = generate_injection_list(signal_split, data.shape[0])
    keys = list(signal_split.keys())

    true_false_dictionary = {}
    metadata = []


    # Loop through the injections
    for key in keys:
        indexes = signal_index_dictionary[key]

        # add the injection type to the data
        data[indexes, :, :, :], dict_to_append = add_injection_type(data[indexes, :, :, :], signal_params, injection_type = key, true_false_split = true_false_split, loading_bar_bool = loading_bar_bool, num_workers = num_workers)
        
        # Update the true_false_dictionary with the new injections
        true_false_dictionary = update_dictionary(true_false_dictionary, dict_to_append)

        # build metadata for the injection
        metadata += build_injection_metadata(
            true_false_dictionary=dict_to_append,
            injection_type=key,
            total_cadences=len(indexes),
            indexes_used=indexes
        )
    data = threshold_and_normalise_data(data, 5)

    return data, metadata

# @TimeMeasure
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
@TimeMeasure
def threshold_and_normalise_data(data: np.ndarray, theshold_sigma):
    data = np.log(data)  # Shape: (N, 6, 16, 4096)

    # Compute mean and std along the last axis (axis=-1, corresponding to 4096)
    mean = np.mean(data, axis=-1, keepdims=True)  # Shape: (N, 6, 16, 1)
    std = np.std(data, axis=-1, keepdims=True)    # Shape: (N, 6, 16, 1)

    # Create a mask for all entries
    mask = data > mean + theshold_sigma * std  # Shape: (N, 6, 16, 4096)

    # Apply the mask to create the altered array
    data = np.where(mask, 1, 0)  # Shape: (N, 6, 16, 4096)

    return data




# @TimeMeasure
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

    # Generate frames for each cadence
    cadences = generate_frames(data, repeat(true_false_index_dictionary), max_workers=num_workers)

    # Go through and add the signals to the cadences

    # Add the linear lines
    if injection_type == "Linear":
        # Parallel processing
        with ThreadPoolExecutor() as executor:
            cadences[true_false_index_dictionary["True"]] = list(
                tqdm(
                    executor.map(
                        add_linear, 
                        cadences[true_false_index_dictionary["True"]], 
                        repeat(signal_params), 
                        repeat(True), 
                        repeat(bin_width),
                    ),
                    total=len(true_false_index_dictionary["True"]), 
                    desc="Injecting True linear signals",
                    )
                )

            
            print("Injecting False linear signals")
            cadences[true_false_index_dictionary["False"]] = list(
                tqdm(
                    executor.map(
                        add_linear, 
                        cadences[true_false_index_dictionary["False"]], 
                        repeat(signal_params), 
                        repeat(False), 
                        repeat(bin_width),
                    ),
                    total=len(true_false_index_dictionary["False"]), 
                    desc="Injecting False linear signals",
                    )
                )
    else:
        raise ValueError(f"Invalid injection type: {injection_type}")


    data = return_to_data(cadences)

    return data, true_false_index_dictionary

# @TimeMeasure
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
        data_list = list(
            tqdm(
                executor.map(extract_cadence_data, cadences),
                total=len(cadences),  # Total number of cadences to process
                desc="Extracting Cadence Data",
            )
        )
    # Stack the list of NumPy arrays into a single array
    data_array = np.stack(data_list, axis=0)
    return data_array


# @TimeMeasure
def add_linear(cadence, signal_params, true_or_false, bin_width=4096):
    """
    Add a linear signal to the "A" observations in the given cadence.

    Parameters:
    - cadence: An OrderedCadence object.
    - signal_params: Parameters for the signal injection (e.g., [f_start, drift_rate, snr]).
    - true_or_false: Boolean, determines if the signal is a true or false signal.
    - bin_width: Width of the frequency bins.

    Returns:
    - Updated OrderedCadence object.
    """

    # Random starting frequency index
    start = int(rd.random() * (bin_width - 1)) + 1
    RandMultiplier = rd.choice([-1, 1])  # Randomly choose the drift direction
    y_points = np.array((0, 15, 31, 47, 63, 79)) # Y-points for the 6 cadences

    # Calculate max slope based on the starting frequency
    if RandMultiplier == -1:
        MaxSlope = -(96 / start) * (18.25361108/2.7939677238464355)
    else:
        spaceToEnd = bin_width - start
        MaxSlope = (96 / spaceToEnd) * (18.25361108/2.7939677238464355)

    # Generate slope and drift
    drift = (1 / MaxSlope) * rd.random()
    drift_factor = 2.7939677238464355/18.25361108

    # Calculate the x-coordinates for the 6 cadences
    x_starts = y_points * drift + start


    # Indices of the cadences where the signal will be injected
    indexes = [0, 2, 4]  # Inject signals into these cadences
    intensity = cadence[0].get_intensity(snr=60)  # Signal intensity based on the first cadence

    # Add the signal to the specified cadences
    for i in indexes:
        cadence[i].add_signal(
            stg.constant_path(
                f_start=cadence[i].get_frequency(index=int(x_starts[i])),
                drift_rate=drift * drift_factor * u.Hz / u.s,
            ),
            stg.constant_t_profile(level=intensity),
            stg.box_f_profile(width=80 * cadence[i].df * u.Hz),
            stg.constant_bp_profile(level=1),
        )

    # Add a false signal to the remaining cadences if true_or_false is False
    if not true_or_false:
        false_indexes = [1, 3, 5]  # Inject false signals into these cadences
        for i in false_indexes:
            cadence[i].add_signal(
                stg.constant_path(
                    f_start=cadence[i].get_frequency(index=int(x_starts[i])),
                    drift_rate=drift * drift_factor * u.Hz / u.s,
                ),
                stg.constant_t_profile(level=intensity),
                stg.box_f_profile(width=80 * cadence[i].df * u.Hz),
                stg.constant_bp_profile(level=1),
            )

    elif true_or_false != True:
        raise ValueError("true_or_false must be either True or False")

    return cadence








# @TimeMeasure
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
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to track progress
        cadences = list(
            tqdm(
                executor.map(process_into_cadence, data, true_false_index_dictionary),
                total=data.shape[0],  # Total number of cadences to process
                desc="Generating Frames",
            )
        )

    return np.array(cadences)

# @TimeMeasure
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

def chunk_and_inject(memmap_file, signal_split, true_false_split, signal_params, data_shape, num_workers=20, chunk_size=10000, return_data=True, start_index=0):
    """
    Process a memory-mapped file in chunks, inject signals into the data, and write the processed chunks back to the file.

    Parameters:
    - memmap_file: Path to the memory-mapped file. -> shape: (N, 6, 16, 4096)
    - signal_split: Dictionary specifying the signal type split (e.g., {"Background": 0.2, "Linear": 0.8}).
    - true_false_split: Dictionary specifying the True/False split (e.g., {"True": 0.3, "False": 0.7}).
    - signal_params: Parameters for the signal injection.
    - data_shape: Shape of the data in the memory-mapped file.
    - num_workers: Number of workers for parallel processing.
    - chunk_size: Number of samples per chunk.
    - return_data: Whether to return the memory-mapped data after processing.
    - start_index: The starting index from which to process the data.

    Returns:
    - The memory-mapped data if `return_data` is True, otherwise None.
    - Metadata array containing information about the injections.
    shape: (N, 6, 16, 4096) where N is the number of cadences

    Also saves the metadata as a NumPy array in the same directory as the memmap file.
    And creates a copy of the original memmap file to preserve it before processing.

    """
    # Create a copy of the file before modifying
    processed_path = memmap_file.replace("seperated_raw_data", "seperated_processed_data")
    print(f"Copying {memmap_file} to {processed_path} to preserve the original.")

    # Do the copy before opening any memmap to prevent accidental write
    with open(memmap_file, 'rb') as fsrc, open(processed_path, 'wb') as fdst:
        fdst.write(fsrc.read())

    # Open the memmap file in read-write mode
    data = np.memmap(processed_path, dtype='float32', mode='r+', shape=data_shape)

    # Adjust the data slice and shape for the start_index
    total_samples = data_shape[0] - start_index  # Total remaining samples from the start_index
    num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceil division for the adjusted range
    all_metadata = []  # List to store metadata for all chunks
    for chunk_idx in range(num_chunks):
        # Determine the start and end indices for this chunk
        start_idx = start_index + chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, data_shape[0])  # Ensure we don't go beyond the end

        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}: indices {start_idx} to {end_idx}")

        # Slice the data for this chunk
        chunk_data = data[start_idx:end_idx]

        # Process the chunk with inject_signals
        processed_chunk, chunk_metadata = inject_signals(chunk_data, signal_split, true_false_split, signal_params, num_workers=num_workers)

        # Append the metadata for this chunk
        all_metadata.extend(chunk_metadata)

        # Write the processed chunk back to the memmap file
        data[start_idx:end_idx] = processed_chunk

        # Ensure data is flushed back to disk
        data.flush()
        print(f"Chunk {chunk_idx + 1} flushed to disk.")

    
    # Sort by index
    all_metadata.sort(key=lambda x: x[0])

    # Strip out index for saving as pure metadata array
    stripped_metadata = [entry[1:] for entry in all_metadata]

    # Convert to NumPy array
    metadata_array = np.array(stripped_metadata, dtype=object)

    # Save it
    np.save(os.path.join(os.path.dirname(memmap_file), "injection_metadata.npy"), metadata_array)

    if return_data:
        return data, metadata_array



def build_injection_metadata(true_false_dictionary: dict, 
                             injection_type: str, 
                             total_cadences: int, 
                             indexes_used: np.ndarray) -> list:
    """
    Build metadata for the injection based on the true/false dictionary and injection type.
    Parameters:
    - true_false_dictionary: Dictionary containing True/False indices.
    - injection_type: Type of injection (e.g., "Background", "Linear").
    - total_cadences: Total number of cadences in the data.
    - indexes_used: Array of indices used for the current injection.
    Returns:
    - List of tuples containing metadata for each cadence.

    """
    metadata = []

    if injection_type == "Background":
        for i in range(total_cadences):
            metadata.append((indexes_used[i], ("Background", False), [False] * 6))
        return metadata

    true_indices = true_false_dictionary.get("True", [])
    false_indices = true_false_dictionary.get("False", [])

    for i in range(total_cadences):
        abs_index = indexes_used[i]
        if i in true_indices:
            metadata.append((abs_index, (injection_type, True), [True if j % 2 == 0 else False for j in range(6)]))
        elif i in false_indices:
            metadata.append((abs_index, (injection_type, False), [True] * 6))
        else:
            raise ValueError(f"Index {i} not found in either true or false injection dictionary.")
    return metadata




            
if __name__ == "__main__":
    signal_split = {"Background": 0.2, "Linear": 0.8}
    # number_slides = 100
    # output_dictionary = generate_injection_list(signal_split, number_slides)

    true_false_split = {"True": 0.3, "False": 0.7}
    # output_dictionary2 = generate_injection_list(true_false_split, number_slides)
    # mask = generate_injection_list(true_false_split, number_slides)


    import os 
    data_shape = np.load('Data/HIP13402-2/shape.npy')
    data_shape = tuple(int(dim) for dim in data_shape)
    file_name = 'Data/HIP13402-2/seperated_raw_data.npy'
    data = np.memmap(file_name, dtype='float32', mode='r+', shape=data_shape)
    # data2 = inject_signals(data[10000:12000], signal_split, true_false_split, np.array([1000, 0, 10000.0]), num_workers=20)
    print(f"Data shape: {data.shape}")
    data2, meta_data = chunk_and_inject(file_name, signal_split, true_false_split, np.array([1000, 0, 10000.0]), data_shape, num_workers=20, chunk_size=10000, start_index = 0)

    for i in range(0, 100, 10):
        # Unpack metadata
        injection_type, frame_flags = meta_data[i]

        # Create a figure with 6 subplots
        fig, axs = plt.subplots(6, 1, figsize=(8, 10))
        for j in range(6):
            axs[j].imshow(data2[i, j, :, :], aspect='auto')
            axs[j].set_ylabel(f"Cadence {j}", fontsize=8)

        # Add a single title to the whole figure
        title_text = f"Index {i} — Type: {injection_type[0]}, True: {injection_type[1]}, Frame flags: {frame_flags}"
        fig.suptitle(title_text, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit suptitle
        plt.savefig(f"2test{i}.png")
        plt.close()

    
