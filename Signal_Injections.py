import numpy as np
import random as rd
import setigen as stg
from numba import njit, prange
from Decorators import TimeMeasure
from astropy import units as u
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from Injection_Flavours import add_linear, add_sinusoid, add_welsh_dragon
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm import tqdm
import pickle
import os


# @TimeMeasure
def generate_injection_list(
    split: dict, number_slides: int, unequal_split_index: int = 0
):
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
        number_each_injection[unequal_split_index] += number_slides - np.sum(
            number_each_injection
        )

    # Generate the injection list
    possible_indexes = np.arange(number_slides)

    output_dictionary = {}

    # Loop through the injections
    for i, number in enumerate(number_each_injection):

        # Select the indexes for the current injection
        selected_numbers = np.random.choice(
            possible_indexes, size=number, replace=False
        )

        # Add the selected indexes to the output dictionary
        output_dictionary[keys[i]] = selected_numbers

        # Remove the selected indexes from the possible indexes
        possible_indexes = np.setdiff1d(possible_indexes, selected_numbers)

    return output_dictionary


@TimeMeasure
def inject_signals(
    data: np.ndarray,
    signal_only: np.ndarray,
    signal_split: dict,
    true_false_split: dict,
    signal_params: np.ndarray,
    loading_bar_bool: bool = True,
    num_workers: int = 20,
):
    """
    Function to inject signals into the data.
    Parameters:
    - data: Input data array with shape (N, 6, 16, 4096).
    - signal_only: Input data array for the signals only with shape (N, 6, 16, 4096).
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

    # ✅ Check for duplicated indices across injection types
    all_indices = []
    for key in keys:
        all_indices.extend(signal_index_dictionary[key])
    unique_indices = np.unique(all_indices)

    if len(all_indices) != len(unique_indices):
        from collections import Counter

        dupes = [item for item, count in Counter(all_indices).items() if count > 1]
        raise ValueError(
            f"❌ Duplicate indices found across injection types: {dupes[:10]}... (total {len(dupes)})"
        )
    else:
        print("✅ No duplicated indices across injection types.")

    true_false_dictionary = {}
    metadata = []

    # Loop through the injections
    for key in keys:
        indexes = signal_index_dictionary[key]

        # add the injection type to the data
        data[indexes, :, :, :], dict_to_append, signal_only[indexes, :, :, :] = (
            add_injection_type(
                data[indexes, :, :, :],
                signal_only[indexes, :, :, :],
                signal_params,
                injection_type=key,
                true_false_split=true_false_split,
                loading_bar_bool=loading_bar_bool,
                num_workers=num_workers,
            )
        )

        # Update the true_false_dictionary with the new injections
        true_false_dictionary = update_dictionary(true_false_dictionary, dict_to_append)

        # build metadata for the injection
        metadata += build_injection_metadata(
            true_false_dictionary=dict_to_append,
            injection_type=key,
            total_cadences=len(indexes),
            indexes_used=indexes,
        )

    data = threshold_and_normalise_data(data, 2)
    signal_only = threshold_and_normalise_data(signal_only, 2)

    return data, signal_only, metadata


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
    for key, value in dict_to_append.items():
        # -------- ensure we always work with Python lists ----------
        if isinstance(value, np.ndarray):  # convert incoming ndarray → list
            value = value.tolist()
        # -----------------------------------------------------------
        if key in dictionary_to_update:
            if isinstance(
                dictionary_to_update[key], np.ndarray
            ):  # convert stored ndarray → list
                dictionary_to_update[key] = dictionary_to_update[key].tolist()
            dictionary_to_update[key].extend(value)
        else:
            dictionary_to_update[key] = value  # already a list
    return dictionary_to_update


# @njit(parallel=True)
@TimeMeasure
# @njit(parallel=True)
# @njit(parallel=True)   # ← re-enable once you’re happy


# def threshold_and_normalise_data(data: np.ndarray, threshold_sigma: float = 5.0
#                                  ) -> np.ndarray:
#     """
#     Log-scale, row-wise (1 × 4096) sigma-clip and binarise the input.

#     Parameters
#     ----------
#     data : np.ndarray
#         Shape (N, 6, 16, 4096) – already loaded in RAM or from a memmap.
#     threshold_sigma : float
#         Threshold in units of σ above the per-row mean.

#     Returns
#     -------
#     np.ndarray
#         Binarised array of identical shape (dtype = uint8).
#     """
#     # 1. Log-scale (avoid log(0) issues if necessary beforehand)
#     # data = np.log(data, where=data > 0, out=np.full_like(data, -np.inf))

#     data = np.log10(data, where=data > 0, out=np.full_like(data, -np.inf))
#     """
#     # 2. Row-wise μ and σ  →  keep last axis only (length 4096)
#     #    Resulting shapes: (N, 6, 16, 1)
#     mean = np.mean(data, axis=-1, keepdims=True)
#     std  = np.std(data, axis=-1, keepdims=True)
#     # print(f"mean: {mean.shape}, std: {std.shape}, data: {data.shape}")
#     # 3. Threshold per row
#     mask = data > mean + threshold_sigma * std

#     # 4. Return binary uint8 (saves 4× memory compared with float32)
#     return mask.astype(np.uint8)

#     """
#     """
#     # 2. (Optional) you no longer need mean/std for normalization, so you can skip that.

#     # 3. Row-wise max → keep last axis
#     #    If data shape is (N, 6, 16, 4096), max_per_row has shape (N, 6, 16, 1)
#     max_per_row = np.max(data, axis=-1, keepdims=True)

#     # 4. Divide by row-wise max, avoiding div by zero
#     normalized = np.divide(
#         data,
#         max_per_row,
#         out=np.zeros_like(data, dtype=np.float32),
#         where=max_per_row != 0
#     )

#     # 5. Return as float32 (values in [0,1])
#     return normalized.astype(np.float32)
#     """


def threshold_and_normalise_data(
    data: np.ndarray, threshold_sigma: float = 5.0
) -> np.ndarray:
    """
    Per-frame normalization: for each (16 × 4096) frame, divide by that
    frame's own maximum value.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, 6, 16, 4096).
    threshold_sigma : float
        Unused (kept for API compatibility).

    Returns
    -------
    np.ndarray
        Normalized array of identical shape (dtype = float32), where each
        [n, s, :, :] has max 1. Frames whose max <= 0 are returned as zeros.
    """
    # Max over the last two axes → shape (N, 6, 1, 1)
    frame_max = np.max(data, axis=(-2, -1), keepdims=True)

    # Safe divide (avoid div-by-zero or non-finite max)
    out = np.zeros_like(data, dtype=np.float32)
    valid = np.isfinite(frame_max) & (frame_max > 0)
    np.divide(data, frame_max, out=out, where=valid)

    return out


# @TimeMeasure
def add_injection_type(
    data: np.ndarray,
    signal_only: np.ndarray,
    signal_params: np.ndarray,
    injection_type: str,
    true_false_split: dict,
    loading_bar_bool: bool = True,
    num_workers: int = 20,
    bin_width=4096,
):
    """
    Add a specific injection type to the data.

    Parameters:
    - data: Input data array.
    - signal_only: Input data array for the signals only.
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
        return data, true_false_index_dictionary, signal_only  # No injection needed

    # Generate true/false index dictionary
    true_false_index_dictionary = generate_injection_list(
        true_false_split, data.shape[0], unequal_split_index=1
    )

    # Generate frames for each cadence
    cadences = generate_frames(
        data, repeat(true_false_index_dictionary), max_workers=num_workers
    )

    # Go through and add the signals to the cadences

    # Add the linear lines
    if injection_type == "Linear":
        add_signal_with_threads(
            cadences,
            signal_only,
            injection_type,
            add_linear,
            true_false_index_dictionary,
            signal_params,
            bin_width=bin_width,
        )

    elif injection_type == "Sinusoid":
        add_signal_with_threads(
            cadences,
            signal_only,
            injection_type,
            add_sinusoid,
            true_false_index_dictionary,
            signal_params,
            bin_width=bin_width,
        )

    elif injection_type == "Welsh_dragon":
        add_signal_with_threads(
            cadences,
            signal_only,
            injection_type,
            add_welsh_dragon,
            true_false_index_dictionary,
            signal_params,
            bin_width=bin_width,
        )
        # for i in range(12):
        #     plt.imshow(cadences[i][0].get_data(), aspect='auto')
        #     plt.savefig(f"test{i}.png")

    data = return_to_data(cadences)
    return data, true_false_index_dictionary, signal_only


def add_signal_with_threads(
    cadences,
    signal_only,
    class_of_injection,
    injection_function,
    true_false_index_dictionary,
    signal_params,
    bin_width=4096,
):
    """
    Run the injection_function on the specified cadences in parallel and
    update BOTH 'cadences' and the matching rows of 'signal_only'.
    """

    def _run(indices, tf_flag, desc):
        # indices may be a numpy array; handle empty safely
        if indices is None:
            return
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return

        # Build per-cadence (cadence_obj, signal_only_slice) pairs
        pairs = [(cadences[i], signal_only[i]) for i in idx]

        def _apply(pair):
            cad_i, sig_i = pair
            # Injection_Flavours functions return (cadence_out, signal_only_out)
            return injection_function(cad_i, sig_i, signal_params, tf_flag, bin_width)

        with ThreadPoolExecutor() as ex:
            results = list(tqdm(ex.map(_apply, pairs), total=len(idx), desc=desc))

        # Unpack and write back to the same rows
        cad_out = [r[0] for r in results]  # list of stg.OrderedCadence
        sig_out = np.stack([r[1] for r in results], 0)  # (len(idx), 6, 16, 4096)

        cadences[idx] = np.array(cad_out, dtype=object)
        signal_only[idx] = sig_out

    _run(
        true_false_index_dictionary.get("True", []),
        True,
        f"Injecting {class_of_injection} (True)",
    )
    _run(
        true_false_index_dictionary.get("False", []),
        False,
        f"Injecting {class_of_injection} (False)",
    )


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
    cadence_index = np.where(data_slice == data_slice)[0][
        0
    ]  # Assuming slice index maps to global index
    if cadence_index in true_false_index_dictionary["True"]:
        order = "ABACAD"
    elif cadence_index in true_false_index_dictionary["False"]:
        order = "AAAAAA"
    else:
        raise ValueError(
            f"Cadence index {cadence_index} not found in True/False index dictionary."
        )

    # Create the frames
    frame_array = np.empty(data_slice.shape[0], dtype=stg.Frame)
    for j in range(data_slice.shape[0]):  # Loop over frames in the cadence
        frame = stg.Frame.from_data(
            df=2.7939677238464355 * u.Hz,
            dt=18.25361108 * u.s,
            fch1=0 * u.MHz,
            ascending=True,
            data=data_slice[j],
        )
        frame_array[j] = frame

    # Create an OrderedCadence for this set of frames
    return stg.OrderedCadence(frame_array, order=order)


def create_signal_only_file(signals_only_path, data_shape):

    os.makedirs(os.path.dirname(signals_only_path), exist_ok=True)

    signals_only = np.memmap(
        signals_only_path, dtype="float32", mode="w+", shape=data_shape
    )
    # ---- Zero-initialize signals_only in chunks (≤ ~20 GB resident) ----
    dtype = np.dtype("float32")
    bytes_per_item = dtype.itemsize  # 4
    items_per_sample = int(
        np.prod(data_shape[1:], dtype=np.int64)
    )  # 6*16*4096 = 393,216
    bytes_per_sample = items_per_sample * bytes_per_item  # ~1.5 MB per sample

    MAX_BYTES = 5 * 1024**3  # 30 GiB ceiling
    chunk_N = max(1, int(MAX_BYTES // bytes_per_sample))  # samples per chunk

    # (Optional) be a bit conservative if you want headroom:
    # chunk_N = max(1, chunk_N // 2)
    print("Creating and zeroing Signal Memmap")
    for start in tqdm(range(0, data_shape[0], chunk_N)):
        end = min(start + chunk_N, data_shape[0])
        signals_only[start:end] = 0.0  # scalar broadcast: no large temp allocation
        signals_only.flush()  # ensure written to disk per chunk
    return signals_only


def chunk_and_inject(
    memmap_file,
    signal_split,
    true_false_split,
    signal_params,
    data_shape,
    num_workers=20,
    chunk_size=10000,
    return_data=True,
    start_index=0,
):
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
    processed_path = memmap_file.replace("raw_data", "processed_data")
    print(f"Copying {memmap_file} to {processed_path} to preserve the original.")

    # 20 GiB buffer
    copy_chunk_size = 30 * 1024**3  # bytes

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    with open(memmap_file, "rb") as fsrc, open(processed_path, "wb") as fdst:
        while True:
            buf = fsrc.read(copy_chunk_size)
            if not buf:
                break
            fdst.write(buf)
            # free the buffer right away
            del buf

    print(f"Copy completed.")

    # Open the memmap file in read-write mode
    data = np.memmap(processed_path, dtype="float32", mode="r+", shape=data_shape)

    signal_only_path = processed_path.replace(".npy", "_signals_only.npy")

    signals_only = create_signal_only_file(signal_only_path, data_shape)

    print("zeroeing complete now chunking")
    # Adjust the data slice and shape for the start_index
    total_samples = (
        data_shape[0] - start_index
    )  # Total remaining samples from the start_index
    num_chunks = (
        total_samples + chunk_size - 1
    ) // chunk_size  # Ceil division for the adjusted range
    all_metadata = []  # List to store metadata for all chunks
    count = 0
    for chunk_idx in range(num_chunks):
        # Determine the start and end indices for this chunk
        start_idx = start_index + chunk_idx * chunk_size
        end_idx = min(
            start_idx + chunk_size, data_shape[0]
        )  # Ensure we don't go beyond the end

        print(
            f"Processing chunk {chunk_idx + 1}/{num_chunks}: indices {start_idx} to {end_idx}"
        )

        # Slice the data for this chunk
        chunk_data = data[start_idx:end_idx]
        chunk_signals = signals_only[start_idx:end_idx]

        # Process the chunk with inject_signals
        processed_chunk, processed_signals_chunk, chunk_metadata = inject_signals(
            chunk_data,
            chunk_signals,
            signal_split,
            true_false_split,
            signal_params,
            num_workers=num_workers,
        )
        chunk_metadata.sort(key=lambda t: int(t[0]))
        chunk_metadata = [(info, flags) for (_, info, flags) in chunk_metadata]
        # print(f"{chunk_metadata=}")
        # Append the metadata for this chunk
        all_metadata.extend(chunk_metadata)

        # Write the processed chunk back to the memmap file
        data[start_idx:end_idx] = processed_chunk

        # Ensure data is flushed back to disk
        data.flush()
        signals_only[start_idx:end_idx] = processed_signals_chunk
        signals_only.flush()

        print(f"Chunk {chunk_idx + 1} flushed to disk.")

        count += 1

    # Convert to NumPy array
    metadata_array = np.array(all_metadata, dtype=object)

    # Save it
    base = os.path.splitext(os.path.basename(processed_path))[0]
    metadata_filename = f"{base}_injection_metadata.npy"
    metadata_path = os.path.join(os.path.dirname(processed_path), metadata_filename)
    np.save(metadata_path, metadata_array)

    shape_save_loc = processed_path.replace(".npy", "_shape.npy")
    np.save(shape_save_loc, data_shape)

    if return_data:
        return data, signals_only, metadata_array


def build_injection_metadata(
    true_false_dictionary: dict,
    injection_type: str,
    total_cadences: int,
    indexes_used: np.ndarray,
) -> list:
    """
    Build metadata for the injection based on the true/false dictionary and injection type.
    Parameters:
    - true_false_dictionary: Dictionary containing True/False indices.
    - injection_type: Type of injection (e.g., "Background", "Linear").
    - total_cadences: Total number of cadences in the injection type.
    - indexes_used: Array of indices used for the current injection.
    Returns:
    - List of tuples containing metadata for each cadence.

    """
    metadata = []
    # print(f"{total_cadences=}, {injection_type=}, {indexes_used=}")
    if injection_type == "Background":
        for i in range(total_cadences):
            metadata.append(
                (indexes_used[i], ("Background", False), ["Background"] * 6)
            )
        return metadata

    true_indices = true_false_dictionary.get("True", [])
    false_indices = true_false_dictionary.get("False", [])

    for i in range(total_cadences):
        abs_index = indexes_used[i]
        if i in true_indices:
            metadata.append(
                (
                    abs_index,
                    (injection_type, True),
                    [injection_type if j % 2 == 0 else "Background" for j in range(6)],
                )
            )
        elif i in false_indices:
            metadata.append((abs_index, (injection_type, False), [injection_type] * 6))
        else:
            raise ValueError(
                f"Index {i} not found in either true or false injection dictionary."
            )
    return metadata


if __name__ == "__main__":
    # signal_split = {"Background": 0.125, "Linear": 0.4435, "Sinusoid": 0.4435, "Welsh_dragon": 0.005}
    signal_split = {
        "Background": 0.12,
        "Linear": 0.8,
        "Sinusoid": 0.04,
        "Welsh_dragon": 0.04,
    }
    # signal_split = {"Background": 0.2, "Linear": 0.4, "Sinusoid": 0.38, "Welsh_dragon": 0.02}
    # signal_split = {"Background": 0.2, "Linear": 0.8}
    # signal_split = {"Welsh_dragon": 1}
    # signal_split = {"Sinusoid": 1}
    # number_slides = 100
    # output_dictionary = generate_injection_list(signal_split, number_slides)

    # true_false_split = {"True": 0.46, "False": 0.54}
    true_false_split = {"True": 0.46, "False": 0.54}

    # output_dictionary2 = generate_injection_list(true_false_split, number_slides)
    # mask = generate_injection_list(true_false_split, number_slides)

    import os

    file_name = "generated_data/background_seperated_raw_data_100k.npy"
    data_shape_name = file_name.replace(".npy", "_shape.npy")
    data_shape = np.load(data_shape_name)
    data_shape = tuple(int(dim) for dim in data_shape)

    data = np.memmap(file_name, dtype="float32", mode="r+", shape=data_shape)
    # for i in range(0,40, 5):

    #     plt.imshow(data[i, 0, :, :], aspect='auto')
    #     plt.show()
    #     plt.savefig(f"test{i}.png")
    # data2 = inject_signals(data[10000:12000], signal_split, true_false_split, np.array([1000, 0, 10000.0]), num_workers=20)
    print(f"Data shape: {data.shape}")
    full_data, injections_only, meta_data = chunk_and_inject(
        file_name,
        signal_split,
        true_false_split,
        np.array([10, 0, 10.0]),
        data_shape,
        num_workers=20,
        chunk_size=10000,
        start_index=0,
    )

    for i in tqdm(range(0, 40, 1)):
        injection_type, frame_flags = meta_data[i]

        fig, axs = plt.subplots(6, 2, figsize=(8, 10), constrained_layout=True)

        for j in range(6):
            im0 = axs[j, 0].imshow(full_data[i, j, :, :], aspect="auto", cmap="viridis")
            axs[j, 0].set_ylabel(f"Cadence {j}", fontsize=8)
            fig.colorbar(im0, ax=axs[j, 0], fraction=0.046, pad=0.02)

            im1 = axs[j, 1].imshow(
                injections_only[i, j, :, :], aspect="auto", cmap="viridis"
            )
            fig.colorbar(im1, ax=axs[j, 1], fraction=0.046, pad=0.02)

        title_text = f"Index {i} — Flavour: {injection_type[0]}, Signal Type: {injection_type[1]}, Frame flags: {frame_flags}"
        fig.suptitle(title_text, fontsize=10)

        plt.savefig(
            f"Output_plots/final_test{i}.png"
        )  # optionally add bbox_inches='tight'
        plt.close()
