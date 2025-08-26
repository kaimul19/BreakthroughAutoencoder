import os
import gc
import numpy as np
import setigen as stg
from astropy import units as u
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# frame constants
freq_increment          = 2.7939677238464355 * u.Hz
time_sample_interval    = 18.25361108        * u.s
first_channel_frequency = 0                  * u.MHz

@lru_cache(maxsize=None)
def get_cached_frame(number_of_frequency_bins: int,
                     total_time_channels: int) -> stg.Frame:
    """
    Cache one Frame instance per (frequency_bins, time_channels) tuple,
    per process.
    """
    return stg.Frame(
        fchans=number_of_frequency_bins,
        tchans=total_time_channels,
        df=freq_increment,
        dt=time_sample_interval,
        fch1=first_channel_frequency
    )

def generate_single_background_cadence(number_of_observations: int,
                                      number_of_time_bins: int,
                                      number_of_frequency_bins: int) -> np.ndarray:
    """
    Draw ONE noise frame of length (observations × time_bins),
    reshape to (observations, time_bins, frequency_bins), and return.
    """
    frame = get_cached_frame(
        number_of_frequency_bins,
        number_of_observations * number_of_time_bins
    )
    noise_array = frame.add_noise_from_obs()  # shape (obs*time_bins, freq_bins)
    cadence_array = noise_array.reshape(
        number_of_observations,
        number_of_time_bins,
        number_of_frequency_bins
    )
    return cadence_array.astype(np.float32)

def _make_cadence(args):
    """
    Wrapper for picklable ProcessPoolExecutor tasks.
    """
    return generate_single_background_cadence(*args)

def generate_background_chunk(samples_per_chunk: int,
                              number_of_observations: int,
                              number_of_time_bins: int,
                              number_of_frequency_bins: int,
                              number_of_workers: int = 20) -> np.ndarray:
    """
    Generate one chunk of cadences in parallel, with a smooth tqdm bar
    that updates on each cadence completion via as_completed().
    Returns an array of shape
      (samples_per_chunk,
       number_of_observations,
       number_of_time_bins,
       number_of_frequency_bins).
    """
    # prepare the args for each cadence
    argument_list = [
        (number_of_observations, number_of_time_bins, number_of_frequency_bins)
    ] * samples_per_chunk

    # submit all tasks at once
    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        future_to_index = {
            executor.submit(_make_cadence, arg): idx
            for idx, arg in enumerate(argument_list)
        }

        # collect results as they finish
        results = [None] * samples_per_chunk
        cadence_bar = tqdm(
            total=samples_per_chunk,
            desc="  Generating cadences",
            unit="cadence",
            position=1,
            leave=False
        )

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            cadence_bar.update(1)

        cadence_bar.close()

    # stack into one ndarray
    chunk_to_be_saved = np.stack(results, axis=0)
    return chunk_to_be_saved

def chunk_and_generate_background(memory_map_file_path: str,
                                  data_dimensions: tuple,
                                  samples_per_chunk: int = 10_000,
                                  number_of_workers: int = 20,
                                  return_memory_map: bool = True,
                                  first_sample_index: int = 0):
    """
    Write into a memory‑mapped file in chunks, flushing each chunk to disk
    and freeing its RAM immediately.
    """
    os.makedirs(os.path.dirname(memory_map_file_path), exist_ok=True)

    memory_map_data = np.memmap(
        memory_map_file_path,
        dtype='float32',
        mode='w+',
        shape=data_dimensions
    )
    # save dimensions for downstream readers
    np.save(memory_map_file_path.replace('.npy', '_shape.npy'), data_dimensions)

    total_samples, number_of_observations, number_of_time_bins, number_of_frequency_bins = data_dimensions
    number_of_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk

    # bytes per single‐cadence slice (to flush just that slice)
    bytes_per_cadence = (
        number_of_observations
        * number_of_time_bins
        * number_of_frequency_bins
        * memory_map_data.dtype.itemsize
    )
    header_offset_bytes = memory_map_data.offset

    chunk_bar = tqdm(
        range(number_of_chunks),
        desc="Processing chunks",
        unit="chunk",
        position=0
    )

    for chunk_index in chunk_bar:
        start_index = first_sample_index + chunk_index * samples_per_chunk
        end_index   = min(start_index + samples_per_chunk, total_samples)
        actual_size = end_index - start_index

        chunk_bar.set_postfix_str(f"indices {start_index}:{end_index}")

        # generate and write this chunk
        chunk_data_to_be_saved = generate_background_chunk(
            actual_size,
            number_of_observations,
            number_of_time_bins,
            number_of_frequency_bins,
            number_of_workers=number_of_workers
        )
        memory_map_data[start_index:end_index] = chunk_data_to_be_saved

        # flush only the bytes we just wrote
        byte_start  = header_offset_bytes + start_index * bytes_per_cadence
        byte_length = actual_size * bytes_per_cadence
        memory_map_data._mmap.flush(byte_start, byte_length)

        # free the chunk's RAM immediately
        del chunk_data_to_be_saved
        gc.collect()

    if return_memory_map:
        return memory_map_data

if __name__ == "__main__":
    memory_map_output_path = "generated_data/background_seperated_raw_data_100k.npy"
    memory_map_dimensions  = (100_000, 6, 16, 4096)
    samples_per_chunk      = 10_000
    number_of_workers      = 20

    background_data = chunk_and_generate_background(
        memory_map_file_path=memory_map_output_path,
        data_dimensions=memory_map_dimensions,
        samples_per_chunk=samples_per_chunk,
        number_of_workers=number_of_workers,
        return_memory_map=True
    )

    print(f"✅ All background data generated. Final shape: {background_data.shape}")

    # plotting the first few cadences
    for sample_to_plot in range(0, 10, 2):
        figure, axes_array = plt.subplots(6, 1, figsize=(8, 10))
        for obs_idx in range(6):
            img = axes_array[obs_idx].imshow(
                background_data[sample_to_plot, obs_idx],
                aspect='auto'
            )
            axes_array[obs_idx].set_ylabel(
                f"cadence {obs_idx}", fontsize=8
            )
            plt.colorbar(img, ax=axes_array[obs_idx]).ax.tick_params(labelsize=6)
        figure.suptitle(f"Index {sample_to_plot} — Background Only", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"final_test_{sample_to_plot}.png")
        plt.close(figure)
