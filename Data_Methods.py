import numpy as np
import blimpy as bl
import matplotlib.pyplot as plt
import os
import sys
import gc
from tqdm import tqdm
from numba import njit, prange
import time

# Numba-compatible loop function
@njit(parallel=True)
def run_loop(data, output_array, number_bins, bin_length=4096):
    for j in prange(number_bins):  # Use prange for parallelization
        output_array[j, 0, :, :] = data[:, j * bin_length : (j + 1) * bin_length]

def data_parsing(file_path, bin_length = 4096, file_name = None, loading_bar_visibie = True):
    import gc
    # Load the initial file
    print(f"Loading first wf file, takes ≈ 20 seconds per")
    wf = bl.Waterfall(file_path[0])
    print(f"Loaded first wf file")

    # Remove end bit so that we dont have any fractionally filled slides
    trim_length = wf.data.shape[2] - (wf.data.shape[2] % bin_length)

    # Calculate N (number of bins of length 4096)
    number_bins = trim_length // bin_length

    # Create final array
    if file_name is None:
        import time, os, glob
        os.makedirs(f"Data/{time.strftime('%d-%m-%Y %H:%M')}", exist_ok=True)
        data_file_name = f"Data/{time.strftime('%d-%m-%Y %H:%M')}/seperated_raw_data.npy"
        freq_file_name = f"Data/{time.strftime('%d-%m-%Y %H:%M')}/seperated_freqs.npy"
        print(f"File name not provided, saving as {data_file_name}")

    final_array = np.memmap(data_file_name, dtype='float32', mode='w+', shape=(number_bins, 6, 16, bin_length))

    # Get frequencies
    print("Getting frequencies")
    final_freq = np.zeros((number_bins, bin_length))
    freq = np.flip(wf.get_freqs())
    for i in tqdm(range(number_bins), desc = ""):
        final_freq[i] = freq[i*bin_length:(i+1)*bin_length]
    np.save(freq_file_name, final_freq)
    shape = (number_bins, 6, 16, bin_length)
    np.save(f"Data/{time.strftime('%d-%m-%Y %H:%M')}/shape.npy", shape)


    # Loop through each file
    print("Starting to loop through files")
    # start the initial progress bar
    if loading_bar_visibie:
        bar1 = tqdm(total=len(file_path), desc="File Progress", position=0)

    for i in range(len(file_path)):
        # Update the progress bar and start the slide progress bar
        if loading_bar_visibie:
            bar1.update(1)
            bar2 = tqdm(total=number_bins, desc=f"Slide Progress for file {i}", position=1)

        # Have already loaded the first file
        if i != 0:
            wf = bl.Waterfall(file_path[i])
        data = np.flip(wf.data[:, 0, :], axis=-1)  # flip due to how data is stored
        data = data[:, :trim_length]  # trim the data to be a multiple of bin_length
        del wf  # delete the waterfall object for data efficiency
        gc.collect()
        time_start = time.time()
        run_loop(data, final_array, number_bins, bin_length)
        time_end = time.time()
        print(f"Time taken: {time_end-time_start}")
        if loading_bar_visibie:
            bar2.close()
        final_array.flush()

    if loading_bar_visibie:
        bar1.close()


    return final_array, freq, data_file_name, freq_file_name



if __name__ == "__main__":
    path = ['/datag/pipeline/AGBT16A_999_212/holding/spliced_blc0001020304050607_guppi_57541_59531_HIP13402_0014.gpuspec.0000.h5',
            '/datag/pipeline/AGBT16A_999_212/holding/spliced_blc0001020304050607_guppi_57541_59890_HIP12402_0015.gpuspec.0000.h5',
            '/datag/pipeline/AGBT16A_999_212/holding/spliced_blc0001020304050607_guppi_57541_60248_HIP13402_0016.gpuspec.0000.h5',
            '/datag/pipeline/AGBT16A_999_212/holding/spliced_blc0001020304050607_guppi_57541_60602_HIP12482_0017.gpuspec.0000.h5',
            '/datag/pipeline/AGBT16A_999_212/holding/spliced_blc0001020304050607_guppi_57541_60955_HIP13402_0018.gpuspec.0000.h5',
            '/datag/pipeline/AGBT16A_999_212/holding/spliced_blc0001020304050607_guppi_57541_61313_HIP12549_0019.gpuspec.0000.h5']

    data, freq, data_file_name, freq_file_name = data_parsing(path, bin_length = 4096, loading_bar_visibie = True)