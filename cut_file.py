import os
import numpy as np

# Load the full original memmap
data_shape = np.load('generated_data/background_data_shape.npy')
data_shape = tuple(int(dim) for dim in data_shape)
file_name = 'generated_data/background_data.npy'
data = np.memmap(file_name, dtype='float32', mode='r+', shape=data_shape)

# Slice the last 2000 entries
data_subset = data[0:2000].copy()

# Path to overwrite the original memmap file
output_path = 'generated_data/first_2000_entries_memmap_seperated_raw_data.npy'  # or whatever the original file is

# Overwrite the file with new memmap content
memmap_subset = np.memmap(output_path, dtype='float32', mode='w+', shape=data_subset.shape)
memmap_subset[:] = data_subset[:]
memmap_subset.flush()

shape_path = 'generated_data/first_2000_shape.npy'
np.save(shape_path, np.array(data_subset.shape, dtype=np.int64))
