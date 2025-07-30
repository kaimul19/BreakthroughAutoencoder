import os
import numpy as np

# Load the full original memmap
data_shape = np.load('Data/HIP13402-02/shape.npy')
data_shape = tuple(int(dim) for dim in data_shape)
file_name = 'Data/HIP13402-02/seperated_raw_data.npy'
data = np.memmap(file_name, dtype='float32', mode='r+', shape=data_shape)

# Slice the last 2000 entries
data_subset = data[0:40000].copy()

# Path to overwrite the original memmap file
output_path = 'Data/HIP13402-02/first_40000_entries_memmap_seperated_raw_data.npy'  # or whatever the original file is

# Overwrite the file with new memmap content
memmap_subset = np.memmap(output_path, dtype='float32', mode='w+', shape=data_subset.shape)
memmap_subset[:] = data_subset[:]
memmap_subset.flush()

shape_path = 'Data/HIP13402-02/first_40000_shape.npy'
np.save(shape_path, np.array(data_subset.shape, dtype=np.int64))
