import os
import numpy as np

# Load the full original memmap
file_name = "generated_data/background_seperated_raw_data_100k.npy"
data_shape_name = file_name.replace(".npy", "_shape.npy")
data_shape = np.load(data_shape_name)
data_shape = tuple(int(dim) for dim in data_shape)

data = np.memmap(file_name, dtype="float32", mode="r+", shape=data_shape)
# Slice the last 2000 entries
num_data = 20000
data_subset = data[0:num_data].copy()

# Path to overwrite the original memmap file
output_path = f"generated_data/first_{str(num_data)}_raw_data.npy"  # or whatever the original file is

# Overwrite the file with new memmap content
memmap_subset = np.memmap(
    output_path, dtype="float32", mode="w+", shape=data_subset.shape
)
memmap_subset[:] = data_subset[:]
memmap_subset.flush()

shape_path = f"generated_data/first_{str(num_data)}_raw_data_shape.npy"
np.save(shape_path, np.array(data_subset.shape, dtype=np.int64))
