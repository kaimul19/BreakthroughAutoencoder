import numpy as np
import matplotlib.pyplot as plt
import numba 

@numba.jit(nopython=True) # Use Numba to speed up this function define it outside class for performance
def row_statistics(signal_snippet: np.ndarray, sigma_multiplier: float = 3.0):
    row_medians = np.nanmedian(signal_snippet, axis=1)                # compute per-row medians
    row_standard_deviations = np.nanstd(signal_snippet, axis=1)       # compute per-row standard deviations
    row_thresholds = row_medians + sigma_multiplier * row_standard_deviations  # compute thresholds
    return row_thresholds, row_medians, row_standard_deviations  # return results

class signal_data:
    def __init__(self, signal_snippet: np.ndarray, sigma_multiplier: float = 3.0):
        """
        Container for 2D signal snippets with row-based statistics and seed operations.
        """
        self.signal_snippet = np.ascontiguousarray(signal_snippet, dtype=np.float64)               # store image as float array
        if self.signal_snippet.ndim != 2:                                           # validate dimensionality
            raise ValueError("image_2d must be a 2D array")                   # raise error if not 2D
        if self.signal_snippet.shape[0] != 16:
            print("Warning: Expected 16 rows in the snippet array") 
            print("These corresponding to the 16 time bins of an observation")  # warn if unexpected shape
        self.number_of_rows, self.number_of_columns = self.signal_snippet.shape     # cache shape information
        self.row_medians = None                                               # placeholder for per-row medians
        self.row_std = None                                                   # placeholder for per-row stds
        self.row_thresholds = None                                            # placeholder for per-row thresholds
        self.initial_boolean_mask = np.empty(self.signal_snippet.shape, 
                                        dtype=bool)                           # placeholder for raw seed mask
        self.consolidated_group_boolean_mask = np.empty(self.signal_snippet.shape, 
                                        dtype=bool)                           # placeholder for consolidated mask
        self.sigma_multiplier = float(sigma_multiplier)                       # store sigma multiplier


    @numba.jit(nopython=True)
    def compute_row_statistics(self):
        """
        Compute median and standard deviation for each row in the 2D signal array.

        This method calculates the median, standard deviation, and thresholds for each row of the signal_snippet array.

        """
        self.row_thresholds, self.row_medians, self.row_std = row_statistics(self.signal_snippet, self.sigma_multiplier)
    
    def get_seeds(self):
        """
        Identify seed pixels exceeding per-row thresholds (vectorised NumPy).
        """
        if self.row_thresholds is None: # Ensure statistics are computed
            raise ValueError("Row statistics must be computed before identifying seeds.")
        
        if self.initial_boolean_mask is not None: # Check if mask already exists
            raise ValueError("Initial boolean mask already exists. You should not call this method twice.")

        # Preallocate boolean mask
        self.initial_boolean_mask = np.empty(self.signal_snippet.shape, dtype=bool)

        # Vectorised broadcast comparison, writing directly into mask (fast, no extra allocs)
        np.greater(self.signal_snippet, self.row_thresholds[:, None], out=self.initial_boolean_mask)


    def consolidate_seeds(self, max_pixel_distance_either_side = 5):
        """
        Consolidate seed pixels into groups based on adjacency.
        """
        if self.initial_boolean_mask is None:
            raise ValueError("Initial boolean mask must be computed before consolidating seeds.")
        if self.consolidated_group_boolean_mask is not None:
            raise ValueError("Consolidated group boolean mask already exists. You should not call this method twice.")
        
        


    
    def prune_functions(self, min_group_size: int = 3, gate_requirment_for_false: string = "and"):
        """
        Prune consolidated groups based on size and other criteria.
        gate_requirment_for_false options are "and", "or", "horizontal", "vertical"
        """
    
    def prune_horizontal(self, min_horizontal_size: int = 3):
        """
        Prune groups that do not meet horizontal size criteria.
        """
    
    def prune_vertical(self, min_vertical_size: int = 3):
        """
        Prune groups that do not meet vertical size criteria.
        """
    
    def grow_seeds(self, growth_threshold: float = 1.0):
        """
        Expand seed groups by including adjacent pixels that meet a lower threshold.
        """
    
    def plot_1D(self):
        """
        Plot the 1D signal data with thresholds and seed indicators.
        """
    
    def plot_2D(self):
        """
        Plot the 2D signal data with thresholds and seed indicators.
        """


