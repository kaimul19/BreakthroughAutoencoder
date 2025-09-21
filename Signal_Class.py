import numpy as np
import matplotlib.pyplot as plt
import numba 


# dont jit this function as it uses non numba functions
def _consolidate_1d(flattened_array: np.ndarray, maximum_gap: int, unique_id:np.ndarray) -> np.ndarray:
    """
    Merge 1-runs in a 1D boolean/integer mask by filling gaps of length <= maximum_gap.
    Returns a boolean 1D array with consolidated seeds.
    """
    distances = np.diff(flattened_array, prepend=0, append=0)   # compute differences to find seed boundaries
    starts = np.flatnonzero(distances == 1)                     # find start indices of seeds
    ends   = np.flatnonzero(distances == -1) - 1                # find end indices of seeds

    if starts.size == 0:                
        print(f"No Seeds Found in the snippet at index: {unique_id}")   # debug message if no seeds found
        return np.zeros_like(flattened_array, dtype=bool)

    gaps = (starts[1:] - ends[:-1]) - 1                         # compute gaps between seeds
    cuts = np.where(gaps > maximum_gap)[0]                      # identify gaps larger than maximum_gap

    merged_starts = np.r_[starts[0], starts[cuts + 1]]          # merge start indices
    merged_ends   = np.r_[ends[cuts], ends[-1]]                 # merge end indices

    difference_buffer = np.zeros(flattened_array.size + 1, dtype=np.int32)    # create difference buffer
    np.add.at(difference_buffer, merged_starts, 1)                            # mark starts
    np.add.at(difference_buffer, merged_ends + 1, -1)                         # mark ends

    return np.cumsum(difference_buffer[:-1]) > 0                     # return consolidated boolean mask






class signal_data:
    def __init__(self, signal_snippet: np.ndarray, unique_id: np.ndarray, sigma_multiplier: float = 3.0):
        """
        Container for 2D signal snippets with row-based statistics and seed operations.
        """
        self.signal_snippet = np.ascontiguousarray(signal_snippet, dtype=np.float64)               # store image as float array
        if self.signal_snippet.ndim != 2:                                           # validate dimensionality
            raise ValueError("image_2d must be a 2D array")                   # raise error if not 2D
        if self.signal_snippet.shape[0] != 16:
            print("Warning: Expected 16 rows in the snippet array") 
            print("These corresponding to the 16 time bins of an observation")  # warn if unexpected shape
        if unique_id is None or not isinstance(unique_id, np.ndarray) or len(unique_id) != 2:
            raise ValueError(f"unique_id must be an np.2darray of [observation_index, snippet_index] instead got {unique_id}, type {type(unique_id)}")
        self.number_of_rows, self.number_of_columns = self.signal_snippet.shape     # cache shape information
        self.row_medians = None                                               # placeholder for per-row medians
        self.row_std = None                                                   # placeholder for per-row stds
        self.row_thresholds = None                                            # placeholder for per-row thresholds
        self.sigma_multiplier = float(sigma_multiplier)                       # store sigma multiplier
        self.unique_id = unique_id                                           # store unique identifier for debugging
        self.initial_boolean_mask = None                                       # placeholder for initial boolean mask of seeds
        self.consolidated_group_boolean_mask = None                            # placeholder for consolidated boolean mask of seed groups
    def _return_shape(self):
        """
        Return the shape of the signal snippet.
        """
        return self.signal_snippet.shape
    
    def _return_statistics(self):
        """
        Return the computed row statistics: thresholds, medians, and standard deviations.
        """
        if self.row_thresholds is None or self.row_medians is None or self.row_std is None:
            raise ValueError("Row statistics have not been computed yet.")
        return self.row_thresholds, self.row_medians, self.row_std
    
    def _return_initial_mask(self):
        """
        Return the initial boolean mask of seed pixels.
        """
        if self.initial_boolean_mask is None:
            raise ValueError("Initial boolean mask has not been computed yet.")
        return self.initial_boolean_mask
    
    def _return_consolidated_mask(self):
        """
        Return the consolidated boolean mask of seed groups.
        """
        if self.consolidated_group_boolean_mask is None:
            raise ValueError("Consolidated group boolean mask has not been computed yet.")
        return self.consolidated_group_boolean_mask, self.consolidated_group_boolean_mask.shape
    def compute_row_statistics(self):
        """
        Compute median and standard deviation for each row in the 2D signal array.

        This method calculates the median, standard deviation, and thresholds for each row of the signal_snippet array.

        """
        self.row_medians = np.median(self.signal_snippet, axis=1)                # compute per-row medians
        self.row_std = np.std(self.signal_snippet, axis=1)       # compute per-row standard deviations
        self.row_thresholds = self.row_medians + self.sigma_multiplier * self.row_std  # compute thresholds

    
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
        
        seperation = max_pixel_distance_either_side + 1  # define seperation with additional index so that runs on edges are not merged
        work = np.zeros((self.number_of_rows, 
                            self.number_of_columns + seperation), dtype=bool) # padded work array to avoid wrap-around
        work[:, :self.number_of_columns] = self.initial_boolean_mask.astype(bool, copy=True) # populate work array
        flat = work.ravel()                                      # flatten for processing
        cons_flat = _consolidate_1d(flat, max_pixel_distance_either_side, unique_id = self.unique_id)     # call merging method
        self.consolidated_group_boolean_mask = cons_flat.reshape(self.number_of_rows, 
                                                    self.number_of_columns + seperation)[:, :self.number_of_columns] # reshape back and trim padding



    def prune_methods(self, 
        min_horizontal_size: int = 3,
        min_vertical_size: int = 1,
        drift_padding_size: int = 1,
        min_neighbours: int = 1,
        pad_mode: str = "constant",
        gate_requirment_for_false: str = "or"
    ):
        """
        Prune consolidated groups based on horizontal and vertical criteria.
        gate_requirment_for_false options are "and", "or", "horizontal", "vertical"
        """
        if self.initial_boolean_mask is None:
            raise ValueError("Initial boolean masks must be computed before pruning.")
        
        if gate_requirment_for_false not in ["and", "or", "horizontal", "vertical"]:
            raise ValueError("gate_requirment_for_false must be one of 'and', 'or', 'horizontal', or 'vertical'.")

        horizontal_pruned_mask = self._prune_horizontal(min_horizontal_size=min_horizontal_size) # first horizontal pruning

        if gate_requirment_for_false == "horizontal":       # shortcut if only horizontal pruning is needed
            final_pruned_mask = horizontal_pruned_mask
            return final_pruned_mask                        # return pruned mask immediately, so vertical pruning is skipped

        vertical_pruned_mask, _, _ = self._prune_vertical(                      # then vertical pruning
            min_vertical_size=min_vertical_size,
            drift_padding_size=drift_padding_size,
            min_neighbours=min_neighbours,
            pad_mode=pad_mode,
        )

        if gate_requirment_for_false == "vertical":         # shortcut if only vertical pruning is needed
            final_pruned_mask = vertical_pruned_mask
            return final_pruned_mask                        # return pruned mask immediately, so horizontal pruning is skipped

        if gate_requirment_for_false == "or":                                   # combine masks with logical OR
            final_pruned_mask = horizontal_pruned_mask | vertical_pruned_mask
        elif gate_requirment_for_false == "and":                                # combine masks with logical AND
            final_pruned_mask = horizontal_pruned_mask & vertical_pruned_mask
        
        return final_pruned_mask
    
    def _prune_horizontal(self, min_horizontal_size: int = 3):
        """
        Prune groups that do not meet horizontal size criteria.
        """


    def _prune_vertical(
        self,
        min_vertical_size: int = 1,            # vertical radius n: rows strictly above/below to include
        drift_padding_size: int = 1,            # horizontal radius k: columns left/right to include
        min_neighbours: int = 1,                # minimum neighbours required to keep a seed
        pad_mode: str = "constant",             # padding mode for edges when building the integral image
    ):
        """
        Keep seeds that have at least `min_neighbours` neighbours in rows strictly
        above/below (±min_vertical_size) within ±drift_padding_size columns.
        The *same row* as the seed is excluded by construction (we sum two rectangles:
        the one above and the one below the centre row).

        Parameters
        - min_vertical_size (int): Minimum vertical size of groups to retain.
        - drift_padding_size (int): Horizontal half-width for neighbour search.
        - min_neighbours (int): Minimum number of neighbours required to keep a seed.
        - pad_mode (str): Padding mode for edges, passed to np.pad.

        Returns
        - pruned_mask (np.ndarray): Boolean mask of seeds that meet neighbour criteria.
        - keep_mask (np.ndarray): Boolean mask of all pixels that meet neighbour criteria.
        - neighbour_count (np.ndarray): Integer array of neighbour counts for each pixel.
        """
        if not isinstance(min_vertical_size, int):                                 # ensure type is integer
            raise ValueError("min_vertical_size must be an integer.")              # raise on bad type
        if not isinstance(drift_padding_size, int):                                # ensure type is integer
            raise ValueError("drift_padding_size must be an integer.")             # raise on bad type
        if not isinstance(min_neighbours, int):                                    # ensure type is integer
            raise ValueError("min_neighbours must be an integer.")                 # raise on bad type
        if min_vertical_size == 0:                                                 # disallow zero vertical radius
            raise ValueError(
                "min_vertical_size must be at least 1."  
                "If you want to just use horizontal pruning, call " 
                "prune_methods with gate_requirment_for_false='horizontal'" 
            )  # guidance
        if min_neighbours < 1:                                                     # disallow non-positive threshold
            raise ValueError(
                "min_neighbours must be at least 1."
                "If you want to just use horizontal pruning,"
                "call prune_methods with gate_requirment_for_false='horizontal'")      # guidance

        # NOTE Cruicial for understanding the comments that follow:
        # n = min_vertical_size, 
        # k = drift_padding_size, 
        # R = number_of_rows, 
        # C = number_of_columns, 
        # W = window_width = 2k+1
        # r = row index of a pixel in original image
        # c = column index of a pixel in original image

        binary_int = self.initial_boolean_mask.astype(np.int32, copy=True)         # convert original seed mask to int for summations, copy to avoid modifying original
        seed_mask_boolean = self.initial_boolean_mask.astype(bool, copy=False)     # boolean version to gate results back to original seeds
        window_width = 2 * drift_padding_size + 1                                  # horizontal window width (2k+1)

        # We pad by n rows and k columns so every pixel has a full ABOVE/BELOW window even at edges.             
        padded = np.pad(                                                           # build padded image around the binary mask
            binary_int,                                                            # source integer mask (0/1)
            pad_width = (
                            (min_vertical_size, min_vertical_size),                               # pad top and bottom by n rows
                            (drift_padding_size, drift_padding_size)
                            ),                             # pad left and right by k cols
            mode=pad_mode,                                                         # padding rule (default constant zeros)
        )  

        # Build a 2D integral image (summed-area table) with an extra top row/left column of zeros.              
        # Convention: integral[r, c] = sum of padded[0:r, 0:c], i.e. a half-open box (row r and col c excluded). 
        integral = np.pad(padded, 
                        pad_width = ((1, 0), (1, 0)), 
                        mode="constant",
                        )               # prepend a zero row and zero column
        integral = integral.cumsum(axis=0).cumsum(axis=1)                          # cumulative sums down then across (2D prefix sums)

        # ----------------------------- how the rectangle sums map (read-only comments) -----------------------------
        # For each original pixel (r, c), we want two rectangles (same columns, different rows):                  
        #   ABOVE: rows [r - n .. r - 1], cols [c - k .. c + k]  
        #   (seed row excluded)                          
        #   BELOW: rows [r + 1 .. r + n], cols [c - k .. c + k]  (seed row excluded)                              
        # After padding by n rows and k columns, (r, c) in the original maps to (r+n, c+k) in the padded image.   
        # With the extra zero row/col, the sum over inclusive rectangle rows [r0..r1], cols [c0..c1] is:    
        #   sum = I[r1+1, c1+1] - I[r0, c1+1] - I[r1+1, c0] + I[r0, c0]       (four-corner inclusion–exclusion)  
        #   (NOTE: The r0/r1/c0/c1 definitions immediately below are for the ABOVE band only;
        #          see the BELOW band comment for the corresponding definitions used there.)
        #   For a pixel at (r_pixel, c_pixel) in the ABOVE band:
        #       r0 = r_pixel - n,  r1 = r_pixel - 1    # inclusive row limits strictly above the seed row
        #       c0 = c_pixel - k,  c1 = c_pixel + k    # inclusive column limits of the horizontal window
        #   (c0/c1 are inclusive; r0/r1 are inclusive; the “+1” appears only in the integral-image indices.)
        #  For the sum above sum = I[A] - I[B] - I[C] + I[D]:
        #       A = I[r1+1, c1+1]   # bottom-right corner lookup (accumulates up to the rectangle’s bottom/right)
        #       B = I[r0,   c1+1]   # top-right    corner lookup (removes everything above the rectangle)
        #       C = I[r1+1, c0   ]  # bottom-left  corner lookup (removes everything left  of the rectangle)
        #       D = I[r0,   c0   ]  # top-left     corner lookup (adds back the overlap subtracted twice)
        # We now create four vectorised slices for each band that represent those corner lookups for *all* pixels. 

        # ----------------------------- ABOVE band: rows [r-n .. r-1], cols [c-k .. c+k] -----------------------------

        above_sum = (                                                              # start inclusion–exclusion for ABOVE
            integral[min_vertical_size : min_vertical_size + self.number_of_rows,  # I[r1+1, ·]: bottom row index for ABOVE → r1 = r-1 ⇒ r1+1 = r
                    window_width : window_width + self.number_of_columns]         # I[·, c1+1]: right col index for window (c+k)+1 → shift by window_width
            - integral[0 : self.number_of_rows,                                    # I[r0,   ·]: top row index for ABOVE → r0 = r-n ⇒ in integral coords = r-n
                    window_width : window_width + self.number_of_columns]       # I[·, c1+1]: same right boundary slice as above
            - integral[min_vertical_size : min_vertical_size + self.number_of_rows,# I[r1+1, c0]: left col index for window (c-k) → in integral coords = c-k
                    0 : self.number_of_columns]                                 # I[·,   c0]: left boundary slice aligned to each start column
            + integral[0 : self.number_of_rows,                                    # I[r0,   c0]: overlap (top-left) added back once
                    0 : self.number_of_columns]                                 # I[·,   c0]: left boundary slice
        )  # end ABOVE

        # Explanation of the slices seen above (intuitive view):                                                             
        # - Rows slice [min_vertical_size : min_vertical_size + R] corresponds to r1+1 for every r (vectorised).        
        # - Rows slice [0 : R] corresponds to r0 for every r (vectorised).                                              
        # - Cols slice [window_width : window_width + C] corresponds to c1+1 for every c (vectorised).                  
        # - Cols slice [0 : C] corresponds to c0 for every c (vectorised).                                              

        

        # ----------------------------- BELOW band: rows [r+1 .. r+n], cols [c-k .. c+k] -----------------------------
        #   For a pixel at (r_pixel, c_pixel) in the BELOW band:
        #       r0 = r_pixel + 1,  r1 = r_pixel + n    # inclusive row limits strictly below the seed row
        #       c0 = c_pixel - k,  c1 = c_pixel + k    # inclusive column limits (same as ABOVE)
        #   (Use the same inclusion–exclusion: sum = I[r1+1, c1+1] - I[r0, c1+1] - I[r1+1, c0] + I[r0, c0].)

        below_sum = (                                                              # start inclusion–exclusion for BELOW
            integral[2 * min_vertical_size + 1 : 2 * min_vertical_size + 1 + self.number_of_rows,  # I[r1+1, ·]: for BELOW, r1 = r+n ⇒ r1+1 = r+n+1 ⇒ shift by (2n+1)
                    window_width : window_width + self.number_of_columns]         # I[·, c1+1]: same right boundary slice
            - integral[min_vertical_size + 1 : min_vertical_size + 1 + self.number_of_rows,        # I[r0,   ·]: for BELOW, r0 = r+1 ⇒ in integral coords = r+1
                    window_width : window_width + self.number_of_columns]       # I[·, c1+1]: same right boundary slice
            - integral[2 * min_vertical_size + 1 : 2 * min_vertical_size + 1 + self.number_of_rows,# I[r1+1, c0]: left boundary slice
                    0 : self.number_of_columns]                                 # I[·,   c0]: left boundary slice
            + integral[min_vertical_size + 1 : min_vertical_size + 1 + self.number_of_rows,        # I[r0,   c0]: overlap (top-left) added back once
                    0 : self.number_of_columns]                                 # I[·,   c0]: left boundary slice
        )  # end BELOW

        # Explanation of the BELOW slices (intuitive view):                                                              
        # - Rows slice [n+1 : n+1+R] supplies r0 for BELOW (top of the lower band).                                     
        # - Rows slice [2n+1 : 2n+1+R] supplies r1+1 for BELOW (bottom of the lower band + 1 for integral coords).      
        # - Column slices are identical to ABOVE: [0:C] for c0 and [W:W+C] for c1+1, vectorised over all start columns. 

        # ----------------------------- combine, threshold, and gate to seeds -----------------------------
        neighbour_count = above_sum + below_sum                                     # total neighbours strictly above+below at every pixel
        keep_mask = neighbour_count >= int(min_neighbours)                          # pixels that meet/beat the neighbour threshold
        pruned_mask = seed_mask_boolean & keep_mask                                 # keep only original seeds that pass the threshold

        return pruned_mask                                                          # outputs

        
    def grow_seeds(self, growth_threshold: float = 1.0):
        """
        Expand seed groups by including adjacent pixels that meet a lower threshold.
        """
    
    def plot_1D(self, nothing_initial_or_consolidated: str = "initial", save_location: str = None, show_plot_bool: bool = True):
        """
        Plot the 1D signal data with thresholds and seed indicators.

        Parameters:
        - none_initial_or_consolidated (str): Choose between plotting the no seed masks, initial seed mask, 
                                       or the consolidated group mask.
                                       Options are "nothin", "initial" or "consolidated".
        - save_location (str): Optional path to save the plot as a PDF.
        - show_plot_bool (bool): Whether to display the plot interactively.

        Produces:
        - A series of 16 subplots (one for each row) showing the signal, threshold, and seed points.
        - Displays the plot if show_plot_bool is True.
        - Saves the plot as a PDF if a save_location is provided.


        Note: Initial boolean mask or consolidated group boolean mask must be computed before calling this method.
        """

        # Raise errors if prerequisites are not met
        if nothing_initial_or_consolidated not in ["nothing", "initial", "consolidated"]:
            raise ValueError("nothing_initial_or_consolidated must be either 'nothing', 'initial' or 'consolidated'")
        if nothing_initial_or_consolidated == "initial" and self.initial_boolean_mask is None:
            raise ValueError("Initial boolean mask must be computed before plotting initial seeds.")
        elif nothing_initial_or_consolidated == "consolidated" and self.consolidated_group_boolean_mask is None:
            raise ValueError("Consolidated group boolean mask must be computed before plotting consolidated seeds.")
        fig, ax = plt.subplots(nrows=self.number_of_rows, ncols=1, figsize=(6, self.number_of_rows*1))  # create figure and axis
        x_axis_values = np.arange(self.number_of_columns)            # x axis (column indices)
        for row_index in range(self.number_of_rows):                 # iterate over rows
            ax[row_index].plot(x_axis_values, self.signal_snippet[row_index], label='Signal', alpha=0.7)
            ax[row_index].axhline(self.row_thresholds[row_index], color='orange', linestyle='--', label='Threshold')  # plot threshold
            ax[row_index].axhline(self.row_medians[row_index], color='blue', linestyle=':', label='Median')  # plot median 
            if nothing_initial_or_consolidated == "initial":
                seed_columns = np.flatnonzero(self.initial_boolean_mask[row_index])
                ax[row_index].scatter(seed_columns, self.signal_snippet[row_index, seed_columns], color='red', label='Initial Seeds', s=5)
                fig.suptitle("1D Signal with Initial Seeds", y=0.995)
            elif nothing_initial_or_consolidated == "consolidated":
                seed_columns = np.flatnonzero(self.consolidated_group_boolean_mask[row_index])
                ax[row_index].scatter(seed_columns, self.signal_snippet[row_index, seed_columns], color='red', label='Consolidated Seeds', s=5)
                fig.suptitle("1D Signal with Consolidated Seeds", y=0.995)
            elif nothing_initial_or_consolidated == "nothing":
                fig.suptitle(f"1D Signal with No Seeds, unique_id: {self.unique_id}", y=0.995)
            ax[row_index].set_ylabel(f"Row {row_index}")
            # remove x labels
            if row_index != self.number_of_rows - 1:
                ax[row_index].set_xticklabels([])
        ax[-1].set_xlabel("Column Index")

        # add just one legend to the right of the plots outside of them
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1), fontsize='small')

        plt.tight_layout()
        if show_plot_bool:
            plt.show()
        if isinstance(save_location, str):
            fig.savefig(f"{save_location}/1D_Plots_With_{nothing_initial_or_consolidated}_Seeds.pdf")
            print(f"1D Plot saved to {save_location}/1D_Plots_With_{nothing_initial_or_consolidated}_Seeds.pdf")
    
    def plot_2D(self, nothing_initial_or_consolidated: str = "initial", save_location: str = None, show_plot_bool: bool = True):
        """
        Plot the 2D signal data with thresholds and seed indicators.

        Parameters:
        - none_nothing_initial_or_consolidated (str): Choose between plotting the no seed masks, initial seed mask, 
                                       or the consolidated group mask.
                                       Options are "nothin", "initial" or "consolidated".
        - save_location (str): Optional path to save the plot as a PDF.
        - show_plot_bool (bool): Whether to display the plot interactively.

        Produces:
        - A 2D plot showing the signal, thresholds, and seed points.
        - Displays the plot if show_plot_bool is True.
        - Saves the plot as a PDF if a save_location is provided.

        Note: Initial boolean mask or consolidated group boolean mask must be computed before calling this method.
        """
        # Raise errors if prerequisites are not met
        if nothing_initial_or_consolidated not in ["nothing", "initial", "consolidated"]:
            raise ValueError("nothing_initial_or_consolidated must be either 'nothing', 'initial' or 'consolidated'")
        if nothing_initial_or_consolidated == "initial" and self.initial_boolean_mask is None:
            raise ValueError("Initial boolean mask must be computed before plotting initial seeds.")
        elif nothing_initial_or_consolidated == "consolidated" and self.consolidated_group_boolean_mask is None:
            raise ValueError("Consolidated group boolean mask must be computed before plotting consolidated seeds.")
        

        fig, ax = plt.subplots(figsize=(12, 8))

        cax = ax.imshow(self.signal_snippet, aspect='auto', cmap='viridis', interpolation='nearest')
        fig.colorbar(cax, ax=ax, label='Signal Intensity')  # colorbar for signal intensity
        x_axis_values = np.arange(self.number_of_columns)            # x axis (column indices)

        for row_index in range(self.number_of_rows):                 # iterate over rows
            if nothing_initial_or_consolidated == "initial":
                seed_columns = np.flatnonzero(self.initial_boolean_mask[row_index])
                ax.scatter(seed_columns, np.full_like(seed_columns, row_index), color='red', label='Initial Seeds' if row_index == 0 else "", s=10)
                fig.suptitle("2D Signal with Initial Seeds", y=0.995)
            elif nothing_initial_or_consolidated == "consolidated":
                seed_columns = np.flatnonzero(self.consolidated_group_boolean_mask[row_index])
                ax.scatter(seed_columns, np.full_like(seed_columns, row_index), color='red', label='Consolidated Seeds' if row_index == 0 else "", s=10)
                fig.suptitle(f"2D Snippet with Consolidated Seeds, unique_id: {self.unique_id}", y=0.995)
            elif nothing_initial_or_consolidated == "nothing":
                fig.suptitle(f"2D Snippet with No Seeds, unique_id: {self.unique_id}", y=0.995)
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.legend(loc='upper right')
        # title
        plt.tight_layout()
        if show_plot_bool:
            plt.show()
        if isinstance(save_location, str):
            fig.savefig(f"{save_location}/1D_Plots_With_{nothing_initial_or_consolidated}_Seeds.pdf")
            print(f"1D Plot saved to {save_location}/1D_Plots_With_{nothing_initial_or_consolidated}_Seeds.pdf")



