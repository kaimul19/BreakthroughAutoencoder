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



