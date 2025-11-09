import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit, prange

# dont jit this function as it uses non numba functions
def _consolidate_1d(
    flattened_array: np.ndarray, maximum_gap: int, unique_id: np.ndarray
) -> np.ndarray:
    """
    Merge 1-runs in a 1D boolean/integer mask by filling gaps of length <= maximum_gap.
    Returns a boolean 1D array with consolidated seeds.
    """
    distances = np.diff(
        flattened_array, prepend=0, append=0
    )  # compute differences to find seed boundaries
    starts = np.flatnonzero(distances == 1)  # find start indices of seeds
    ends = np.flatnonzero(distances == -1) - 1  # find end indices of seeds

    if starts.size == 0:
        print(
            f"No Seeds Found in the snippet at index: {unique_id}"
        )  # debug message if no seeds found
        return np.zeros_like(flattened_array, dtype=bool)

    gaps = (starts[1:] - ends[:-1]) - 1  # compute gaps between seeds
    cuts = np.where(gaps > maximum_gap)[0]  # identify gaps larger than maximum_gap

    merged_starts = np.r_[starts[0], starts[cuts + 1]]  # merge start indices
    merged_ends = np.r_[ends[cuts], ends[-1]]  # merge end indices

    difference_buffer = np.zeros(
        flattened_array.size + 1, dtype=np.int32
    )  # create difference buffer
    np.add.at(difference_buffer, merged_starts, 1)  # mark starts
    np.add.at(difference_buffer, merged_ends + 1, -1)  # mark ends

    return np.cumsum(difference_buffer[:-1]) > 0  # return consolidated boolean mask


@njit(parallel=True)
def scan_horizontal_surroundings(
    pruned_2d: np.ndarray,
    true_indicies: np.ndarray,
    max_horizontal_gap: int,
) -> np.ndarray:

    # make a verified_array where it is starting as a False array
    for i in prange(len(true_indicies)):
        row, col = true_indicies[i]
        left_column_index = max(0, col - max_horizontal_gap)  # ensure not negative 
        right_column_index = min(pruned_2d.shape[1]-1, col + max_horizontal_gap)  # ensure not out of bounds

        sum_region = pruned_2d[row, left_column_index:right_column_index+1].sum()  # sum in the region
        if sum_region < min_neighbours + 1:  # +1 to account for the seed itself
            pruned_2d[row, col] = 0  # prune the seed if not enough neighbours
    return pruned_2d


class signal_data:
    def __init__(
        self,
        signal_snippet: np.ndarray,
        unique_id: np.ndarray,
        sigma_multiplier: float = 3.0,
    ):
        """
        Container for 2D signal snippets with row-based statistics and seed operations.

        General idea with this class is to encapsulate a 2D signal snippet (e.g., time vs frequency)
        for pruning methods we would generally suggest the following workflow:
            1. Instantiate the class with a 2D signal snippet and unique identifier.
            2. Call compute_row_statistics() to calculate per-row medians, stds, and thresholds.
            3. Call get_seeds() to identify initial seed pixels exceeding thresholds.
            4. Call Prune_methods() to prune seeds based on horizontal and vertical criteria.
            5. Call consolidate_seeds() to merge adjacent seed pixels into groups.
            6. Call grow_seeds() to expand seed groups by including adjacent pixels that meet a lower threshold.
        Parameters:
        - signal_snippet (np.ndarray): 2D array of signal data (rows x columns).
        - unique_id (np.ndarray): 2D array [observation_index, snippet_index] for debugging.
        - sigma_multiplier (float): Multiplier for standard deviation to set threshold.
        """
        self.signal_snippet = np.ascontiguousarray(
            signal_snippet, dtype=np.float64
        )  # store image as float array
        if self.signal_snippet.ndim != 2:  # validate dimensionality
            raise ValueError("image_2d must be a 2D array")  # raise error if not 2D
        if self.signal_snippet.shape[0] != 16:
            print("Warning: Expected 16 rows in the snippet array")
            print(
                "These corresponding to the 16 time bins of an observation"
            )  # warn if unexpected shape
        if (
            unique_id is None
            or not isinstance(unique_id, np.ndarray)
            or len(unique_id) != 2
        ):
            raise ValueError(
                f"unique_id must be an np.2darray of [observation_index, snippet_index] instead got {unique_id}, type {type(unique_id)}"
            )
        self.number_of_rows, self.number_of_columns = (
            self.signal_snippet.shape
        )  # cache shape information
        self.row_medians = None  # placeholder for per-row medians
        self.row_std = None  # placeholder for per-row stds
        self.row_thresholds = None  # placeholder for per-row thresholds
        self.sigma_multiplier = float(sigma_multiplier)  # store sigma multiplier
        self.unique_id = unique_id  # store unique identifier for debugging
        self.initial_boolean_mask = (
            None  # placeholder for initial boolean mask of seeds
        )
        self.final_pruned_mask = (
            None
        )  # placeholder for final pruned boolean mask of seeds
        self.consolidated_group_boolean_mask = (
            None  # placeholder for consolidated boolean mask of seed groups
        )

    def _return_shape(self):
        """
        Return the shape of the signal snippet.
        """
        return self.signal_snippet.shape

    def _return_statistics(self):
        """
        Return the computed row statistics: thresholds, medians, and standard deviations.
        """
        if (
            self.row_thresholds is None
            or self.row_medians is None
            or self.row_std is None
        ):
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
            raise ValueError(
                "Consolidated group boolean mask has not been computed yet."
            )
        return (
            self.consolidated_group_boolean_mask,
            self.consolidated_group_boolean_mask.shape,
        )
    
    def _return_final_pruned_mask(self):
        """
        Return the final pruned boolean mask of seed pixels.
        """
        if self.final_pruned_mask is None:
            raise ValueError("Final pruned mask has not been computed yet.")
        return self.final_pruned_mask
    
    def _return_grown_seed_mask(self):
        """
        Return the grown seed boolean mask.
        """
        if self.grown_seed_mask is None:
            raise ValueError("Grown seed mask has not been computed yet.")
        return self.grown_seed_mask

    def compute_row_statistics(self):
        """
        Compute median and standard deviation for each row in the 2D signal array.

        This method calculates the median, standard deviation, and thresholds for each row of the signal_snippet array.

        """
        self.row_medians = np.median(
            self.signal_snippet, axis=1
        )  # compute per-row medians
        self.row_std = np.std(
            self.signal_snippet, axis=1
        )  # compute per-row standard deviations
        self.row_thresholds = (
            self.row_medians + self.sigma_multiplier * self.row_std
        )  # compute thresholds

    def get_seeds(self):
        """
        Identify seed pixels exceeding per-row thresholds (vectorised NumPy).
        """
        if self.row_thresholds is None:  # Ensure statistics are computed
            raise ValueError(
                "Row statistics must be computed before identifying seeds."
            )

        if self.initial_boolean_mask is not None:  # Check if mask already exists
            raise ValueError(
                "Initial boolean mask already exists. You should not call this method twice."
            )

        # Preallocate boolean mask
        self.initial_boolean_mask = np.empty(self.signal_snippet.shape, dtype=bool)

        # Vectorised broadcast comparison, writing directly into mask (fast, no extra allocs)
        np.greater(
            self.signal_snippet,
            self.row_thresholds[:, None],
            out=self.initial_boolean_mask,
        )

    def consolidate_seeds(self, max_pixel_distance_either_side=5, use_final_mask=True):
        """
        Consolidate seed pixels into groups based on adjacency.
        This method merges seed pixels that are within a specified maximum pixel distance
        Parameters:
        - max_pixel_distance_either_side (int): Maximum gap of non-seed pixels to bridge when consolidating.
        - use_final_mask (bool): Whether to use the final pruned mask for consolidation. If False, uses the initial boolean mask.

        Returns:
        - consolidated_group_boolean_mask (np.ndarray): Boolean mask of consolidated seed groups.

        """

        if self.consolidated_group_boolean_mask is not None:
            raise ValueError(
                "Consolidated group boolean mask already exists. You should not call this method twice."
            )

        if use_final_mask:
            if self.final_pruned_mask is None:
                raise ValueError(
                    "Final pruned mask must be computed before consolidating seeds. First perform pruning or set use_final_mask to False to use the initial boolean mask instead. "
                )
        else:
            if self.initial_boolean_mask is None:   # First check final as it is pruned and hence implies initial exists
                raise ValueError(
                    "Initial boolean mask must be computed before consolidating seeds."
                )
            print("Using initial boolean mask for consolidation, I would not currently suggest prunning after this as functionality has not been completely verified.")


        seperation = (
            max_pixel_distance_either_side + 1
        )  # define seperation with additional index so that runs on edges are not merged
        work = np.zeros(
            (self.number_of_rows, self.number_of_columns + seperation), dtype=bool
        )  # padded work array to avoid wrap-around
        if use_final_mask: # use final mask
            work[:, : self.number_of_columns] = self.final_pruned_mask.astype(
                bool, copy=True
                )  # populate work array
        else: # use initial mask
            work[:, : self.number_of_columns] = self.final_pruned_mask.astype(
                bool, copy=True
                )
        flat = work.ravel()  # flatten for processing
        cons_flat = _consolidate_1d(
            flat, max_pixel_distance_either_side, unique_id=self.unique_id
        )  # call merging method
        self.consolidated_group_boolean_mask = cons_flat.reshape(
            self.number_of_rows, self.number_of_columns + seperation
        )[
            :, : self.number_of_columns
        ]  # reshape back and trim padding

        return self.consolidated_group_boolean_mask

    def prune_methods(
        self,
        max_horizontal_gap: int = 3,
        max_vertical_gap: int = 1,
        drift_padding_gap: int = 1,
        min_neighbours: int = 1,
        gate_requirment_for_false: str = "or",
        return_masks: bool = False,
    ):
        """
        Prune consolidated groups based on horizontal and vertical criteria.
        gate_requirment_for_false options are "and", "or", "horizontal", "vertical"

        Parameters:
        - max_horizontal_gap (int): number of consecutive non-seed pixels on either side allowed horizontally within a group.
        - max_vertical_gap (int): max number of rows strictly above/below to consider for vertical pruning.
        - drift_padding_gap (int): number of columns left/right to consider for vertical pruning.
        - min_neighbours (int): minimum number of neighbours required to keep a seed during vertical pruning.

        Returns:
        - final_pruned_mask (np.ndarray): Boolean mask of seeds after applying pruning criteria.
        """
        if self.initial_boolean_mask is None:
            raise ValueError("Initial boolean masks must be computed before pruning.")

        if gate_requirment_for_false not in ["and", "or", "horizontal", "vertical"]:
            raise ValueError(
                "gate_requirment_for_false must be one of 'and', 'or', 'horizontal', or 'vertical'."
            )

        horizontal_pruned_mask = self._prune_horizontal(
            max_horizontal_gap=max_horizontal_gap
        )  # first horizontal pruning

        if (
            gate_requirment_for_false == "horizontal"
        ):  # shortcut if only horizontal pruning is needed
            final_pruned_mask = horizontal_pruned_mask
            return final_pruned_mask  # return pruned mask immediately, so vertical pruning is skipped

        vertical_pruned_mask = self._prune_vertical(  # then vertical pruning
            max_vertical_gap=max_vertical_gap,
            drift_padding_gap=drift_padding_gap,
            min_neighbours=min_neighbours,
        )

        if (
            gate_requirment_for_false == "vertical"
        ):  # shortcut if only vertical pruning is needed
            final_pruned_mask = vertical_pruned_mask
            return final_pruned_mask  # return pruned mask immediately, so horizontal pruning is skipped

        if gate_requirment_for_false == "or":  # combine masks with logical OR
            final_pruned_mask = horizontal_pruned_mask | vertical_pruned_mask
        elif gate_requirment_for_false == "and":  # combine masks with logical AND
            final_pruned_mask = horizontal_pruned_mask & vertical_pruned_mask

        self.final_pruned_mask = final_pruned_mask  # store final pruned mask

        if return_masks:
            return final_pruned_mask, horizontal_pruned_mask, vertical_pruned_mask

        return final_pruned_mask  # return final pruned mask

    def _prune_horizontal(self, max_horizontal_gap: int = 3, min_neighbours: int = 1):
        """
        Prune groups that do not meet horizontal size criteria.

        Parameters:
        - max_horizontal_gap (int): Maximum allowed gap of non-seed pixels within a group. 
                                        if checking e.g. the 5th index in a row, then pixels 2,3,4 and 6,7,8 are included in the check.
        - min_neighbours (int): Minimum neighbours required in that range to keep a seed.

        Returns:
        - pruned_mask (np.ndarray): Boolean mask of seeds that meet horizontal size criteria.
        """
        true_indicies = np.argwhere(self.initial_boolean_mask)  # get indices of true seeds 

        #initial_boolean_mask is already in type bool
        pruned_2d = self.initial_boolean_mask.astype(np.int8, copy=True) # can use this for both prune mask and neighbor count



        pruned_2d = scan_horizontal_surroundings(pruned_2d, 
                                                    true_indicies=true_indicies, 
                                                    max_horizontal_gap=max_horizontal_gap)

        pruned_mask = pruned_2d.astype(bool, copy=False)  # convert back to boolean mask

        return pruned_mask  # output




    def _prune_vertical(
        self,
        max_vertical_gap: int = 1,  # vertical radius n: rows strictly above/below to include
        drift_padding_gap: int = 1,  # horizontal radius k: columns left/right to include
        min_neighbours: int = 1,  # minimum neighbours required to keep a seed
    ):
        """
        Keep seeds that have at least `min_neighbours` neighbours in rows strictly
        above/below (±max_vertical_gap) within ±drift_padding_gap columns.
        The *same row* as the seed is excluded by construction (we sum two rectangles:
        the one above and the one below the centre row).

        Parameters:
        - max_vertical_gap (int): Minimum vertical size of groups to retain.
        - drift_padding_gap (int): Horizontal half-width for neighbour search.
        - min_neighbours (int): Minimum number of neighbours required to keep a seed.

        Returns:
        - pruned_mask (np.ndarray): Boolean mask of seeds that meet neighbour criteria.
        - keep_mask (np.ndarray): Boolean mask of all pixels that meet neighbour criteria.
        - neighbour_count (np.ndarray): Integer array of neighbour counts for each pixel.
        """
        if not isinstance(max_vertical_gap, int):  # ensure type is integer
            raise ValueError(
                "max_vertical_gap must be an integer."
            )  # raise on bad type
        if not isinstance(drift_padding_gap, int):  # ensure type is integer
            raise ValueError(
                "drift_padding_gap must be an integer."
            )  # raise on bad type
        if not isinstance(min_neighbours, int):  # ensure type is integer
            raise ValueError("min_neighbours must be an integer.")  # raise on bad type
        if max_vertical_gap == 0:  # disallow zero vertical radius
            raise ValueError(
                "max_vertical_gap must be at least 1."
                "If you want to just use horizontal pruning, call "
                "prune_methods with gate_requirment_for_false='horizontal'"
            )  # guidance
        if min_neighbours < 1:  # disallow non-positive threshold
            raise ValueError(
                "min_neighbours must be at least 1."
                "If you want to just use horizontal pruning,"
                "call prune_methods with gate_requirment_for_false='horizontal'"
            )  # guidance

        # NOTE Cruicial for understanding the comments that follow:
        # n = max_vertical_gap,
        # k = drift_padding_gap,
        # R = number_of_rows,
        # C = number_of_columns,
        # W = window_width = 2k+1
        # r = row index of a pixel in original image
        # c = column index of a pixel in original image

        binary_int = self.initial_boolean_mask.astype(
            np.int32, copy=True
        )  # convert original seed mask to int for summations, copy to avoid modifying original
        seed_mask_boolean = self.initial_boolean_mask.astype(
            bool, copy=False
        )  # boolean version to gate results back to original seeds
        window_width = 2 * drift_padding_gap + 1  # horizontal window width (2k+1)

        # We pad by n rows and k columns so every pixel has a full ABOVE/BELOW window even at edges.
        pad_mode = "constant"  # enforce constant padding for integral image method
        padded = np.pad(  # build padded image around the binary mask
            binary_int,  # source integer mask (0/1)
            pad_width=(
                (max_vertical_gap, max_vertical_gap),  # pad top and bottom by n rows
                (drift_padding_gap, drift_padding_gap),
            ),  # pad left and right by k cols
            mode=pad_mode,  # padding rule (default constant zeros)
        )

        # Build a 2D integral image (summed-area table) with an extra top row/left column of zeros.
        # Convention: integral[r, c] = sum of padded[0:r, 0:c], i.e. a half-open box (row r and col c excluded).
        integral = np.pad(
            padded,
            pad_width=((1, 0), (1, 0)),
            mode="constant",
        )  # prepend a zero row and zero column
        integral = integral.cumsum(axis=0).cumsum(
            axis=1
        )  # cumulative sums down then across (2D prefix sums)

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

        above_sum = (  # start inclusion–exclusion for ABOVE
            integral[
                max_vertical_gap : max_vertical_gap
                + self.number_of_rows,  # I[r1+1, ·]: bottom row index for ABOVE → r1 = r-1 ⇒ r1+1 = r
                window_width : window_width + self.number_of_columns,
            ]  # I[·, c1+1]: right col index for window (c+k)+1 → shift by window_width
            - integral[
                0 : self.number_of_rows,  # I[r0,   ·]: top row index for ABOVE → r0 = r-n ⇒ in integral coords = r-n
                window_width : window_width + self.number_of_columns,
            ]  # I[·, c1+1]: same right boundary slice as above
            - integral[
                max_vertical_gap : max_vertical_gap
                + self.number_of_rows,  # I[r1+1, c0]: left col index for window (c-k) → in integral coords = c-k
                0 : self.number_of_columns,
            ]  # I[·,   c0]: left boundary slice aligned to each start column
            + integral[
                0 : self.number_of_rows,  # I[r0,   c0]: overlap (top-left) added back once
                0 : self.number_of_columns,
            ]  # I[·,   c0]: left boundary slice
        )  # end ABOVE

        # Explanation of the slices seen above (intuitive view):
        # - Rows slice [max_vertical_gap : max_vertical_gap + R] corresponds to r1+1 for every r (vectorised).
        # - Rows slice [0 : R] corresponds to r0 for every r (vectorised).
        # - Cols slice [window_width : window_width + C] corresponds to c1+1 for every c (vectorised).
        # - Cols slice [0 : C] corresponds to c0 for every c (vectorised).

        # ----------------------------- BELOW band: rows [r+1 .. r+n], cols [c-k .. c+k] -----------------------------
        #   For a pixel at (r_pixel, c_pixel) in the BELOW band:
        #       r0 = r_pixel + 1,  r1 = r_pixel + n    # inclusive row limits strictly below the seed row
        #       c0 = c_pixel - k,  c1 = c_pixel + k    # inclusive column limits (same as ABOVE)
        #   (Use the same inclusion–exclusion: sum = I[r1+1, c1+1] - I[r0, c1+1] - I[r1+1, c0] + I[r0, c0].)

        below_sum = (  # start inclusion–exclusion for BELOW
            integral[
                2 * max_vertical_gap
                + 1 : 2 * max_vertical_gap
                + 1
                + self.number_of_rows,  # I[r1+1, ·]: for BELOW, r1 = r+n ⇒ r1+1 = r+n+1 ⇒ shift by (2n+1)
                window_width : window_width + self.number_of_columns,
            ]  # I[·, c1+1]: same right boundary slice
            - integral[
                max_vertical_gap
                + 1 : max_vertical_gap
                + 1
                + self.number_of_rows,  # I[r0,   ·]: for BELOW, r0 = r+1 ⇒ in integral coords = r+1
                window_width : window_width + self.number_of_columns,
            ]  # I[·, c1+1]: same right boundary slice
            - integral[
                2 * max_vertical_gap
                + 1 : 2 * max_vertical_gap
                + 1
                + self.number_of_rows,  # I[r1+1, c0]: left boundary slice
                0 : self.number_of_columns,
            ]  # I[·,   c0]: left boundary slice
            + integral[
                max_vertical_gap
                + 1 : max_vertical_gap
                + 1
                + self.number_of_rows,  # I[r0,   c0]: overlap (top-left) added back once
                0 : self.number_of_columns,
            ]  # I[·,   c0]: left boundary slice
        )  # end BELOW

        # Explanation of the BELOW slices (intuitive view):
        # - Rows slice [n+1 : n+1+R] supplies r0 for BELOW (top of the lower band).
        # - Rows slice [2n+1 : 2n+1+R] supplies r1+1 for BELOW (bottom of the lower band + 1 for integral coords).
        # - Column slices are identical to ABOVE: [0:C] for c0 and [W:W+C] for c1+1, vectorised over all start columns.

        # ----------------------------- combine, threshold, and gate to seeds -----------------------------
        neighbour_count = (
            above_sum + below_sum
        )  # total neighbours strictly above+below at every pixel
        keep_mask = neighbour_count >= int(
            min_neighbours
        )  # pixels that meet/beat the neighbour threshold
        pruned_mask = (
            seed_mask_boolean & keep_mask
        )  # keep only original seeds that pass the threshold

        return pruned_mask 

    def grow_seeds(self, growth_threshold_above_median: float = 1.0):
        """
        Grow seeds horizontally out to the full contiguous above-threshold region.

        For each row:
            - Define eligible pixels as those where
                signal[row, col] > (row_medians[row] + growth_threshold_above_median)
            - Break eligible into contiguous horizontal runs.
            - Keep (fill) an entire run if that run contains at least one seed pixel
            from the chosen base mask (consolidated > final_pruned > initial).
            - Drop runs that contain no seeds.

        This is done in one vectorized pass using 1D flattening with row padding to
        avoid wraparound between rows.

        Result is stored in self.grown_seed_mask and also returned.
        """

        seed_mask_type = "consolidated"
        if self.consolidated_group_boolean_mask is None:
            print("Warning: Consolidated group boolean mask has not been computed yet.")
            print("Will first attempt to consolidate seeds using final pruned mask.")
            seed_mask_type = "final_pruned"
            if self.final_pruned_mask is None:
                print("Warning: Final pruned mask has not been computed yet. So will use initial boolean mask instead.")

                seed_mask_type = "initial"
                if self.initial_boolean_mask is None:
                    raise ValueError(
                        "Initial boolean mask must be computed before growing seeds. You must have at least one seed to grow from."
                    )
        if seed_mask_type == "consolidated":
            seed_mask = self.consolidated_group_boolean_mask.copy()
        elif seed_mask_type == "final_pruned":
            seed_mask = self.final_pruned_mask.copy()
        elif seed_mask_type == "initial":  # initial
            seed_mask = self.initial_boolean_mask.copy()
        else:
            raise ValueError(f"Unknown seed mask type: {seed_mask_type}")

        
        # ---------- build the eligibility mask based on intensity threshold ----------
        # per-row threshold: median[row] + growth_threshold_above_median
        # shape (rows, 1) so it broadcasts across columns
        growth_thresholds = self.row_medians[:, None] + growth_threshold_above_median
        eligible_mask = self.signal_snippet > growth_thresholds  # shape (Rows, Columns), bool

        # We only care about eligible pixels; we'll later keep only the eligible runs
        # that intersect at least one True in seed_mask.

        # ---------- pad each row with one False column to break adjacency between rows ----------
        # so runs never connect across row boundaries when flattened.

        padded_len = self.number_of_columns + 1 # single-column separator of zeros between rows in the flattened view

        # allocate padded arrays
        padded_eligible = np.zeros((self.number_of_rows, padded_len), dtype=np.bool_)
        padded_seeds    = np.zeros((self.number_of_rows, padded_len), dtype=np.bool_)

        padded_eligible[:, :self.number_of_columns] = eligible_mask
        padded_seeds[:,    :self.number_of_columns] = seed_mask

        # flatten
        flat_eligible = padded_eligible.ravel()  # 1D bool
        flat_seeds    = padded_seeds.ravel()     # 1D bool

        # ---------- find contiguous runs of eligible==True in the flattened array ----------
        # Similar method as that in _consolidate_flattened_rows.
        # diff on flat_eligible to get run starts (+1) and run ends (-1).
        diff = np.diff(flat_eligible.astype(np.uint8), prepend=0, append=0)
        run_starts = np.flatnonzero(diff == 1)  # these have value +1
        run_ends   = np.flatnonzero(diff == -1) - 1  # these have value -1; inclusive ends

        # If there are no eligible runs, growth mask is just the original seed mask.
        if run_starts.size == 0:
            print("Warning: No eligible runs found for seed growth; returning original seed mask.")
            grown_mask = seed_mask.copy()
            self.grown_seed_mask = grown_mask
            return grown_mask

        # ---------- for each run, decide if it contains at least one seed ----------
        # We'll use a prefix sum over flat_seeds to query "any seeds in [start:end]"
        prefix_seeds = np.cumsum(flat_seeds.astype(np.int32))

        # For a run [a, b] inclusive, number of seeds in it is:
        #   prefix_seeds[b] - prefix_seeds[a-1]  (careful at a==0) as it is the number of seeds up to b minus those up to a-1
        # We'll vectorize that:
        left_vals = np.zeros_like(run_starts, dtype=np.int32)
        left_vals[run_starts > 0] = prefix_seeds[run_starts[run_starts > 0] - 1] 

        seeds_in_run = (prefix_seeds[run_ends] - left_vals) > 0  # bool per run > 0 
        # above is the calculation of seeds in each run prefix_seeds[run_ends] is the number of seeds up to the end of each run
        # left_vals is the number of seeds up to the start of each run -1 (or 0 if start is 0)

        # ---------- now build a new flat mask that keeps only runs that had seeds ----------
        flat_grown = np.zeros_like(flat_eligible, dtype=np.bool_)

        # We'll "paint" in only the runs that contain at least one seed.
        # We can do this with a difference buffer like in consolidation.
        diffbuf = np.zeros(flat_eligible.size + 1, dtype=np.int32)

        # Add +1 at run_starts and -1 at run_ends+1 (+1 here for inclusive ends) BUT ONLY for runs with seeds.
        valid_starts = run_starts[seeds_in_run]
        valid_ends   = run_ends[seeds_in_run]

        np.add.at(diffbuf, valid_starts, 1)
        np.add.at(diffbuf, valid_ends + 1, -1)

        # prefix sum to fill in the kept runs
        flat_grown = (np.cumsum(diffbuf[:-1]) > 0)

        # ---------- reshape back to (self.number_of_rows, padded_len), then crop off the separator col ----------
        grown_padded = flat_grown.reshape(self.number_of_rows, padded_len)
        grown_mask = grown_padded[:, :self.number_of_columns]

        # ---------- store and return ----------
        grown_mask |= seed_mask  # ensure original seeds are kept
        self.grown_seed_mask = grown_mask
        return grown_mask


    def _return_signal_masked(self, signal_mask):
        """
        Return the signal values at the grown seed mask locations.
        """
        if signal_mask is None:
            raise ValueError("signal_mask cannot be None.")
        
        if signal_mask.shape != self.signal_snippet.shape:
            raise ValueError(
                f"signal_mask shape {signal_mask.shape} does not match signal_snippet shape {self.signal_snippet.shape}."
            )
        
        false_indicies = np.where(~signal_mask)  # get indices of false pixels
        masked_signal = self.signal_snippet.copy()  # copy original signal
        masked_signal[false_indicies] = 0  # zero out non-grown seed locations
        return masked_signal


        





    def plot_1D(
        self,
        mask_type: str | np.ndarray | None = "initial",
        save_location: str = None,
        show_plot_bool: bool = True,
    ):
        """
        Plot the 1D signal data with thresholds and seed indicators.

        Parameters:
        - mask_type (str): Choose between plotting the no seed masks, initial seed mask,
                                       or the consolidated group mask.
                                       Options are "consolidated", "final_pruned", "grown", or "None" or a NumPy array mask.
        - save_location (str): Optional path to save the plot as a PDF.
        - show_plot_bool (bool): Whether to display the plot interactively.

        Produces:
        - A series of 16 subplots (one for each row) showing the signal, threshold, and seed points.
        - Displays the plot if show_plot_bool is True.
        - Saves the plot as a PDF if a save_location is provided.


        Note: Initial boolean mask or consolidated group boolean mask must be computed before calling this method.
        """

        valid_mask_types = [
            "consolidated",
            "final_pruned",
            "grown",
        ]

        if mask_type is None:
            # then we are plotting the original so my mask is just Falses
            mask = np.zeros(self.signal_snippet.shape, dtype=bool)
            mask_type_to_print = "No Mask"
        
        elif isinstance(mask_type, np.ndarray):
            # then we are plotting a custom mask
            if mask_type.shape != self.signal_snippet.shape:
                raise ValueError(
                    f"Custom mask shape {mask_type.shape} does not match signal_snippet shape {self.signal_snippet.shape}."
                )
            mask = mask_type
            mask_type_to_print = "Custom Mask"

        elif isinstance(mask_type, str):
            # if its not in the valid options raise error
            if mask_type not in valid_mask_types:
                raise ValueError(
                    f"mask_type must be one of {valid_mask_types}, None, or a NumPy array mask."
                )
            # then we are plotting one of the predefined masks
            if mask_type == "consolidated":
                if self.consolidated_group_boolean_mask is None:
                    raise ValueError(
                        "Consolidated group boolean mask must be computed before plotting consolidated seeds."
                    )
                mask = self.consolidated_group_boolean_mask
                mask_type_to_print = "Consolidated Mask"
            elif mask_type == "final_pruned":
                if self.final_pruned_mask is None:
                    raise ValueError(
                        "Final pruned mask must be computed before plotting final pruned seeds."
                    )
                mask = self.final_pruned_mask
                mask_type_to_print = "Final Pruned Mask"
            elif mask_type == "grown":
                if self.grown_seed_mask is None:
                    raise ValueError(
                        "Grown seed mask must be computed before plotting grown seeds."
                    )
                mask = self.grown_seed_mask
                mask_type_to_print = "Grown Seed Mask"
        else:
            raise ValueError(
                f"mask_type must be one of {valid_mask_types}, None, or a NumPy array mask."
            )

        fig, ax = plt.subplots(
            nrows=self.number_of_rows, ncols=1, figsize=(6, self.number_of_rows * 1)
        )  # create figure and axis
        x_axis_values = np.arange(self.number_of_columns)  # x axis (column indices)

        # first plot the signal, threshold and median lines
        for row_index in range(self.number_of_rows):  # iterate over rows
            ax[row_index].plot(
                x_axis_values, self.signal_snippet[row_index], label="Signal", alpha=0.7
            )
            ax[row_index].axhline(
                self.row_thresholds[row_index],
                color="orange",
                linestyle="--",
                label="Threshold",
            )  # plot threshold
            ax[row_index].axhline(
                self.row_medians[row_index], color="blue", linestyle=":", label="Median"
            )  # plot median
            # now plot the seeds from the chosen mask
            seed_columns = np.flatnonzero(mask[row_index])
            ax[row_index].scatter(
                seed_columns,
                self.signal_snippet[row_index, seed_columns],
                color="red",
                label=f"{mask_type_to_print} Seeds",
                s=5,
            )
            fig.suptitle(
                f"1D Signal with {mask_type_to_print}, unique_id: {self.unique_id}",
                y=0.995,
            )

            ax[row_index].set_ylabel(f"Row {row_index}")
            # remove x labels
            if row_index != self.number_of_rows - 1:
                ax[row_index].set_xticklabels([])
        ax[-1].set_xlabel("Column Index")

        # add just one legend to the right of the plots outside of them
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(1.0, 1),
            fontsize="small",
        )

        plt.tight_layout()
        if show_plot_bool:
            plt.show()
        if isinstance(save_location, str):
            fig.savefig(
                f"{save_location}/1D_Plots_With_{mask_type_to_print}_Seeds.pdf"
            )
            print(
                f"1D Plot saved to {save_location}/1D_Plots_With_{mask_type_to_print}_Seeds.pdf"
            )

    def plot_2D(
        self,
        mask_type: str | np.ndarray | None = "initial",
        save_location: str = None,
        show_plot_bool: bool = True,
    ):
        """
        Plot the 2D signal data with thresholds and seed indicators.

        Parameters:
        - mask_type (str): Choose between plotting the no seed masks, initial seed mask,
                                       or the consolidated group mask.
                                       Options are "consolidated", "final_pruned", "grown", or "None" or a NumPy array mask.
        - save_location (str): Optional path to save the plot as a PDF.
        - show_plot_bool (bool): Whether to display the plot interactively.

        Produces:
        - A 2D plot showing the signal, thresholds, and seed points.
        - Displays the plot if show_plot_bool is True.
        - Saves the plot as a PDF if a save_location is provided.

        Note: Initial boolean mask or consolidated group boolean mask must be computed before calling this method.
        """
        # Raise errors if prerequisites are not met
        valid_mask_types = [
            "consolidated",
            "final_pruned",
            "grown",
        ]

        if mask_type is None:
            # then we are plotting the original so my mask is just Falses
            mask = np.zeros(self.signal_snippet.shape, dtype=bool)
            mask_type_to_print = "No Mask"
        
        elif isinstance(mask_type, np.ndarray):
            # then we are plotting a custom mask
            if mask_type.shape != self.signal_snippet.shape:
                raise ValueError(
                    f"Custom mask shape {mask_type.shape} does not match signal_snippet shape {self.signal_snippet.shape}."
                )
            mask = mask_type
            mask_type_to_print = "Custom Mask"

        elif isinstance(mask_type, str):
            # if its not in the valid options raise error
            if mask_type not in valid_mask_types:
                raise ValueError(
                    f"mask_type must be one of {valid_mask_types}, None, or a NumPy array mask."
                )
            # then we are plotting one of the predefined masks
            if mask_type == "consolidated":
                if self.consolidated_group_boolean_mask is None:
                    raise ValueError(
                        "Consolidated group boolean mask must be computed before plotting consolidated seeds."
                    )
                mask = self.consolidated_group_boolean_mask
                mask_type_to_print = "Consolidated Mask"
            elif mask_type == "final_pruned":
                if self.final_pruned_mask is None:
                    raise ValueError(
                        "Final pruned mask must be computed before plotting final pruned seeds."
                    )
                mask = self.final_pruned_mask
                mask_type_to_print = "Final Pruned Mask"
            elif mask_type == "grown":
                if self.grown_seed_mask is None:
                    raise ValueError(
                        "Grown seed mask must be computed before plotting grown seeds."
                    )
                mask = self.grown_seed_mask
                mask_type_to_print = "Grown Seed Mask"
        else:
            raise ValueError(
                f"mask_type must be one of {valid_mask_types}, None, or a NumPy array mask."
            )

        fig, ax = plt.subplots(figsize=(12, 8))

        cax = ax.imshow(
            self.signal_snippet, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        fig.colorbar(
            cax, ax=ax, label="Signal Intensity"
        )  # colorbar for signal intensity
        x_axis_values = np.arange(self.number_of_columns)  # x axis (column indices)

        for row_index in range(self.number_of_rows):  # iterate over rows
            seed_columns = np.flatnonzero(mask[row_index])
            ax.scatter(
                seed_columns,
                np.full_like(seed_columns, row_index),
                color="red",
                label=f"{mask_type_to_print} Seeds" if row_index == 0 else "",
                s=10,
            )
            fig.suptitle(
                f"2D Signal with {mask_type_to_print}, unique_id: {self.unique_id}",
                y=0.995,
            )
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.legend(loc="upper right")
        # title
        plt.tight_layout()
        if show_plot_bool:
            plt.show()
        if isinstance(save_location, str):
            fig.savefig(
                f"{save_location}/1D_Plots_With_{mask_type_to_print}_Seeds.pdf"
            )
            print(
                f"1D Plot saved to {save_location}/1D_Plots_With_{mask_type_to_print}_Seeds.pdf"
            )


if __name__ == "__main__":
    # Example usage
    test_dataset = np.load("test_dataset.npy")  # Load a test dataset
    test_labels = np.load(
        "test_labels.npy", allow_pickle=True
    )  # Load corresponding labels
    test_signal_only = np.load("test_signal_only_dataset.npy")  # Load signal-only data
    id_pairs = np.array(list(test_labels[:, 3]), dtype=int)  # shape (M, 2), int

    testing_snippet = test_dataset[0]

    signal_instance = signal_data(
        testing_snippet, sigma_multiplier=3.0, unique_id=id_pairs[0]
    )
    signal_instance.compute_row_statistics()
    signal_instance.get_seeds()
    maximum_gap = 20

    max_vertical_gap = 1
    drift_padding_gap = 2
    min_neighbours = 4

    vertical_pruned_mask = signal_instance._prune_vertical(
        max_vertical_gap=max_vertical_gap,
        drift_padding_gap=maximum_gap,   
        min_neighbours=min_neighbours,
    )

    vertical_signal = signal_instance._return_signal_masked(vertical_pruned_mask)
    print(f"{vertical_pruned_mask.sum()=}")
    
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 6))
    cax1 = ax[0].imshow(
        signal_instance.signal_snippet, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    fig.colorbar(
        cax1, ax=ax[0], label="Signal Intensity"
    )  # colorbar for signal intensity
    ax[0].set_title("Original Signal Snippet")
    # cax2 = ax[1].imshow(
    #     vertical_signal, aspect="auto", cmap="viridis", interpolation="nearest"
    # )
    # fig.colorbar(
    #     cax2, ax=ax[1], label="Pruned Seed Mask"
    # )  # colorbar for signal intensity
    # ax[1].set_title(f"Vertically Pruned Seed Mask (min_neighbours={min_neighbours})")
    # plt.tight_layout()
    # plt.show()

    # merge horizontal and vertical pruning


    ################################################################################

    horizontal_pruned_mask = signal_instance._prune_horizontal(
        max_horizontal_gap=maximum_gap
    )

    horizontal_signal = signal_instance._return_signal_masked(horizontal_pruned_mask)

    # print(f"{horizontal_pruned_mask.sum()=}")
    # cax3 = ax[2].imshow(
    #     horizontal_signal, aspect="auto", cmap="viridis", interpolation="nearest"
    # )
    # fig.colorbar(
    #     cax3, ax=ax[2], label="Pruned Seed Mask"
    # )  # colorbar for signal intensity
    # ax[2].set_title(f"Horizontally Pruned Seed Mask (max_horizontal_gap={maximum_gap})")


    ################################################################################

    merged_pruned_mask, hor_mask_2d, ver_mask_2d = signal_instance.prune_methods(
        gate_requirment_for_false="and",
        max_horizontal_gap=maximum_gap,
        max_vertical_gap=max_vertical_gap,
        drift_padding_gap=maximum_gap,
        min_neighbours=min_neighbours,
        return_masks=True,
    )

    print(f"{merged_pruned_mask.sum()=}")
    print(f"{hor_mask_2d.sum()=}")
    print(f"{ver_mask_2d.sum()=}")

    pruned_signal = signal_instance._return_signal_masked(merged_pruned_mask)

    cax4 = ax[1].imshow(
        pruned_signal, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    fig.colorbar(
        cax4, ax=ax[1], label="Pruned Seed Mask"
    )  # colorbar for signal intensity

    ax[1].set_title(f"Merged Pruned Seed Mask (AND)")


    # consolidate seeds
    consolidated_seeds = signal_instance.consolidate_seeds(
        max_pixel_distance_either_side=maximum_gap
    )
    print(f"{consolidated_seeds.sum()=}")
    consolidated_signal = signal_instance._return_signal_masked(consolidated_seeds)
    cax5 = ax[2].imshow(
        consolidated_signal, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    fig.colorbar(
        cax5, ax=ax[2], label="Consolidated Seed Mask"
    )  # colorbar for signal intensity
    ax[2].set_title(f"Consolidated Seed Mask (max_pixel_distance_either_side={maximum_gap})")
    # plot grown seeds

    grown_seeds = signal_instance.grow_seeds(
        growth_threshold_above_median=0
    )

    grown_signal = signal_instance._return_signal_masked(grown_seeds)

    print(f"{grown_seeds.sum()=}")
    print(f"{grown_signal.sum()=}")
    cax6 = ax[3].imshow(
        grown_signal, aspect="auto", cmap="viridis", interpolation="nearest"
    )

    fig.colorbar(
        cax6, ax=ax[3], label="Grown Seed Signals"
    )  # colorbar for signal intensity
    ax[3].set_title(f"Grown Seed Mask (growth_threshold_above_median=1.0)")

    # plot diff between grown seeds and initial signal

    diff_signal = testing_snippet - grown_signal

    cax7 = ax[4].imshow(
        diff_signal, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    fig.colorbar(
        cax7, ax=ax[4], label="Diff Signal"
    )  # colorbar for signal intensity
    ax[4].set_title(f"Diff between Original Signal and Grown Seed Signal")

    print(f"diff between gronw seeds and consolidated seeds: {(grown_seeds != consolidated_seeds).sum()}")

    plt.tight_layout()
    plt.show()


    ################################################################################

    # plot the consolidated seeds using plot_1D



    signal_instance.plot_1D(
        mask_type="consolidated",
        save_location=None,
        show_plot_bool=True,
    )
    signal_instance.plot_1D(
        mask_type="grown",
        save_location=None,
        show_plot_bool=True,
    )

    signal_instance.plot_2D(
        mask_type="grown",
        save_location=None,
        show_plot_bool=True,
    )





    








    # signal_instance.consolidate_seeds(max_pixel_distance_either_side=maximum_gap)

    # consolidated, shape = signal_instance._return_consolidated_mask()
    # print(f"{consolidated=}")
    # print(f"{shape=}")
    # print(f"{consolidated.sum()=}")

    # plot initial seeds
    # signal_instance.plot_1D(
    #     nothing_initial_or_consolidated="consolidated",
    #     save_location=None,
    #     show_plot_bool=True,
    # )
    # # plot on 2d
    # signal_instance.plot_2D(
    #     nothing_initial_or_consolidated="consolidated",
    #     save_location=None,
    #     show_plot_bool=True,
    # )

    # print(f"{test_dataset.shape=}, {test_labels.shape=}, {test_signal_only.shape=}")
    # print(f"{test_labels[0:10]=}")
    # fig, axs = plt.subplots(nrows= 2, ncols=2, figsize=(10, 6))
    # for i in range(2):
    #     axs[i,0].imshow(test_dataset[i], cmap='viridis', aspect='auto')
    #     axs[i,1].imshow(test_signal_only[i], cmap='viridis', aspect='auto')
    #     axs[i,0].set_ylabel(f"Snippet {i}")
    #     axs[i,0].set_xlabel("Column Index")
    #     # add subtitles
    #     axs[i,1].set_xlabel(f"Column Index for label {test_labels[i]}, unique_id: {id_pairs[i]}")
    # axs[0,0].set_title("Raw Data")
    # axs[0,1].set_title("Signal Only Data")

    # plt.tight_layout()
    # plt.show()
