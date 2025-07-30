import numpy as np              # np.array, arithmetic
import setigen as stg           # stg.constant_path, stg.Frame, …
from astropy import units as u  # u.Hz, u.s  (Astropy units)
from numba import njit          # JIT compilation for performance
from Decorators import TimeMeasure

# for welsh dragon
import random
from WelshFlagGeneration import generate_welsh_flag_array   # ← your original function



@njit(cache=True)
def initial_params_generator(bin_width=4096):
    # Random starting frequency index
    start = 1 + np.random.randint(0, bin_width - 1)
    rand_multiplier = 1 if np.random.randint(0, 2) else -1
    y_points = np.array((0, 15, 31, 47, 63, 79)) # Y-points for the 6 cadences

    # Calculate max slope based on the starting frequency
    if rand_multiplier == -1:
        max_slope = -(96 / start) * (18.25361108/2.7939677238464355)
    else:
        space_to_end = bin_width - start
        max_slope = (96 / space_to_end) * (18.25361108/2.7939677238464355)

    # Generate slope and drift
    drift = (1 / max_slope) * np.random.rand()
    drift_factor = 2.7939677238464355/18.25361108

    # Calculate the x-coordinates for the 6 cadences
    x_starts = y_points * drift + start

    return start, drift, drift_factor, x_starts

# @TimeMeasure
def add_linear(cadence, signal_params, true_or_false, bin_width=4096):
    """
    Add a linear signal to the "A" observations in the given cadence.

    Parameters:
    - cadence: An OrderedCadence object.
    - signal_params: Parameters for the signal injection (e.g., [f_start, drift_rate, snr]).
    - true_or_false: Boolean, determines if the signal is a true or false signal.
    - bin_width: Width of the frequency bins.

    Returns:
    - Updated OrderedCadence object.
    """
    # Generate initial parameters
    start, drift, drift_factor, x_starts = initial_params_generator(bin_width)
    signal_to_noise = signal_params[2]
    # Indices of the cadences where the signal will be injected
    indexes = [0, 2, 4]  # Inject signals into these cadences
    intensity = cadence[0].get_intensity(snr=signal_to_noise)  # Signal intensity based on the first cadence

    # Add the signal to the specified cadences
    for i in indexes:
        cadence[i].add_signal(
            stg.constant_path(
                f_start=cadence[i].get_frequency(index=int(x_starts[i])),
                drift_rate=drift * drift_factor * u.Hz / u.s,
            ),
            stg.constant_t_profile(level=intensity),
            stg.box_f_profile(width=80 * cadence[i].df * u.Hz),
            stg.constant_bp_profile(level=1),
        )

    # Add a false signal to the remaining cadences if true_or_false is False
    if not true_or_false:
        false_indexes = [1, 3, 5]  # Inject false signals into these cadences
        for i in false_indexes:
            cadence[i].add_signal(
                stg.constant_path(
                    f_start=cadence[i].get_frequency(index=int(x_starts[i])),
                    drift_rate=drift * drift_factor * u.Hz / u.s,
                ),
                stg.constant_t_profile(level=intensity),
                stg.box_f_profile(width=80 * cadence[i].df * u.Hz),
                stg.constant_bp_profile(level=1),
            )

    elif true_or_false != True:
        raise ValueError("true_or_false must be either True or False")

    return cadence



def add_sinusoid(cadence, signal_params, true_or_false, bin_width= 4096):

        # Generate initial parameters
    start, drift, drift_factor, x_starts = initial_params_generator(bin_width)
    signal_to_noise = signal_params[2]
    # Indices of the cadences where the signal will be injected
    indexes = [0, 2, 4]  # Inject signals into these cadences
    intensity = cadence[0].get_intensity(snr=signal_to_noise)  # Signal intensity based on the first cadence
    
    # Add the signal to the specified cadences
    for i in indexes:
        cadence[i].add_signal(stg.sine_path(f_start=cadence[i].get_frequency(index=int(x_starts[i])),
                                        drift_rate=drift * drift_factor * u.Hz / u.s,
                                        period=100*u.s,
                                        amplitude=200*u.Hz),
                          stg.constant_t_profile(level=intensity),
                          stg.box_f_profile(width=80*cadence[i].df * u.Hz),
                          stg.constant_bp_profile(level=1))
    # Add a false signal to the remaining cadences if true_or_false is False
    if not true_or_false:
        false_indexes = [1, 3, 5]  # Inject false signals into these cadences
        for i in false_indexes:
            cadence[i].add_signal(stg.sine_path(f_start=cadence[i].get_frequency(index=int(x_starts[i])),
                                            drift_rate=drift * drift_factor * u.Hz / u.s,
                                            period=100*u.s,
                                            amplitude=200*u.Hz),
                            stg.constant_t_profile(level=intensity),
                            stg.box_f_profile(width=80*cadence[i].df * u.Hz),
                            stg.constant_bp_profile(level=1))

    elif true_or_false != True:
        raise ValueError("true_or_false must be either True or False")

    return cadence
    

def add_welsh_dragon(
    cadence,
    signal_params,          # last element interpreted as SNR if present
    true_or_false: bool,
    bin_width: int = 4096,  # unused for this flavour but kept for API parity
    flag_path: str = "WelshFlag.npy",
):
    """
    Inject a Welsh-flag bitmap into selected frames of an OrderedCadence.
    Mirrors the add_linear/add_sinusoid API so the dispatcher can treat
    every flavour the same.
    """

    # 1) generate the bitmap (values 0–1, dtype float32)
    flag = generate_welsh_flag_array(flag_path)      # (h, w)
    h, w = flag.shape

    # 2) choose a random placement inside each frame
    F, T = cadence[0].data.shape                    # rows=freq, cols=time
    y0 = random.randint(0, F - h)
    x0 = random.randint(0, T - w)

    # 3) scale intensity according to requested SNR
    signal_to_noise = signal_params[2]
    intensity = cadence[0].get_intensity(snr=signal_to_noise)
    flag      = flag * (intensity / flag.max())

    # 4) decide which frames get the flag
    true_frames  = [0, 2, 4]
    false_frames = [1, 3, 5]
    for idx in true_frames:
        array_data = cadence[idx].get_data()          # direct view
        array_data[y0:y0 + h, x0:x0 + w] += flag
        cadence[idx].data[:] = array_data                         # write back
    
    if not true_or_false:
        false_frames = [1, 3, 5]
        for idx in false_frames:
            array_data = cadence[idx].get_data()          # direct view
            array_data[y0:y0 + h, x0:x0 + w] += flag
            cadence[idx].data[:] = array_data                         # write back

    if true_or_false not in (True, False):
        raise ValueError("true_or_false must be either True or False")

    return cadence




