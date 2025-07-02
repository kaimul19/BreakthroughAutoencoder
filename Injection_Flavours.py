

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

    # Random starting frequency index
    start = int(rd.random() * (bin_width - 1)) + 1
    RandMultiplier = rd.choice([-1, 1])  # Randomly choose the drift direction
    y_points = np.array((0, 15, 31, 47, 63, 79)) # Y-points for the 6 cadences

    # Calculate max slope based on the starting frequency
    if RandMultiplier == -1:
        MaxSlope = -(96 / start) * (18.25361108/2.7939677238464355)
    else:
        spaceToEnd = bin_width - start
        MaxSlope = (96 / spaceToEnd) * (18.25361108/2.7939677238464355)

    # Generate slope and drift
    drift = (1 / MaxSlope) * rd.random()
    drift_factor = 2.7939677238464355/18.25361108

    # Calculate the x-coordinates for the 6 cadences
    x_starts = y_points * drift + start


    # Indices of the cadences where the signal will be injected
    indexes = [0, 2, 4]  # Inject signals into these cadences
    intensity = cadence[0].get_intensity(snr=60)  # Signal intensity based on the first cadence

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