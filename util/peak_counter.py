import numpy as np

def count_peaks(spectogram, window_size=3, floor=30.):
    """Count the peaks and their start point in |spectogram|.

    Args:
        spectogram: 2d array of (mel bins, time dim)
        window_size: how many samples backwards and and forward to look
        floor: intensity under which to "ignore" peaks

    Returns:
        tuple: number of peaks, list of the center index
    """ 
    # This works by looking at the summed intensity and finding
    # peaks, defined as: the last |window_size| samples monotonically increase
    # and the next |window_size| samples monotonically decrease
    total_intensities = np.sum(np.exp(spectogram), axis=0)
    # Floor out the poor ones.
    total_intensities[total_intensities < floor] = 0.
    # Let's make an array of whether a[i] > a[i-1]
    decreasing = np.zeros(shape=total_intensities.shape)
    for i in np.arange(1, len(total_intensities)):
        if total_intensities[i-1] > total_intensities[i]:
            decreasing[i-1] = 1
    # Now, let's sum all the numbers up to when we see a 0.
    # TODO(coconutruben): this does only the forward pass
    # we might want to implement the backwards pass (monotonically decreasing)
    # as well
    up_to = np.zeros(shape=total_intensities.shape)
    for i in np.arange(len(total_intensities)-1, 0, -1):
        if decreasing[i] == 1:
            up_to[i] += 1 + up_to[i+1]
            up_to[i+1] = 0
    # Now, we can simply look for the window-size numbers
    indexes = np.argwhere(up_to > window_size)
    count = len(indexes)
    return count, indexes.ravel()
