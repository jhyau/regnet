import numpy as np

def get_im_overlay(indexes, width, height):
    """TODO"""
    im = np.zeros((height, width))
    im[:,indexes] = 1.
    return im

def count_peaks(spectogram, window_size=3, floor=30., delta=0.):
    """Count the peaks and their start point in |spectogram|.

    Args:
        spectogram: 2d array of (mel bins, time dim)
        window_size: how many samples backwards and and forward to look
        floor: intensity under which to "ignore" peaks
        delta: how large the delta between windows should be

    Returns:
        tuple: number of peaks, list of the center index
    """ 
    m = np.ones(window_size)
    total_intensities = np.sum(np.exp(spectogram - np.min(spectogram)), axis=0)
    # total_intensities = np.sum(spectogram, axis=0)
    # try out percentage based flooring
    floor = np.max(total_intensities) * 0.02
    # Floor out the poor ones.
    # total_intensities[total_intensities < floor] = 0.
    # Let's make an array of whether a[i] > a[i-1]
    window_intensities = np.convolve(total_intensities, m, mode='valid')
    # floor = np.max(window_intensities) * 0.50
    indexes = []
    for idx in range(1, len(window_intensities)-1):
        if window_intensities[idx-1] + delta < window_intensities[idx]:
            if window_intensities[idx+1] + delta < window_intensities[idx]:
                # The offset is necessary since that's where the original
                # conv window was centered.
                ridx = idx + window_size // 2 - 1
                if total_intensities[ridx] > floor:
                    indexes.append(ridx)
    count = len(indexes)
    return count, indexes
