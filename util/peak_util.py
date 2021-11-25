import numpy as np
from scipy.signal import find_peaks

def count_peaks(spectrogram, threshold_to_max=0.5):
    """Count the peaks and their start point in |spectrogram|.

    Args:
        spectrogram: 2d array of (mel bins, time dim)
        threshold_to_max: float, 0.0 < x < 1.0, determine the threshold

    Returns:
        tuple: number of peaks, list of the center index
    """ 
    h, w =  spectrogram.shape
    # We will only consider half the buckets. There can be impacts that have no
    # higher layers, but there are no impacts that lack the low frequencies as
    # well
    flat = np.sum(np.exp(spectrogram), axis=0)
    # Do not threshold across all, some peaks don't reach all mel buckets
    th = np.max(np.exp(spectrogram)) * threshold_to_max
    # flat[flat < np.max(flat) * threshold_to_max] = 0.
    # peaks, _ = find_peaks(flat, height=None, width=2)
    peaks, _ = find_peaks(flat, height=th, width=2)
    return len(peaks), np.array(peaks)

pred_color = 2 # Blue for predicted peaks
gt_color = 0 # Red for gt peaks

def peaks_diff_image(pred_peaks, peaks, height, width):
    """Generate a peak visualization to see how far off they are."""
    img = np.ones((height, width, 3))
    cutoff = height // 2
    # zero-out the peaks i.e. make them black.
    img[:cutoff,pred_peaks,:] = 0
    img[:cutoff,pred_peaks,pred_color] = 1.
    img[cutoff:,peaks,:] = 0.
    img[cutoff:,peaks,gt_color] = 1.
    return img
