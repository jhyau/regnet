import numpy as np

def get_misaligned_starts(num_frames, window_size, max_overlap=0):
    """Get indexes that are misaligned.

    Args:
      num_frames: total length of segment
      window_size: how large the window in |num_frames| should be
      min_overlap: the maximum number of overlapping frames that two sets might have
                   e.g. |10| would indicate that the last 10 or the first 10
                   frames of the misaligned window *might* overlap with the real
                   window

    Note: the return indexes guarantee that a window_size starting
        at that index can be retrieved

    Returns:
      tuple: start_index, tuple(start_index,..), where they are misaligned
    """
    # randomly pick a valid starting point so that it can still cover
    # the entire window size. This means picking a point that is at most
    # window_size away from num_frames
    assert num_frames >= window_size * 3, 'provide a window-size that fits 3x into the frames'
    last_valid_start_frame = num_frames - window_size
    start_frame = np.random.choice(np.arange(0,last_valid_start_frame))
    misaligned_choices = np.arange(0, last_valid_start_frame)
    # Need to remove the choices dictated by |start_frame| and |window_size|
    # NOTE: no need to check if |max_overlap| will put end before start or
    # whatever since that just leads to an empty array.
    start = start_frame + max_overlap
    end = start_frame + window_size - max_overlap
    mask = np.ones(last_valid_start_frame).astype(bool)
    mask[start:end] = False
    misaligned_choices = misaligned_choices[mask]
    misaligned_start_frame = np.random.choice(misaligned_choices)
    return start_frame, misaligned_start_frame
