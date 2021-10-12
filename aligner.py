import numpy as np

def get_misaligned_starts(num_frames, window_size, overlap_ok=True):
  """Get indexes that are misaligned.

  Args:
    num_frames: total length of segment
    window_size: how large the window in |num_frames| should be
    overlap_ok: whether the misalignment can have overlap
   
  Note: the return indexes guarantee that a window_size starting
      at that index can be retrieved

  Returns:
    tuple: start_index, start_index, where they are misaligned
  """
  # randomly pick a valid starting point so that it can still cover
  # the entire window size. This means picking a point that is at most
  # window_size away from num_frames
  last_valid_start_frame = num_frames - window_size
  start_frame = np.random.choice(np.arange(0,last_valid_start_frame))
  # assert overlap_ok, 'overlap_ok=False not implemented yet'
  misaligned_start_frame = start_frame
  frames_covered = set(np.arange(start_frame, start_frame + window_size))
  while misaligned_start_frame == start_frame:
    misaligned_start_frame = np.random.choice(np.arange(0,last_valid_start_frame))
    if overlap_ok is False:
      if misaligned_start_frame in frames_covered:
        # Check that it doesn't start in the window above
        # reset to start_frame to continue the loop
        misaligned_start_frame = start_frame
      if misaligned_start_frame + window_size - 1 in frames_covered:
        # Check that it doesn't end in the window above
        # reset to start_frame to continue the loop
        misaligned_start_frame = start_frame
  return start_frame, misaligned_start_frame
      

