import unittest

import aligner

class TestAligner(unittest.TestCase):

  def setUp(self):
    self.frames = 1024
    self.window_size = 64

  def tearDown(self):
    pass

  def test_regular(self):
    """Test regular misalignment works."""
    start, misaligned_start = aligner.get_misaligned_starts(self.frames,
                                                            self.window_size)
    assert start >= 0
    assert start <= self.frames - self.window_size
    assert misaligned_start != start
    assert misaligned_start >= 0
    assert misaligned_start <= self.frames - self.window_size

  
  def test_overlap(self):
    """Test overlap_ok=False ensures that the windows don't overlap."""
    start, misaligned_start = aligner.get_misaligned_starts(self.frames,
                                                            self.window_size,
                                                            overlap_ok=False)
    first = min(start, misaligned_start)
    second = max(start, misaligned_start)
    assert first + self.window_size <= second