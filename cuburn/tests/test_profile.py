import argparse
import unittest
import numpy as np

from cuburn import profile

class ProfileTest(unittest.TestCase):
    def _get_profile(self, args=None):
        args = args or []
        return profile.get_from_args(profile.add_args().parse_args(args))

    def test_enumerate_times(self):
        name, prof = self._get_profile()
        gprof = profile.wrap(prof, {"type":"edge"})
        frames = list(profile.enumerate_times(gprof))
        frame_times = np.linspace(0, 1 - 1/720., 720) + 0.5/720
        self.assertEquals(frames, list(enumerate(frame_times, 1)))

    def test_nframes_for_sharding_equal(self):
        name, prof = self._get_profile(
                ['-P', '720p', '--fps=1', '--duration=5', '--shard=5'])
        gprof = profile.wrap(prof, {"type":"edge"})
        frames = list(profile.enumerate_times(gprof))
        frame_times = np.linspace(0, 1 - 1/5., 5) + 0.5/5
        self.assertEquals(len(frames), 1)
        self.assertEquals(frames[0][0], 1)
        self.assertItemsEqual(frames[0][1], frame_times)
