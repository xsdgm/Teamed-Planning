"""Metric and evidence checks for the real-data conversion boundary."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulator.prepare_wildscenes import load_poses, prepare, read_scan, scan_pairs, timestamp_key


class ForestConversionTests(unittest.TestCase):
    def make_sequence(self, root, xyz, labels, quaternion=(0., 0., 0., 1.)):
        seq = root/'V-test'
        (seq/'Clouds').mkdir(parents=True)
        (seq/'Labels').mkdir()
        stamp = '1623377796.111739618'
        cp, lp = seq/'Clouds'/f'{stamp}.bin', seq/'Labels'/f'{stamp}.label'
        np.asarray(xyz, dtype='<f4').tofile(cp)
        np.asarray(labels, dtype='<i4').tofile(lp)
        qx, qy, qz, qw = quaternion
        with (seq/'poses3d.csv').open('w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(['', 'timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            writer.writerow(['2021-06-11 02:16:36', stamp, 100, -50, 20, qw, qx, qy, qz])
        return seq, stamp, cp, lp

    def test_pose_rotation_and_exact_timestamp(self):
        self.assertEqual(timestamp_key('1624325401.005615500'), timestamp_key('1624325401.0056155'))
        self.assertNotEqual(timestamp_key('1624325401.005615501'), timestamp_key('1624325401.005615500'))
        with tempfile.TemporaryDirectory() as d:
            seq, key, cp, lp = self.make_sequence(Path(d), [[1, 0, 0]], [1],
                quaternion=(0, 0, np.sqrt(.5), np.sqrt(.5)))
            poses = load_poses(seq/'poses3d.csv')
            self.assertEqual(list(poses), [key])
            xyz, _ = read_scan(key, cp, lp, poses)
            np.testing.assert_allclose(xyz, [[100, -49, 20]])
            self.assertEqual(len(scan_pairs(seq, poses)), 1)
            lp.unlink()
            with self.assertRaises(ValueError):
                scan_pairs(seq, poses)

    def test_ground_canopy_and_unknown_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Three ground observations per 0.5m cell; two occupied columns
            # contain either high canopy (free below) or a low trunk (blocked).
            xy = np.array([[x+.25, y+.25] for y in np.arange(0, 5, .5)
                           for x in np.arange(0, 5, .5)])
            ground = np.repeat(np.column_stack([xy, np.zeros(len(xy))]), 3, axis=0)
            xyz = np.concatenate([ground, np.tile([1.25, 1.25, 10.], (3, 1)),
                                  np.tile([3.25, 3.25, 1.], (3, 1))])
            labels = np.r_[np.ones(len(ground)), [12]*3, [13]*3]
            seq, _, _, _ = self.make_sequence(root, xyz, labels)
            metadata = prepare(seq, root/'out', interpolation=0.)
            with np.load(root/'out'/'forest_map.npz') as m:
                offset = m['native_origin_xy']
                def ij(x, y):
                    cell = np.floor((np.array([x+100, y-50])-offset)/.5).astype(int)
                    return tuple(cell[::-1])
                self.assertTrue(m['known'][ij(1.25, 1.25)])
                self.assertFalse(m['occupied'][ij(1.25, 1.25)])
                self.assertTrue(m['occupied'][ij(3.25, 3.25)])
                self.assertFalse(m['known'][ij(8.25, 8.25)])
            self.assertEqual(metadata['coordinate_scale'], 1.)
            self.assertEqual(metadata['scans'], 1)


if __name__ == '__main__':
    unittest.main()
