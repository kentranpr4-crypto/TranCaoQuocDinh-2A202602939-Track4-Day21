"""Small geometry checks for the CP2 projection implementation."""
import unittest

import numpy as np

from starter.kitti_io import load_calib
from starter.projection import cam_to_image, velo_to_cam


class ProjectionTest(unittest.TestCase):
    def test_synthetic_reference_point(self):
        calib = load_calib("data/synthetic/training/calib/000000.txt")
        camera = velo_to_cam(np.array([[10.0, 0.0, 0.0]]), calib)
        uv, depth, mask = cam_to_image(camera, calib.P2, (375, 1242, 3))
        self.assertAlmostEqual(depth[0], 9.7273, places=3)
        np.testing.assert_allclose(uv[0], [613.964, 175.007], atol=0.01)
        np.testing.assert_array_equal(mask, [True])

    def test_invalid_and_outside_points_keep_input_mask_order(self):
        projection = np.array([[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]])
        points = np.array([[0, 0, 10], [np.nan, 0, 10], [0, 0, -1],
                           [100, 0, 10], [1, 0, 10]], dtype=float)
        uv, depth, mask = cam_to_image(points, projection, (100, 100, 3))
        np.testing.assert_array_equal(mask, [True, False, False, False, True])
        np.testing.assert_allclose(uv, [[50, 50], [60, 50]])
        np.testing.assert_allclose(depth, [10, 10])


if __name__ == "__main__":
    unittest.main()
