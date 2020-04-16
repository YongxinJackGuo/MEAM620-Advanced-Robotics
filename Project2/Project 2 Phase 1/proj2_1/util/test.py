# Imports
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

from proj2_1.code.estimate_pose_ransac import solve_w_t
from proj2_1.code.estimate_pose_ransac import find_inliers

from proj2_1.code.complementary_filter import complementary_filter_update


class TestBase(unittest.TestCase):
    def solve_w_t_test(self, fname):
        with open(fname, 'r') as file:

            print ('running : ' + fname)

            d = json.load(file)

            # Run a test
            R0 = Rotation.from_quat(d['initial_rotation'])
            uvd1 = np.array(d['uvd1'])
            uvd2 = np.array(d['uvd2'])

            output_w = np.array(d['output_w'])
            output_t = np.array(d['output_t'])

            w, t = solve_w_t(uvd1, uvd2, R0)

            self.assertTrue(norm(w.ravel() - output_w) < 1e-5, 'failed ' + fname)
            self.assertTrue(norm(t.ravel() - output_t) < 1e-5, 'failed ' + fname)

    def find_inliers_test(self, fname):
        with open(fname, 'r') as file:

            print ('running : ' + fname)

            d = json.load(file)

            # Run a test
            R0 = Rotation.from_quat(d['initial_rotation'])
            uvd1 = np.array(d['uvd1'])
            uvd2 = np.array(d['uvd2'])

            w = np.array(d['w'])
            t = np.array(d['t'])

            ransac_threshold = d['ransac_threshold']

            output_inliers = d['output_inliers']

            inliers = find_inliers(w, t, uvd1, uvd2, R0, ransac_threshold)

            self.assertTrue(all(inliers == output_inliers), 'failed ' + fname)

    def complementary_filter_update_test(self, fname):
        with open(fname, 'r') as file:

            print ('running : ' + fname)

            d = json.load(file)

            # Run a test
            R0 = Rotation.from_quat(d['initial_rotation'])
            w = np.array(d['angular_velocity'])
            a = np.array(d['linear_acceleration'])
            dt = d['dt']

            output_rotation = Rotation.from_quat(d['output_rotation'])

            rout = complementary_filter_update(R0, w, a, dt)

            temp = rout.inv() * output_rotation

            self.assertTrue(temp.magnitude() < 1e-4, 'failed ' + fname)

    # test complementary_filter_update
    def test_complementary_filter_update_00(self):
        self.complementary_filter_update_test('test_complementary_filter_00.json')

    def test_complementary_filter_update_01(self):
        self.complementary_filter_update_test('test_complementary_filter_01.json')

    def test_complementary_filter_update_02(self):
        self.complementary_filter_update_test('test_complementary_filter_02.json')

    def test_complementary_filter_update_03(self):
        self.complementary_filter_update_test('test_complementary_filter_03.json')

    def test_complementary_filter_update_04(self):
        self.complementary_filter_update_test('test_complementary_filter_04.json')


    # test solve_w_t
    def test_solve_w_t_00(self):
        self.solve_w_t_test('test_solve_w_t_00.json')

    def test_solve_w_t_01(self):
        self.solve_w_t_test('test_solve_w_t_01.json')

    def test_solve_w_t_02(self):
        self.solve_w_t_test('test_solve_w_t_02.json')

    def test_solve_w_t_03(self):
        self.solve_w_t_test('test_solve_w_t_03.json')

    def test_solve_w_t_04(self):
        self.solve_w_t_test('test_solve_w_t_04.json')

    # test find_inliers
    def test_find_inliers_00(self):
        self.find_inliers_test('test_find_inliers_00.json')

    def test_find_inliers_01(self):
        self.find_inliers_test('test_find_inliers_01.json')

    def test_find_inliers_02(self):
        self.find_inliers_test('test_find_inliers_02.json')

    def test_find_inliers_03(self):
        self.find_inliers_test('test_find_inliers_03.json')

    def test_find_inliers_04(self):
        self.find_inliers_test('test_find_inliers_04.json')


if __name__ == '__main__':
    unittest.main()
