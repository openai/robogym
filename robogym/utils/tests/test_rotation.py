import itertools as it
import unittest

import numpy as np
from mujoco_py import functions
from numpy.random import randint, uniform
from numpy.testing import assert_allclose
from scipy.linalg import inv, sqrtm
from transforms3d import euler, quaternions

from robogym.utils.rotation import (
    any_orthogonal,
    euler2mat,
    euler2quat,
    mat2euler,
    mat2quat,
    quat2euler,
    quat2mat,
    quat_average,
    quat_magnitude,
    quat_normalize,
    rot_xyz_aligned,
    vectors2quat,
)

N = 10  # Number of trials to run


def normalize_mat(mat):
    if np.abs(np.linalg.det(mat)) < 1e-10:
        raise ValueError("Matrix too close to singular")
    mat = np.real(mat.dot(inv(sqrtm(mat.T.dot(mat)))))
    if np.linalg.det(mat) < 0:
        mat *= -1
    return mat


def normalize_quat(quat):
    quat /= np.sqrt(np.sum(np.square(quat)))
    if quat[0] < 0:
        quat *= -1
    return quat


def random_unit_length_vec():
    v = np.random.randn(3) * 5

    while np.linalg.norm(v) < 1e-4:
        v = np.random.randn(3)

    return v / np.linalg.norm(v)


class RotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(112358)

    def test_euler2mat(self):
        s = (N, N, 3)
        eulers = uniform(-4, 4, size=s) * randint(2, size=s)
        mats = euler2mat(eulers)
        self.assertEqual(mats.shape, (N, N, 3, 3))
        for i in range(N):
            for j in range(N):
                res = euler.euler2mat(*eulers[i, j], axes="rxyz")
                np.testing.assert_almost_equal(mats[i, j], res)

    def test_euler2quat(self):
        s = (N, N, 3)
        eulers = uniform(-4, 4, size=s) * randint(2, size=s)
        quats = euler2quat(eulers)
        self.assertEqual(quats.shape, (N, N, 4))
        for i in range(N):
            for j in range(N):
                res = euler.euler2quat(*eulers[i, j], axes="rxyz")
                np.testing.assert_almost_equal(quats[i, j], res)

    def test_mat2euler(self):
        s = (N, N, 3, 3)
        mats = uniform(-4, 4, size=s) * randint(2, size=s)
        eulers = mat2euler(mats)
        self.assertEqual(eulers.shape, (N, N, 3))
        for i in range(N):
            for j in range(N):
                res = euler.mat2euler(mats[i, j], axes="rxyz")
                np.testing.assert_almost_equal(eulers[i, j], res)

    def test_mat2quat(self):
        s = (N, N, 3, 3)
        mats = uniform(-4, 4, size=s) * randint(2, size=s)
        quats = mat2quat(mats)
        self.assertEqual(quats.shape, (N, N, 4))
        for i in range(N):
            for j in range(N):
                # Compare to transforms3d
                res = quaternions.mat2quat(mats[i, j])
                np.testing.assert_almost_equal(quats[i, j], res)
                # Compare to MuJoCo
                try:
                    mat = normalize_mat(mats[i, j])
                except (np.linalg.linalg.LinAlgError, ValueError):
                    continue  # Singular matrix, NaNs
                res[:] = 0
                functions.mju_mat2Quat(res, mat.flatten())
                res = normalize_quat(res)
                quat = mat2quat(mat)
                # quat is the same rotation as -quat
                assert np.allclose(quat, res) or np.allclose(
                    -quat, res
                ), "quat {} res {}".format(quat, res)

    def test_quat2euler(self):
        s = (N, N, 4)
        quats = uniform(-1, 1, size=s) * randint(2, size=s)
        eulers = quat2euler(quats)
        self.assertEqual(eulers.shape, (N, N, 3))
        for i in range(N):
            for j in range(N):
                res = euler.quat2euler(quats[i, j], axes="rxyz")
                np.testing.assert_almost_equal(eulers[i, j], res)

    def test_quat2mat(self):
        s = (N, N, 4)
        quats = uniform(-1, 1, size=s) * randint(2, size=s)
        mats = quat2mat(quats)
        self.assertEqual(mats.shape, (N, N, 3, 3))
        for i in range(N):
            for j in range(N):
                # Compare to transforms3d
                res = quaternions.quat2mat(quats[i, j])
                np.testing.assert_almost_equal(mats[i, j], res)
                # Compare to MuJoCo
                quat = normalize_quat(quats[i, j])
                mat = np.zeros(9, dtype=np.float64)
                functions.mju_quat2Mat(mat, quat)
                if np.isnan(mat).any():
                    continue  # MuJoCo returned NaNs
                np.testing.assert_almost_equal(quat2mat(quat), mat.reshape((3, 3)))

    def test_mat2quat2euler2mat(self):
        s = (N, N, 3, 3)
        mats = uniform(-np.pi, np.pi, size=s) * randint(2, size=s)
        for i in range(N):
            for j in range(N):
                try:
                    mat = normalize_mat(mats[i, j])
                except:  # noqa
                    continue  # Singular Matrix or NaNs
                result = euler2mat(quat2euler(mat2quat(mat)))
                np.testing.assert_allclose(mat, result, atol=1e-8, rtol=1e-6)

    def test_mat2euler2quat2mat(self):
        s = (N, N, 3, 3)
        mats = uniform(-np.pi, np.pi, size=s) * randint(2, size=s)
        for i in range(N):
            for j in range(N):
                try:
                    mat = normalize_mat(mats[i, j])
                except:  # noqa
                    continue  # Singular Matrix or NaNs
                result = quat2mat(euler2quat(mat2euler(mat)))
                np.testing.assert_allclose(mat, result, atol=1e-8, rtol=1e-6)

    def test_quat_average(self):
        max_angle = 1.0
        euler1 = np.zeros(3)
        euler2 = np.array([max_angle, 0.0, 0.0])

        q1 = euler2quat(euler1)
        q2 = euler2quat(euler2)

        assert_allclose(q1, quat_average([q1]))

        mid_q = quat_average([q1, q2])
        assert_allclose(quat2euler(mid_q), [max_angle / 2.0, 0.0, 0.0])

        for weight in [0.0, 0.5, 1.0]:
            avg_q = quat_average([q1, q2], weights=[1 - weight, weight])
            assert_allclose(quat2euler(avg_q), [max_angle * weight, 0.0, 0.0])

    def test_quat_normalize(self):
        """ Test quaternion normalization """

        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])
        q3 = np.array([0.0, 1.0, 0.0, 0.0])

        assert np.linalg.norm(quat_normalize(q1) - q1) < 1e-8

        assert np.linalg.norm(quat_normalize(q2) + q2) < 1e-8

        assert np.linalg.norm(quat_normalize(q3) - q3) < 1e-8

        for q in [q1, q2, q3]:
            assert quat_normalize(q)[0] >= 0.0

    def test_any_orthogonal(self):
        """ Test finding any orthogonal vector to given """
        vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            random_unit_length_vec(),
            random_unit_length_vec(),
            random_unit_length_vec(),
        ]

        for v in vectors:
            orthogonal = any_orthogonal(v)

            # Vectors are indeed orthogonal
            assert np.abs(np.dot(v, orthogonal)) < 1e-8

            # orthogonal vector has unit length
            assert np.abs(np.linalg.norm(orthogonal) - 1) < 1e-8

    def test_vectors2quat(self):
        """ Test constructing quaternion from two given vectors """
        vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            -np.array([1, 0, 0]),
            -np.array([0, 1, 0]),
            -np.array([0, 0, 1]),
            random_unit_length_vec(),
            random_unit_length_vec(),
            random_unit_length_vec(),
        ]

        for v1, v2 in it.product(vectors, vectors):
            quat = vectors2quat(v1, v2)
            mat = quat2mat(quat)

            maybe_v2 = mat @ v1

            # Test that quat is normalized
            assert np.abs(np.linalg.norm(quat) - 1.0) < 1e-8

            # Make sure that given quaterion is the shortest path rotation
            # np.clip is necessary due to potential minor numerical instabilities
            assert quat_magnitude(quat) <= np.arccos(np.clip(v1 @ v2, -1.0, 1.0)) + 1e-8

            # Test that quaternion rotates input vector to output vector
            assert np.linalg.norm(maybe_v2 - v2) < 1e-8


def test_rot_xyz_aligned():
    """ Test function 'rot_xyz_aligned' """
    # Identity configuration
    initial_configuration = np.array([1.0, 0.0, 0.0, 0.0])

    # Cube is aligned in initial condition
    assert rot_xyz_aligned(initial_configuration, 0.01)

    # Rotate along each axis more than the threshold
    transformations = np.eye(3) * 0.5

    for i in range(3):
        quat = euler2quat(transformations[i])

        if i in [0, 1]:
            # For rotations along x,y cube is not aligned
            assert not rot_xyz_aligned(quat, 0.4)
        else:
            # Cube is aligned for rotation along z axis
            assert rot_xyz_aligned(quat, 0.4)

    # Rotate along each axis so much that the threshold is met again
    transformations = np.eye(3) * (np.pi / 2 - 0.3)

    for i in range(3):
        quat = euler2quat(transformations[i])

        # Cube is aligned again
        assert rot_xyz_aligned(quat, 0.4)
