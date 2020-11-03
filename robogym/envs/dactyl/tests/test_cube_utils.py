import numpy as np

import robogym.envs.dactyl.common.cube_utils as cube_utils
import robogym.utils.rotation as rotation


def test_align_quat_up():
    """ Test function 'align_quat_up' """
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

    assert (
        np.linalg.norm(cube_utils.align_quat_up(identity_quat) - identity_quat) < 1e-8
    )

    # Rotate along each axis but only slightly
    transformations = np.eye(3) * 0.4

    for i in range(3):
        quat = rotation.euler2quat(transformations[i])

        # For axes 0, 1 identity rotation is the proper rotation
        if i in [0, 1]:
            assert np.linalg.norm(cube_utils.align_quat_up(quat) - identity_quat) < 1e-8
        else:
            # For axis 2 cube is already aligned
            assert np.linalg.norm(cube_utils.align_quat_up(quat) - quat) < 1e-8

    # Rotate along each axis so much that another face is now on top
    transformations = np.eye(3) * (np.pi / 2 - 0.3)

    full_transformations = np.eye(3) * (np.pi / 2)

    for i in range(3):
        quat = rotation.euler2quat(transformations[i])
        aligned = cube_utils.align_quat_up(quat)

        if i in [0, 1]:
            new_euler_angles = rotation.quat2euler(aligned)

            assert np.linalg.norm(new_euler_angles - full_transformations[i]) < 1e-8

        else:
            # For axis 2 cube is already aligned
            assert np.linalg.norm(cube_utils.align_quat_up(quat) - quat) < 1e-8


def test_up_axis_with_sign():
    """ Test function 'up_axis_with_sign' """
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

    assert cube_utils.up_axis_with_sign(identity_quat) == (2, 1)

    # Rotate along each axis so much that another face is now on top
    transformations = np.eye(3) * (np.pi / 2 - 0.3)

    for i in range(3):
        quat = rotation.euler2quat(transformations[i])
        axis, sign = cube_utils.up_axis_with_sign(quat)

        if i == 0:
            assert axis == 1
            assert sign == 1
        elif i == 1:
            assert axis == 0
            assert sign == -1
        else:
            assert axis == 2
            assert sign == 1

    transformations = -np.eye(3) * (np.pi / 2 - 0.3)

    for i in range(3):
        quat = rotation.euler2quat(transformations[i])
        axis, sign = cube_utils.up_axis_with_sign(quat)

        if i == 0:
            assert axis == 1
            assert sign == -1
        elif i == 1:
            assert axis == 0
            assert sign == 1
        else:
            assert axis == 2
            assert sign == 1


def test_distance_quat_from_being_up():
    """ Test function 'distance_quat_from_being_up' """
    initial_configuration = np.array([1.0, 0.0, 0.0, 0.0])

    assert (
        np.linalg.norm(
            cube_utils.distance_quat_from_being_up(initial_configuration, 2, 1)
            - initial_configuration
        )
        < 1e-8
    )

    assert (
        np.abs(
            rotation.quat_magnitude(
                cube_utils.distance_quat_from_being_up(initial_configuration, 2, -1)
            )
            - np.pi
        )
        < 1e-8
    )

    assert (
        np.abs(
            rotation.quat_magnitude(
                cube_utils.distance_quat_from_being_up(initial_configuration, 0, 1)
            )
            - np.pi / 2
        )
        < 1e-8
    )

    assert (
        np.abs(
            rotation.quat_magnitude(
                cube_utils.distance_quat_from_being_up(initial_configuration, 0, -1)
            )
            - np.pi / 2
        )
        < 1e-8
    )

    assert (
        np.abs(
            rotation.quat_magnitude(
                cube_utils.distance_quat_from_being_up(initial_configuration, 1, 1)
            )
            - np.pi / 2
        )
        < 1e-8
    )

    assert (
        np.abs(
            rotation.quat_magnitude(
                cube_utils.distance_quat_from_being_up(initial_configuration, 1, -1)
            )
            - np.pi / 2
        )
        < 1e-8
    )

    # Rotate along each axis but only slightly
    transformations = np.eye(3) * 0.4

    for i in range(3):
        quat = rotation.euler2quat(transformations[i])
        distance_quat = cube_utils.distance_quat_from_being_up(quat, 2, 1)

        if i in [0, 1]:
            result = rotation.quat_mul(quat, distance_quat)
            assert np.linalg.norm(result - initial_configuration) < 1e-8
        else:
            assert np.linalg.norm(distance_quat - initial_configuration) < 1e-8

    transformations = np.eye(3) * (np.pi / 2 - 0.3)

    for i in range(3):
        quat = rotation.euler2quat(transformations[i])

        if i == 0:
            distance_quat = cube_utils.distance_quat_from_being_up(quat, 1, 1)
            assert np.abs(rotation.quat_magnitude(distance_quat) - 0.3) < 1e-8
        elif i == 1:
            distance_quat = cube_utils.distance_quat_from_being_up(quat, 0, -1)
            assert np.abs(rotation.quat_magnitude(distance_quat) - 0.3) < 1e-8
        else:
            distance_quat = cube_utils.distance_quat_from_being_up(quat, 2, 1)
            assert np.linalg.norm(distance_quat - initial_configuration) < 1e-8
