import numpy as np
import pycuber

import robogym.utils.rotation as rotation
from robogym.envs.dactyl.full_perpendicular import FullPerpendicularSimulation

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2


NEGATIVE_SIDE = 0
POSITIVE_SIDE = 1


def _full_side_idx(axis, side):
    # DRIVER ORDER IS:
    # -x, +x, -y, +y, -z, +z
    return axis * 2 + side


def test_cube_manipulator_drivers():
    """
    Test CubeManipulator class if it manages to manipulates cubelets properly
    """
    mujoco_simulation = FullPerpendicularSimulation.build(n_substeps=10)

    for axis in [X_AXIS, Y_AXIS, Z_AXIS]:
        for side in [NEGATIVE_SIDE, POSITIVE_SIDE]:
            # Reset simulation
            mujoco_simulation.set_qpos("cube_all_joints", 0.0)

            mujoco_simulation.cube_model.rotate_face(axis, side, np.pi / 2)

            target_angle = np.zeros(6, dtype=float)

            target_angle[_full_side_idx(axis, side)] = np.pi / 2

            assert (
                np.linalg.norm(
                    mujoco_simulation.get_qpos("cube_drivers") - target_angle
                )
                < 1e-6
            )


def test_cube_manipulator_drivers_sequence():
    """
    Test CubeManipulator class if it manages to manipulate cubelets properly
    """
    mujoco_simulation = FullPerpendicularSimulation.build(n_substeps=10)

    mujoco_simulation.cube_model.rotate_face(X_AXIS, POSITIVE_SIDE, np.pi / 2)
    mujoco_simulation.cube_model.rotate_face(Y_AXIS, POSITIVE_SIDE, np.pi / 2)
    mujoco_simulation.cube_model.rotate_face(Z_AXIS, POSITIVE_SIDE, np.pi / 2)
    mujoco_simulation.cube_model.rotate_face(Y_AXIS, POSITIVE_SIDE, np.pi / 2)
    mujoco_simulation.cube_model.rotate_face(X_AXIS, POSITIVE_SIDE, np.pi / 2)

    target_angle = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi / 2])

    assert (
        np.linalg.norm(mujoco_simulation.get_qpos("cube_drivers") - target_angle) < 1e-6
    )


POSSIBLE_COORDS = [-1, 0, 1]


def _assert_cubelet_coords(manipulator, original_coords, current_coords):
    """ Check if given cubelet is present at given coords"""
    indexes = original_coords.round().astype(int) + 1
    coord_idx = indexes[0] * 9 + indexes[1] * 3 + indexes[2]
    meta_info = manipulator.cubelet_meta_info[coord_idx]

    assert np.linalg.norm(meta_info["coords"] - original_coords) < 1e-6

    if meta_info["type"] == "cubelet":
        mtx = manipulator._cubelet_rotation_matrix(meta_info, manipulator.sim.data.qpos)

        actual_current_coords = mtx @ original_coords.astype(float)

        assert np.linalg.norm(actual_current_coords - current_coords) < 1e-6


def test_cube_manipulator_cubelet_positions():
    """
    Test CubeManipulator class if it manages to manipulates cubelets properly
    """

    mujoco_simulation = FullPerpendicularSimulation.build(n_substeps=10)

    for x_coord in POSSIBLE_COORDS:
        for y_coord in POSSIBLE_COORDS:
            for z_coord in POSSIBLE_COORDS:
                coords = np.array([x_coord, y_coord, z_coord])
                _assert_cubelet_coords(mujoco_simulation.cube_model, coords, coords)

    mujoco_simulation.cube_model.rotate_face(X_AXIS, POSITIVE_SIDE, np.pi / 2)

    # These are not touched
    for x_coord in [0, -1]:
        for y_coord in POSSIBLE_COORDS:
            for z_coord in POSSIBLE_COORDS:
                coords = np.array([x_coord, y_coord, z_coord])
                _assert_cubelet_coords(mujoco_simulation.cube_model, coords, coords)

    # Let's check four corner cubelets just to be sure
    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, 1, 1]), np.array([1, -1, 1])
    )
    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, -1, 1]), np.array([1, -1, -1])
    )
    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, -1, -1]), np.array([1, 1, -1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, 1, -1]), np.array([1, 1, 1])
    )

    mujoco_simulation.cube_model.rotate_face(Y_AXIS, POSITIVE_SIDE, np.pi / 2)

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([-1, 1, -1]), np.array([-1, 1, 1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([-1, 1, 1]), np.array([1, 1, 1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, 1, -1]), np.array([1, 1, -1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, -1, -1]), np.array([-1, 1, -1])
    )


def test_snap_rotate_face_with_threshold():
    from robogym.envs.dactyl.full_perpendicular import FullPerpendicularSimulation

    mujoco_simulation = FullPerpendicularSimulation.build(n_substeps=10)

    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        X_AXIS, POSITIVE_SIDE, np.pi / 2
    )

    # These are not touched
    for x_coord in [0, -1]:
        for y_coord in POSSIBLE_COORDS:
            for z_coord in POSSIBLE_COORDS:
                coords = np.array([x_coord, y_coord, z_coord])
                _assert_cubelet_coords(mujoco_simulation.cube_model, coords, coords)

    # Let's check four corner cubelets just to be sure
    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, 1, 1]), np.array([1, -1, 1])
    )
    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, -1, 1]), np.array([1, -1, -1])
    )
    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, -1, -1]), np.array([1, 1, -1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, 1, -1]), np.array([1, 1, 1])
    )

    # Rotate this face again by 45 degrees
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        X_AXIS, POSITIVE_SIDE, np.pi / 4
    )

    cubelets_before = mujoco_simulation.get_qpos("cube_cubelets").copy()

    # None of these should do anything
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        Y_AXIS, POSITIVE_SIDE, np.pi / 8
    )
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        Y_AXIS, NEGATIVE_SIDE, np.pi / 8
    )
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        Z_AXIS, POSITIVE_SIDE, np.pi / 8
    )
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        Z_AXIS, NEGATIVE_SIDE, np.pi / 8
    )

    cubelets_after = mujoco_simulation.get_qpos("cube_cubelets").copy()

    assert np.linalg.norm(cubelets_before - cubelets_after) < 1e-6

    # Revert
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        X_AXIS, POSITIVE_SIDE, -np.pi / 4
    )
    # Move a little
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        X_AXIS, POSITIVE_SIDE, 0.05
    )

    # Make sure cube gets realigned
    mujoco_simulation.cube_model.snap_rotate_face_with_threshold(
        Y_AXIS, POSITIVE_SIDE, np.pi / 2
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([-1, 1, -1]), np.array([-1, 1, 1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([-1, 1, 1]), np.array([1, 1, 1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, 1, -1]), np.array([1, 1, -1])
    )

    _assert_cubelet_coords(
        mujoco_simulation.cube_model, np.array([1, -1, -1]), np.array([-1, 1, -1])
    )

    cubelets_final = rotation.normalize_angles(
        mujoco_simulation.get_qpos("cube_cubelets").copy()
    )

    assert (
        np.linalg.norm(
            cubelets_final - rotation.round_to_straight_angles(cubelets_final)
        )
        < 1e-8
    )


def test_pycuber_conversion():
    from robogym.envs.dactyl.full_perpendicular import FullPerpendicularSimulation

    mujoco_simulation = FullPerpendicularSimulation.build()

    for i in range(5):
        cube = pycuber.Cube()

        for action in np.random.choice(list("LRFBDU"), size=20, replace=True):
            cube(str(action))

        mujoco_simulation.cube_model.from_pycuber(cube)
        cube2 = mujoco_simulation.cube_model.to_pycuber()

        assert cube == cube2
