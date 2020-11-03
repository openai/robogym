import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose

from robogym.envs.dactyl.full_perpendicular import make_env
from robogym.utils import rotation


class TestRubikSolvers(unittest.TestCase):
    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

    NEGATIVE_SIDE = 0
    POSITIVE_SIDE = 1

    CW = 1
    CCW = -1

    # Apply B R U rotations to solved cube
    scrambles = [
        {
            "rotation": {
                "axis": Z_AXIS,
                "side": POSITIVE_SIDE,
                "angle": CW * np.pi / 2,
            },
            "recovery_flips": np.array([[1, 1, 1]]),
        },
        {
            "rotation": {
                "axis": X_AXIS,
                "side": POSITIVE_SIDE,
                "angle": CW * np.pi / 2,
            },
            "recovery_flips": np.array([[0, 0, 1]]),
        },
        {
            "rotation": {
                "axis": Y_AXIS,
                "side": POSITIVE_SIDE,
                "angle": CW * np.pi / 2,
            },
            "recovery_flips": np.array([[0, 1, 1]]),
        },
    ]

    def test_face_cube_solver(self):
        constants = {
            "goal_generation": "face_cube_solver",
            "num_scramble_steps": 3,
            "randomize_face_angles": False,
            "randomize": False,
        }
        env = make_env(constants=constants)
        unwrapped = env.unwrapped

        # start from deterministic straight qpos
        unwrapped.mujoco_simulation.set_qpos("cube_rotation", [1.0, 0.0, 0.0, 0.0])
        assert_allclose(
            unwrapped.mujoco_simulation.get_qpos("cube_rotation"), [1.0, 0.0, 0.0, 0.0]
        )

        current_face_rotations = np.zeros(6)

        for step in self.scrambles:
            rot = step["rotation"]
            unwrapped.mujoco_simulation.cube_model.rotate_face(
                rot["axis"], rot["side"], rot["angle"]
            )
            current_face_rotations[rot["axis"] * 2 + rot["side"]] += rot["angle"]

        # track remaining face rotations on cube
        assert_allclose(
            current_face_rotations, [0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2]
        )

        unwrapped.reset_goal_generation()

        steps_left = len(self.scrambles)

        for step in reversed(self.scrambles):
            # assert state before recovery flip
            _, reached, goal_info = env.unwrapped.goal_info()
            assert not reached
            assert goal_info["goal_reachable"]
            assert goal_info["goal_dist"]["steps_to_solve"] == steps_left
            goal = goal_info["goal"]
            assert goal["goal_type"] == "flip"

            # check if expected quat goal is met
            recovery_quat = rotation.apply_euler_rotations(
                unwrapped.mujoco_simulation.get_qpos("cube_rotation"),
                step["recovery_flips"],
            )
            assert_allclose(goal["cube_quat"], recovery_quat, atol=1e-8)
            assert_allclose(goal["cube_face_angle"], current_face_rotations)

            # apply target quat rotation to cube and recompute goal
            unwrapped.mujoco_simulation.set_qpos("cube_rotation", recovery_quat)
            unwrapped.update_goal_info()
            _, reached, info = unwrapped.goal_info()
            assert reached
            unwrapped.mujoco_simulation.forward()
            unwrapped.reset_goal()

            solution = step["rotation"]

            _, reached, goal_info = env.unwrapped.goal_info()
            assert not reached
            assert goal_info["goal_reachable"]
            assert goal_info["goal_dist"]["steps_to_solve"] == steps_left
            goal = goal_info["goal"]
            assert goal["goal_type"] == "rotation"
            assert goal["axis_nr"] == solution["axis"]
            assert goal["axis_sign"][0] == solution["side"]

            current_face_rotations[solution["axis"] * 2 + solution["side"]] -= solution[
                "angle"
            ]
            assert_allclose(goal["cube_face_angle"], current_face_rotations)

            # actually rotate cube in the opposite direction of the original rotation
            unwrapped.mujoco_simulation.cube_model.rotate_face(
                solution["axis"], solution["side"], -solution["angle"]
            )

            unwrapped.update_goal_info()
            _, reached, info = unwrapped.goal_info()
            assert reached
            unwrapped.mujoco_simulation.forward()
            unwrapped.reset_goal()

            steps_left -= 1

        assert steps_left == 0

    def test_release_cube_solver(self):
        constants = {
            "goal_generation": "release_cube_solver",
            "num_scramble_steps": 3,
            "randomize_face_angles": False,
            "randomize": False,
        }
        env = make_env(constants=constants)
        unwrapped = env.unwrapped

        # start from deterministic straight qpos
        unwrapped.mujoco_simulation.set_qpos("cube_rotation", [1.0, 0.0, 0.0, 0.0])
        assert_allclose(
            unwrapped.mujoco_simulation.get_qpos("cube_rotation"), [1.0, 0.0, 0.0, 0.0]
        )

        current_face_rotations = np.zeros(6)

        for step in self.scrambles:
            rot = step["rotation"]
            unwrapped.mujoco_simulation.cube_model.rotate_face(
                rot["axis"], rot["side"], rot["angle"]
            )
            current_face_rotations[rot["axis"] * 2 + rot["side"]] += rot["angle"]

        # track remaining face rotations on cube
        assert_allclose(
            current_face_rotations, [0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2]
        )

        unwrapped.reset_goal_generation()

        steps_left = len(self.scrambles)

        for step in reversed(self.scrambles):
            # assert state before recovery flip
            _, reached, goal_info = env.unwrapped.goal_info()
            assert not reached
            assert goal_info["goal_reachable"]
            assert goal_info["goal_dist"]["steps_to_solve"] == steps_left
            goal = goal_info["goal"]
            assert goal["goal_type"] == "flip"

            # check if expected quat goal is met
            recovery_quat = rotation.apply_euler_rotations(
                unwrapped.mujoco_simulation.get_qpos("cube_rotation"),
                step["recovery_flips"],
            )
            assert_allclose(goal["cube_quat"], recovery_quat, atol=1e-8)
            assert_allclose(goal["cube_face_angle"], current_face_rotations)

            # apply target quat rotation to cube and recompute goal
            unwrapped.mujoco_simulation.set_qpos("cube_rotation", recovery_quat)
            unwrapped.update_goal_info()
            _, reached, info = unwrapped.goal_info()
            assert reached
            unwrapped.mujoco_simulation.forward()
            unwrapped.reset_goal()

            solution = step["rotation"]

            _, reached, goal_info = env.unwrapped.goal_info()
            assert not reached
            assert goal_info["goal_reachable"]
            assert goal_info["goal_dist"]["steps_to_solve"] == steps_left
            goal = goal_info["goal"]
            assert goal["goal_type"] == "rotation"
            assert goal["axis_nr"] == solution["axis"]
            assert goal["axis_sign"][0] == solution["side"]

            current_face_rotations[solution["axis"] * 2 + solution["side"]] -= solution[
                "angle"
            ]
            assert_allclose(goal["cube_face_angle"], current_face_rotations)

            # actually rotate cube in the opposite direction of the original rotation
            unwrapped.mujoco_simulation.cube_model.rotate_face(
                solution["axis"], solution["side"], -solution["angle"]
            )

            unwrapped.update_goal_info()
            _, reached, info = unwrapped.goal_info()
            assert reached
            unwrapped.mujoco_simulation.forward()
            unwrapped.reset_goal()

            steps_left -= 1

        assert steps_left == 0
        _, _, info = unwrapped.goal_info()
        assert info["solved"]
        unwrapped.mujoco_simulation.forward()
        assert info["solved"]

    def test_unconstrained_cube_solver(self):
        constants = {
            "goal_generation": "unconstrained_cube_solver",
            "num_scramble_steps": 3,
            "randomize_face_angles": False,
            "randomize": False,
        }
        env = make_env(constants=constants)
        unwrapped = env.unwrapped

        # start from deterministic straight qpos
        unwrapped.mujoco_simulation.set_qpos("cube_rotation", [1.0, 0.0, 0.0, 0.0])
        assert_allclose(
            unwrapped.mujoco_simulation.get_qpos("cube_rotation"), [1.0, 0.0, 0.0, 0.0]
        )

        current_face_rotations = np.zeros(6)

        for step in self.scrambles:
            rot = step["rotation"]
            unwrapped.mujoco_simulation.cube_model.rotate_face(
                rot["axis"], rot["side"], rot["angle"]
            )
            current_face_rotations[rot["axis"] * 2 + rot["side"]] += rot["angle"]

        # track remaining face rotations on cube
        assert_allclose(
            current_face_rotations, [0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2]
        )

        unwrapped.reset_goal_generation()

        steps_left = len(self.scrambles)

        for step in reversed(self.scrambles):
            solution = step["rotation"]

            _, reached, goal_info = env.unwrapped.goal_info()
            assert not reached
            assert goal_info["goal_reachable"]
            assert goal_info["goal_dist"]["steps_to_solve"] == steps_left
            goal = goal_info["goal"]
            assert goal["goal_type"] == "rotation"

            current_face_rotations[solution["axis"] * 2 + solution["side"]] -= solution[
                "angle"
            ]
            assert_allclose(goal["cube_face_angle"], current_face_rotations)

            # actually rotate cube in the opposite direction of the original rotation
            unwrapped.mujoco_simulation.cube_model.rotate_face(
                solution["axis"], solution["side"], -solution["angle"]
            )

            unwrapped.update_goal_info()
            _, reached, info = unwrapped.goal_info()
            assert reached
            unwrapped.mujoco_simulation.forward()
            unwrapped.reset_goal()

            steps_left -= 1

        assert steps_left == 0


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("rot_direction", [-1, 1])
def test_unconstrained_cube_solver(axis, side, rot_direction):
    constants = {
        "goal_generation": "unconstrained_cube_solver",
        "num_scramble_steps": 0,
        "randomize_face_angles": False,
        "randomize": False,
    }
    env = make_env(constants=constants)
    unwrapped = env.unwrapped
    # Rotate each face and make sure goal generator is able to solve the cube in one step
    unwrapped.mujoco_simulation.cube_model.rotate_face(
        axis, side, np.pi / 2 * rot_direction
    )
    unwrapped.reset_goal_generation()
    _, _, goal_info = env.unwrapped.goal_info()
    assert goal_info["goal_reachable"]
    assert goal_info["goal_dist"]["steps_to_solve"] == 1
    goal = goal_info["goal"]
    assert goal["goal_type"] == "rotation"
    assert_allclose(goal["cube_face_angle"], np.zeros(6))
