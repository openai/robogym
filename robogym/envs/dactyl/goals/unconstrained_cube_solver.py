import logging
import typing

import numpy as np

from robogym.envs.dactyl.goals.rubik_cube_solver import RubikCubeSolver
from robogym.utils import rotation

logger = logging.getLogger(__name__)


class UnconstrainedCubeSolver(RubikCubeSolver):
    """
    Generates a series of goals to solve a Rubik's cube.
    Goals are not constrained to apply to a particular face.
    """

    def __init__(
        self,
        mujoco_simulation,
        success_threshold: typing.Dict[str, float],
        face_geom_names: typing.List[str],
        num_scramble_steps: int,
    ):
        """
        Creates new UnconstrainedCubeSolver object
        """
        self.success_threshold = success_threshold
        super().__init__(
            mujoco_simulation=mujoco_simulation,
            face_geom_names=face_geom_names,
            num_scramble_steps=num_scramble_steps,
        )

    def _is_goal_met(self, current_face_state, threshold):
        """
        Check if current face state matches current goal state.
        """

        face_diff = rotation.normalize_angles(current_face_state - self.goal_face_state)
        return np.linalg.norm(face_diff, axis=-1) < threshold

    def next_goal(self, random_state, current_state):
        """ Generates a new goal from current cube goal state """
        cube_pos = current_state["cube_pos"]
        cube_quat = current_state["cube_quat"]
        cube_face = current_state["cube_face_angle"]

        # Success threshold parameters
        face_threshold = self.success_threshold["cube_face_angle"]

        # Check if current state already meets goal state.
        if self._is_goal_met(cube_face, face_threshold):
            # Step forward in goal sequence to get next goal.
            self._step_goal()

        # Directly rotate the face indicated by the goal action.
        goal_action = self._get_goal_action()
        face_to_shift = goal_action.face_idx

        self.mujoco_simulation.target_model.rotate_face(
            face_to_shift // 2, face_to_shift % 2, goal_action.face_angle
        )

        # align cube quat for visualization purposes, has no effect on goal being met
        cube_quat = rotation.quat_normalize(rotation.round_to_straight_quat(cube_quat))

        return {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "cube_face_angle": self.goal_face_state,
            "goal_type": "rotation",
        }

    def relative_goal(self, goal_state, current_state):
        """
        Calculate a difference in the 'goal space' between current state and the target goal
        """
        goal_type = goal_state["goal_type"]
        assert goal_type == "rotation", 'unknown goal_type "{}"'.format(goal_type)

        return {
            # Cube pos does not count
            "cube_pos": np.zeros(goal_state["cube_pos"].shape),
            # Quaternion difference of a rotation
            "cube_quat": np.zeros(goal_state["cube_quat"].shape),
            # Angle differences
            "cube_face_angle": rotation.normalize_angles(
                goal_state["cube_face_angle"] - current_state["cube_face_angle"]
            ),
        }

    def goal_distance(self, goal_state, current_state):
        """ Distance from the current goal to the target state. """
        relative_goal = self.relative_goal(goal_state, current_state)

        goal_distance = {
            "cube_pos": 0.0,
            "cube_quat": 0.0,  # qpos has no effect on whether we consider goal achieved
            "cube_face_angle": np.linalg.norm(relative_goal["cube_face_angle"]),
            "steps_to_solve": len(self.goal_sequence)
            - (self.goal_step % len(self.goal_sequence)),
        }

        return goal_distance

    def goal_reachable(self, goal_state, current_state):
        """ Check if goal is in reach from current state."""
        relative_goal = self.relative_goal(goal_state, current_state)
        face_rotation_angles = relative_goal["cube_face_angle"]
        goal_type = goal_state["goal_type"]
        assert goal_type == "rotation", 'unknown goal_type "{}"'.format(goal_type)
        eps = 1e-6

        rounded_rotation_angles = rotation.round_to_straight_angles(
            np.abs(rotation.normalize_angles(face_rotation_angles))
        )

        rotated_faces = list(np.where(rounded_rotation_angles > eps)[0])

        goal_face_idx = self._get_goal_action().face_idx

        return rounded_rotation_angles[
            goal_face_idx
        ] < np.pi / 2 + eps and rotated_faces in ([], [goal_face_idx])
