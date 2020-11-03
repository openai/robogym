import logging
import typing

import numpy as np

from robogym.envs.dactyl.common import cube_utils
from robogym.envs.dactyl.goals.rubik_cube_solver import RubikCubeSolver
from robogym.utils import rotation

logger = logging.getLogger(__name__)


class FaceCubeSolverGoal(RubikCubeSolver):
    """
    Generates a series of goals to solve a Rubik's cube.
    Goals are generated in a way to always rotate the top face.
    """

    def __init__(
        self,
        mujoco_simulation,
        success_threshold: typing.Dict[str, float],
        face_geom_names: typing.List[str],
        num_scramble_steps: int,
    ):
        """
        Create new FaceCubeSolverGoalGenerator object

        :param success_threshold: Dictionary of threshold levels for cube orientation and face
            rotation, for which we consider the cube "aligned" with the goal
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
        face_up = cube_utils.face_up(self.mujoco_simulation.sim, self.face_geom_names)
        goal_face_idx = self._get_goal_action().face_idx

        face_diff = rotation.normalize_angles(current_face_state - self.goal_face_state)
        return (
            face_up == goal_face_idx and np.linalg.norm(face_diff, axis=-1) < threshold
        )

    def face_threshold(self):
        return self.success_threshold["cube_face_angle"]

    def next_goal(self, random_state, current_state):
        """ Generate a new goal from current cube goal state """
        cube_pos = current_state["cube_pos"]
        cube_quat = current_state["cube_quat"]
        cube_face = current_state["cube_face_angle"]

        # Success threshold parameters
        face_threshold = self.face_threshold()
        rot_threshold = self.success_threshold["cube_quat"]

        rounded_current_face = rotation.round_to_straight_angles(cube_face)

        # Face aligned - are faces in the current cube aligned within the threshold
        current_face_diff = rotation.normalize_angles(cube_face - rounded_current_face)
        face_aligned = np.linalg.norm(current_face_diff, axis=-1) < face_threshold

        # Z aligned - is there a cube face looking up within the rotation threshold
        z_aligned = rotation.rot_xyz_aligned(cube_quat, rot_threshold)

        axis_nr, axis_sign = cube_utils.up_axis_with_sign(cube_quat)

        cube_aligned = face_aligned and z_aligned

        # Check if current state already meets goal state.
        if cube_aligned and self._is_goal_met(cube_face, face_threshold):
            # Step forward in goal sequence to get next goal.
            self._step_goal()

        goal_action = self._get_goal_action()

        if cube_aligned:
            # Choose index from the geoms that is highest on the z axis
            face_to_shift = cube_utils.face_up(
                self.mujoco_simulation.sim, self.face_geom_names
            )

            # Rotate face if the face to rotate for next goal is facing up.
            rotate_face = face_to_shift == goal_action.face_idx
        else:
            rotate_face = False

        if rotate_face:
            self.mujoco_simulation.target_model.rotate_face(
                face_to_shift // 2, face_to_shift % 2, goal_action.face_angle
            )

            goal_quat = cube_utils.align_quat_up(cube_quat)
            goal_face = self.goal_face_state
        else:  # need to flip cube

            # Rotate cube so that goal face is on the top. We currently apply
            # a deterministic transformation here that would get the goal face to the top,
            # which is _not_ the minimal possible orientation change, which may be
            # worth addressing in the future.
            goal_quat = self.goal_quat_for_face[goal_action.face_idx]

            # No need to rotate face, just align them.
            goal_face = rounded_current_face

        goal_quat = rotation.quat_normalize(goal_quat)

        return {
            "cube_pos": cube_pos,
            "cube_quat": goal_quat,
            "cube_face_angle": goal_face,
            "goal_type": "rotation" if rotate_face else "flip",
            "axis_nr": axis_nr,
            "axis_sign": axis_sign,
        }

    def relative_goal(self, goal_state, current_state):
        """
        Calculate a difference in the 'goal space' between current state and the target goal
        """
        if goal_state["goal_type"] == "rotation":
            orientation_distance = cube_utils.distance_quat_from_being_up(
                current_state["cube_quat"],
                goal_state["axis_nr"],
                goal_state["axis_sign"],
            )
        elif goal_state["goal_type"] == "flip":
            orientation_distance = rotation.quat_difference(
                goal_state["cube_quat"], current_state["cube_quat"]
            )
        else:
            raise ValueError('unknown goal_type "{}"'.format(goal_state["goal_type"]))

        return {
            # Cube pos does not count
            "cube_pos": np.zeros(goal_state["cube_pos"].shape),
            # Quaternion difference of a rotation
            "cube_quat": orientation_distance,
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
            "cube_quat": rotation.quat_magnitude(relative_goal["cube_quat"]),
            "cube_face_angle": np.linalg.norm(relative_goal["cube_face_angle"]),
            "steps_to_solve": len(self.goal_sequence)
            - (self.goal_step % len(self.goal_sequence)),
            "goal_step": self.goal_step,
        }

        return goal_distance

    def goal_reachable(self, goal_state, current_state):
        """ Check if goal is in reach from current state."""
        relative_goal = self.relative_goal(goal_state, current_state)
        face_rotation_angles = relative_goal["cube_face_angle"]
        goal_type = goal_state["goal_type"]
        eps = 1e-6

        rounded_rotation_angles = rotation.round_to_straight_angles(
            np.abs(rotation.normalize_angles(face_rotation_angles))
        )

        rotated_faces = list(np.where(rounded_rotation_angles > eps)[0])

        if goal_type == "rotation":
            # When doing face rotation, three conditions should met:
            # 1. Goal face should face up
            # 2. Rounded rotation angle for goal face should be at most 90 degree.
            # 3. Rounded rotation angle for other faces should be 0.
            goal_face_idx = self._get_goal_action().face_idx
            face_up = cube_utils.face_up(
                self.mujoco_simulation.sim, self.face_geom_names
            )

            return (
                goal_face_idx == face_up
                and rounded_rotation_angles[goal_face_idx] < np.pi / 2 + eps
                and rotated_faces in ([], [goal_face_idx])
            )
        elif goal_type == "flip":
            # When doing flipping, rounded rotation angles should be 0.
            return len(rotated_faces) == 0
        else:
            raise ValueError('unknown goal_type "{}"'.format(goal_type))
