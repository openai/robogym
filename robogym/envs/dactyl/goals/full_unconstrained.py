import typing

import numpy as np

from robogym.envs.dactyl.common import cube_utils
from robogym.goal.goal_generator import GoalGenerator
from robogym.utils import rotation


class FullUnconstrainedGoal(GoalGenerator):
    """
    Rotate any face, no orientation objectives for the Z axis.
    """

    def __init__(
        self,
        mujoco_simulation,
        success_threshold: dict,
        face_geom_names: typing.List[str],
        goal_directions: typing.Optional[typing.List[str]] = None,
        round_target_face: bool = True,
    ):
        """
        Create new FullUnconstrainedGoal object

        :param mujoco_simulation: A SimulationInterface object for a mujoco simulation considered
        :param success_threshold: Dictionary of threshold levels for cube orientation and face
            rotation, for which we consider the cube "aligned" with the goal
        :param face_geom_names: Names of 6 face geoms of the cube for which we measure the rotation
        :param goal_directions: Whether to rotate faces only clockwise, counterclockwise or both
        :param round_target_face: Whether target face rotations should be only round angles
            (multiplies of pi/2) or not
        :param p_face_flip: If the cube is aligned, what is the probability of flipping the cube
            vs rotating the face
        """
        super().__init__()

        assert len(face_geom_names) == 6, "Only supports full cube for now"

        self.mujoco_simulation = mujoco_simulation

        self.success_threshold = success_threshold
        self.face_geom_names = face_geom_names

        if goal_directions is None:
            self.goal_directions = ["cw", "ccw"]
        else:
            self.goal_directions = goal_directions

        self.round_target_face = round_target_face
        self.goal_candidates = list(range(len(self.face_geom_names)))

    def next_goal(self, random_state, current_state):
        """ Generate a new goal from current cube goal state """
        cube_face = current_state["cube_face_angle"]

        self.mujoco_simulation.clone_target_from_cube()
        self.mujoco_simulation.target_model.soft_align_faces()

        # Make the goal so that any face is rotated at random
        face_to_shift = random_state.choice(self.goal_candidates)

        # Rotate given face by a random angle and return both, new rotations and an angle
        goal_face, delta_angle = cube_utils.rotated_face_with_angle(
            cube_face,
            face_to_shift,
            random_state,
            self.round_target_face,
            directions=self.goal_directions,
        )

        self.mujoco_simulation.target_model.rotate_face(
            face_to_shift // 2, face_to_shift % 2, delta_angle
        )

        return {
            "cube_pos": np.zeros(3),
            "cube_quat": np.zeros(4),
            "cube_face_angle": goal_face,
            "goal_type": "rotation",
        }

    def current_state(self):
        """ Extract current cube goal state """
        return {
            "cube_pos": self.mujoco_simulation.get_qpos("cube_position"),
            "cube_quat": self.mujoco_simulation.get_qpos("cube_rotation"),
            "cube_face_angle": self.mujoco_simulation.get_face_angles("cube"),
        }

    def relative_goal(self, goal_state, current_state):
        """
        Calculate a difference in the 'goal space' between current state and the target goal
        """
        assert goal_state["goal_type"] == "rotation"

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
            "cube_quat": 0.0,
            "cube_face_angle": np.linalg.norm(relative_goal["cube_face_angle"]),
        }

        return goal_distance

    def goal_types(self) -> typing.Set[str]:
        return {"rotation"}
