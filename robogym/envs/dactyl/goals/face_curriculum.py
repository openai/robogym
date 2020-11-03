import typing

import numpy as np

from robogym.envs.dactyl.common import cube_utils
from robogym.goal.goal_generator import GoalGenerator
from robogym.utils import rotation


class FaceCurriculumGoal(GoalGenerator):
    """ 'Face curriculum' goal generation. Generate goals that specify a fully aligned cube at a
    desired orientation with the specified face being up.
    """

    def __init__(
        self,
        mujoco_simulation,
        success_threshold: dict,
        face_geom_names: typing.List[str],
        goal_directions: typing.Optional[typing.List[str]] = None,
        round_target_face: bool = True,
        p_face_flip: float = 0.25,
    ):
        """
        Create new FaceCurriculumGoal object

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

        assert len(face_geom_names) in {2, 6}, "Only supports full cube or face cube"

        self.mujoco_simulation = mujoco_simulation

        self.success_threshold = success_threshold
        self.face_geom_names = face_geom_names

        if goal_directions is None:
            self.goal_directions = ["cw", "ccw"]
        else:
            self.goal_directions = goal_directions

        self.round_target_face = round_target_face

        self.p_face_flip = p_face_flip

        self.goal_quat_for_face = cube_utils.face_up_quats(
            mujoco_simulation.sim, "cube:cube:rot", self.face_geom_names
        )

    def next_goal(self, random_state, current_state):
        """ Generate a new goal from current cube goal state """
        cube_pos = current_state["cube_pos"]
        cube_quat = current_state["cube_quat"]
        cube_face = current_state["cube_face_angle"]

        # Success threshold parameters
        face_threshold = self.success_threshold["cube_face_angle"]
        rot_threshold = self.success_threshold["cube_quat"]

        self.mujoco_simulation.clone_target_from_cube()
        self.mujoco_simulation.align_target_faces()

        rounded_current_face = rotation.round_to_straight_angles(cube_face)

        # Face aligned - are faces in the current cube aligned within the threshold
        current_face_diff = rotation.normalize_angles(cube_face - rounded_current_face)
        face_aligned = np.linalg.norm(current_face_diff, axis=-1) < face_threshold

        # Z aligned - is there a cube face looking up within the rotation threshold
        if len(self.face_geom_names) == 2:
            z_aligned = rotation.rot_z_aligned(cube_quat, rot_threshold)
        else:  # len(self.face_geom_names) == 6
            z_aligned = rotation.rot_xyz_aligned(cube_quat, rot_threshold)

        # Do reorientation - with some probability, just reorient the cube
        do_reorientation = random_state.uniform() < self.p_face_flip

        # Rotate face - should we rotate face or reorient the cube
        rotate_face = face_aligned and z_aligned and not do_reorientation

        if rotate_face:
            # Chose index from the geoms that is highest on the z axis
            face_to_shift = cube_utils.face_up(
                self.mujoco_simulation.sim, self.face_geom_names
            )

            # Rotate given face by a random angle and return both, new rotations and an angle
            goal_face, delta_angle = cube_utils.rotated_face_with_angle(
                cube_face,
                face_to_shift,
                random_state,
                self.round_target_face,
                directions=self.goal_directions,
            )

            if len(self.face_geom_names) == 2:
                self.mujoco_simulation.rotate_target_face(face_to_shift, delta_angle)
            else:
                self.mujoco_simulation.rotate_target_face(
                    face_to_shift // 2, face_to_shift % 2, delta_angle
                )

            goal_quat = rotation.round_to_straight_quat(cube_quat)
        else:  # need to flip cube
            # Gaol for face rotations is just aligning them
            goal_face = rounded_current_face

            # Make the goal so that a given face is straight up
            candidates = list(range(len(self.face_geom_names)))
            face_to_shift = random_state.choice(candidates)

            z_quat = cube_utils.uniform_z_aligned_quat(random_state)
            face_up_quat = self.goal_quat_for_face[face_to_shift]
            goal_quat = rotation.quat_mul(z_quat, face_up_quat)

        goal_quat = rotation.quat_normalize(goal_quat)

        return {
            "cube_pos": cube_pos,
            "cube_quat": goal_quat,
            "cube_face_angle": goal_face,
            "goal_type": "rotation" if rotate_face else "flip",
        }

    def current_state(self):
        """ Extract current cube goal state """
        cube_pos = np.zeros(3)

        return {
            "cube_pos": cube_pos,
            "cube_quat": self.mujoco_simulation.get_qpos("cube_rotation"),
            "cube_face_angle": self.mujoco_simulation.get_face_angles("cube"),
        }

    def relative_goal(self, goal_state, current_state):
        """
        Calculate a difference in the 'goal space' between current state and the target goal
        """
        return {
            # Cube pos does not count
            "cube_pos": np.zeros(goal_state["cube_pos"].shape),
            # Quaternion difference of a rotation
            "cube_quat": rotation.quat_difference(
                goal_state["cube_quat"], current_state["cube_quat"]
            ),
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
        }

        return goal_distance

    def goal_types(self) -> typing.Set[str]:
        return {"rotation", "flip"}
