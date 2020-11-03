import functools
import logging
from typing import List

import attr
import numpy as np

import robogym.utils.rotation as rotation
from robogym.envs.dactyl.common.cube_env import (
    CubeEnv,
    CubeSimulationInterface,
    DactylCubeEnvConstants,
    DactylCubeEnvParameters,
)
from robogym.envs.dactyl.common.mujoco_modifiers import PerpendicularCubeSizeModifier
from robogym.envs.dactyl.goals.face_curriculum import FaceCurriculumGoal
from robogym.envs.dactyl.goals.face_free import FaceFreeGoal
from robogym.envs.dactyl.observation.cube import (
    GoalCubeRotObservation,
    MujocoCubePosObservation,
    MujocoCubeRotObservation,
)
from robogym.envs.dactyl.observation.face_perpendicular import (
    GoalCubePosObservation,
    GoalFaceAngleObservation,
    MujocoFaceAngleObservation,
)
from robogym.envs.dactyl.observation.shadow_hand import (
    MujocoShadowhandAngleObservation,
    MujocoShadowhandRelativeFingertipsObservation,
)
from robogym.goal.goal_generator import GoalGenerator
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.observation.mujoco import MujocoQposObservation, MujocoQvelObservation
from robogym.robot_env import ObservationMapValue as omv

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class FacePerpendicularEnvParameters(DactylCubeEnvParameters):
    """ Parameters of the Dactyl Face Perpendicular env - possible to change for each episode"""

    pass


@attr.s(auto_attribs=True)
class FacePerpendicularEnvConstants(DactylCubeEnvConstants):
    """ Parameters of the Dactyl Perpendicular env - set once and for all """

    # Threshold for success conditions
    success_threshold: dict = {"cube_quat": 0.4, "cube_face_angle": 0.2}

    # What kind of goal generation we want for the environment
    goal_generation: str = "face_curr"

    #####################
    # Curriculum settings

    # Which directions do we rotate the faces
    goal_directions: List[str] = ["cw", "ccw"]

    # Are faces always rotated to round angles
    round_target_face: bool = True

    # Probability of cube reorient vs face rotation
    p_face_flip: float = 0.25


class FacePerpendicularSimulation(CubeSimulationInterface):
    """
    Simulation of a shadow hand manipulating a face cube
    """

    @classmethod
    def _build_mujoco_cube_xml(cls, xml, cube_xml_path):
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .add_name_prefix("cube:")
            .set_named_objects_attr("cube:middle", tag="body", pos=[1.0, 0.87, 0.2])
            # Leave +/- z driver joints
            .remove_objects_by_name(names="cube:cubelet:driver:neg_x", tag="joint")
            .remove_objects_by_name(names="cube:cubelet:driver:pos_x", tag="joint")
            .remove_objects_by_name(names="cube:cubelet:driver:neg_y", tag="joint")
            .remove_objects_by_name(names="cube:cubelet:driver:pos_y", tag="joint")
            # Remove x/y cubelet hinge joints
            .remove_objects_by_prefix(prefix="cube:cubelet:rotx:", tag="joint")
            .remove_objects_by_prefix(prefix="cube:cubelet:roty:", tag="joint")
            # Delete springs for now
            .remove_objects_by_prefix(prefix="cube:cubelet:spring:", tag="joint")
            # Remove remaining cubelet joints we're not interested in
            .remove_objects_by_name(
                names=[
                    "cube:cubelet:rotz:neg_x_pos_y",
                    "cube:cubelet:rotz:neg_x_neg_y",
                    "cube:cubelet:rotz:pos_x_pos_y",
                    "cube:cubelet:rotz:pos_x_neg_y",
                ],
                tag="joint",
            )
        )

        # Target
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .add_name_prefix("target:")
            .set_named_objects_attr("target:middle", tag="body", pos=[1.0, 0.87, 0.2])
            # Disable collisions
            .set_objects_attr(tag="geom", group="2", conaffinity="0", contype="0")
            # Leave +/- z driver joints
            .remove_objects_by_name(names="target:cubelet:driver:neg_x", tag="joint")
            .remove_objects_by_name(names="target:cubelet:driver:pos_x", tag="joint")
            .remove_objects_by_name(names="target:cubelet:driver:neg_y", tag="joint")
            .remove_objects_by_name(names="target:cubelet:driver:pos_y", tag="joint")
            # Remove x/y cubelet hinge joints
            .remove_objects_by_prefix(prefix="target:cubelet:rotx:", tag="joint")
            .remove_objects_by_prefix(prefix="target:cubelet:roty:", tag="joint")
            .remove_objects_by_prefix(prefix="target:cubelet:spring:", tag="joint")
            # Remove remaining cubelet joints we're not interested in
            .remove_objects_by_name(
                names=[
                    "target:cubelet:rotz:neg_x_pos_y",
                    "target:cubelet:rotz:neg_x_neg_y",
                    "target:cubelet:rotz:pos_x_pos_y",
                    "target:cubelet:rotz:pos_x_neg_y",
                ],
                tag="joint",
            )
        )

    def __init__(self, mujoco_simulation):
        super().__init__(mujoco_simulation)

        self.register_joint_group("cube_position", prefix="cube:cube:t")
        self.register_joint_group("cube_rotation", prefix="cube:cube:rot")

        self.register_joint_group_by_name(
            "cube_top_face_driver", name="cube:cubelet:driver:pos_z"
        )
        self.register_joint_group_by_name(
            "cube_bottom_face_driver", name="cube:cubelet:driver:neg_z"
        )

        self.register_joint_group_by_name(
            "cube_drivers",
            name=["cube:cubelet:driver:pos_z", "cube:cubelet:driver:neg_z"],
        )

        self.register_joint_group_by_name(
            "cube_top_face",
            name=[
                "cube:cubelet:driver:pos_z",
                "cube:cubelet:rotz:neg_x_pos_y_pos_z",
                "cube:cubelet:rotz:neg_x_neg_y_pos_z",
                "cube:cubelet:rotz:neg_x_pos_z",
                "cube:cubelet:rotz:pos_x_pos_z",
                "cube:cubelet:rotz:pos_x_neg_y_pos_z",
                "cube:cubelet:rotz:pos_x_pos_y_pos_z",
                "cube:cubelet:rotz:neg_y_pos_z",
                "cube:cubelet:rotz:pos_y_pos_z",
            ],
        )

        self.register_joint_group_by_name(
            "cube_bottom_face",
            name=[
                "cube:cubelet:driver:neg_z",
                "cube:cubelet:rotz:neg_x_pos_y_neg_z",
                "cube:cubelet:rotz:neg_x_neg_y_neg_z",
                "cube:cubelet:rotz:neg_x_neg_z",
                "cube:cubelet:rotz:pos_x_neg_z",
                "cube:cubelet:rotz:pos_x_neg_y_neg_z",
                "cube:cubelet:rotz:pos_x_pos_y_neg_z",
                "cube:cubelet:rotz:neg_y_neg_z",
                "cube:cubelet:rotz:pos_y_neg_z",
            ],
        )

        self.register_joint_group("cube_springs", prefix="cube:cubelet:spring:")

        self.register_joint_group("target_position", prefix="target:cube:t")
        self.register_joint_group("target_rotation", prefix="target:cube:rot")

        self.register_joint_group_by_name(
            "target_top_face_driver", name="target:cubelet:driver:pos_z"
        )
        self.register_joint_group_by_name(
            "target_bottom_face_driver", name="target:cubelet:driver:neg_z"
        )

        self.register_joint_group_by_name(
            "target_drivers",
            name=["target:cubelet:driver:pos_z", "target:cubelet:driver:neg_z"],
        )

        self.register_joint_group_by_name(
            "target_top_face",
            name=[
                "target:cubelet:driver:pos_z",
                "target:cubelet:rotz:neg_x_pos_y_pos_z",
                "target:cubelet:rotz:neg_x_neg_y_pos_z",
                "target:cubelet:rotz:neg_x_pos_z",
                "target:cubelet:rotz:pos_x_pos_z",
                "target:cubelet:rotz:pos_x_neg_y_pos_z",
                "target:cubelet:rotz:pos_x_pos_y_pos_z",
                "target:cubelet:rotz:neg_y_pos_z",
                "target:cubelet:rotz:pos_y_pos_z",
            ],
        )

        self.register_joint_group_by_name(
            "target_bottom_face",
            name=[
                "target:cubelet:driver:neg_z",
                "target:cubelet:rotz:neg_x_pos_y_neg_z",
                "target:cubelet:rotz:neg_x_neg_y_neg_z",
                "target:cubelet:rotz:neg_x_neg_z",
                "target:cubelet:rotz:pos_x_neg_z",
                "target:cubelet:rotz:pos_x_neg_y_neg_z",
                "target:cubelet:rotz:pos_x_pos_y_neg_z",
                "target:cubelet:rotz:neg_y_neg_z",
                "target:cubelet:rotz:pos_y_neg_z",
            ],
        )

        self.register_joint_group("target_springs", prefix="target:cubelet:spring:")

        self.register_joint_group("target_all_joints", prefix="target:")

        self.register_joint_group("hand_angle", prefix="robot0:")

    def set_face_angles(self, target, angles):
        assert target in {"cube", "target"}
        self.set_qpos("{}_top_face".format(target), angles[0])
        self.set_qpos("{}_bottom_face".format(target), angles[1])

    def get_face_angles(self, target):
        assert target in {"cube", "target"}
        return self.get_qpos("{}_drivers".format(target))

    def rotate_target_face(self, side, angle):
        """ Rotate given face of the target by given angle """
        qpos = self.get_face_angles("target")
        qpos[side] += angle
        self.set_face_angles("target", qpos)

    def clone_target_from_cube(self):
        """ Clone target internal state from cube state """
        self.set_face_angles("target", self.get_face_angles("cube"))

    def align_target_faces(self):
        """ Align target orientation to straight orientation of the cube """
        self.set_face_angles(
            "target", rotation.round_to_straight_angles(self.get_face_angles("target"))
        )


class FacePerpendicularEnv(
    CubeEnv[
        FacePerpendicularEnvParameters,
        FacePerpendicularEnvConstants,
        FacePerpendicularSimulation,
    ]
):
    """
    A dactyl Rubik's cube environment that aims to replicate dactyl face env using simpler
    code
    """

    # Target angle is two numbers: top face angle and bottom face angle
    TARGET_ANGLE_SHAPE = 2

    FACE_GEOM_NAMES = ["cube:cubelet:pos_z", "cube:cubelet:neg_z"]

    FACE_JOINT_NAMES = [
        "cubelet:driver:pos_z",
        "cubelet:rotz:neg_x_pos_y_pos_z",
        "cubelet:rotz:neg_x_neg_y_pos_z",
        "cubelet:rotz:neg_x_pos_z",
        "cubelet:rotz:pos_x_pos_z",
        "cubelet:rotz:pos_x_neg_y_pos_z",
        "cubelet:rotz:pos_x_pos_y_pos_z",
        "cubelet:rotz:neg_y_pos_z",
        "cubelet:rotz:pos_y_pos_z",
        "cubelet:driver:neg_z",
        "cubelet:rotz:neg_x_pos_y_neg_z",
        "cubelet:rotz:neg_x_neg_y_neg_z",
        "cubelet:rotz:neg_x_neg_z",
        "cubelet:rotz:pos_x_neg_z",
        "cubelet:rotz:pos_x_neg_y_neg_z",
        "cubelet:rotz:pos_x_pos_y_neg_z",
        "cubelet:rotz:neg_y_neg_z",
        "cubelet:rotz:pos_y_neg_z",
    ]

    def _default_observation_map(self):
        return {
            "cube_pos": omv({"mujoco": MujocoCubePosObservation}),
            "cube_quat": omv({"mujoco": MujocoCubeRotObservation}),
            "cube_face_angle": omv({"mujoco": MujocoFaceAngleObservation}),
            "qpos": omv({"mujoco": MujocoQposObservation}),
            "qvel": omv({"mujoco": MujocoQvelObservation}),
            "perp_qpos": omv({"mujoco": MujocoQposObservation}),  # Duplicate of qpos.
            "perp_qvel": omv({"mujoco": MujocoQvelObservation}),  # Duplicate of qvel.
            "hand_angle": omv({"mujoco": MujocoShadowhandAngleObservation}),
            "fingertip_pos": omv(
                {"mujoco": MujocoShadowhandRelativeFingertipsObservation}
            ),
            "goal_pos": omv({"goal": GoalCubePosObservation}),
            "goal_quat": omv({"goal": GoalCubeRotObservation}),
            "goal_face_angle": omv({"goal": GoalFaceAngleObservation}),
        }

    @classmethod
    def build_goal_generation(
        cls,
        constants: FacePerpendicularEnvConstants,
        mujoco_simulation: FacePerpendicularSimulation,
    ) -> GoalGenerator:
        """ Construct a goal generation object """
        if constants.goal_generation == "face_curr":
            return FaceCurriculumGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                goal_directions=constants.goal_directions,
                round_target_face=constants.round_target_face,
                p_face_flip=constants.p_face_flip,
            )
        elif constants.goal_generation == "face_free":
            return FaceFreeGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                goal_directions=constants.goal_directions,
                round_target_face=constants.round_target_face,
                p_face_flip=constants.p_face_flip,
            )
        else:
            raise RuntimeError(
                "Invalid 'goal_generation' constant '{}'".format(
                    constants.goal_generation
                )
            )

    @classmethod
    def build_simulation(cls, constants, parameters):
        return FacePerpendicularSimulation.build(
            n_substeps=constants.mujoco_substeps,
            simulation_params=parameters.simulation_params,
        )

    @classmethod
    def build_mujoco_modifiers(cls):
        modifiers = super().build_mujoco_modifiers()
        modifiers["cube_size_multiplier"] = PerpendicularCubeSizeModifier("cube:")
        return modifiers

    ###############################################################################################
    # Internal API - to be overridden - environment randomization
    def _randomize_cube_initial_position(self):
        """ Draw a random initial position for a cube """
        # Original env had this, but I'm not really sure it's needed
        for i in range(self.constants.reset_initial_steps):
            ctrl = self.mujoco_simulation.shadow_hand.denormalize_position_control(
                self.mujoco_simulation.shadow_hand.zero_control()
            )
            self.mujoco_simulation.shadow_hand.set_position_control(ctrl)
            self.mujoco_simulation.step()

        cube_translation = (
            self._random_state.randn(3) * self.parameters.cube_position_wiggle_std
        )
        self.mujoco_simulation.add_qpos("cube_position", cube_translation)

        cube_orientation = rotation.uniform_quat(self._random_state)
        self.mujoco_simulation.set_qpos("cube_rotation", cube_orientation)

        random_face_angle = self._random_state.uniform(-np.pi / 4, np.pi / 4, 2)
        self.mujoco_simulation.set_face_angles("cube", random_face_angle)

        # Need to call this after the qpos is modified
        self.mujoco_simulation.forward()

        action = self._random_state.uniform(-1.0, 1.0, self.action_space.shape[0])

        for _ in range(self.parameters.n_random_initial_steps):
            ctrl = self.mujoco_simulation.shadow_hand.denormalize_position_control(
                action
            )
            self.mujoco_simulation.shadow_hand.set_position_control(ctrl)
            self.mujoco_simulation.step()

    ###############################################################################################
    # External API - to establish communication with other parts of the system
    @property
    def cube_type(self):
        """ Type of cube """
        return "face-perpendicular"

    @property
    def face_joint_names(self):
        """ Used by some wrapper """
        return self.FACE_JOINT_NAMES

    ###############################################################################################
    # Fully internal methods
    def _render_callback(self, _sim, _viewer):
        """ Set a render callback """
        self.mujoco_simulation.set_qpos("target_position", np.array([0.15, 0, -0.03]))
        self.mujoco_simulation.set_qpos("target_rotation", self._goal["cube_quat"])
        self.mujoco_simulation.set_face_angles("target", self._goal["cube_face_angle"])

        self.mujoco_simulation.set_qvel("target_position", 0.0)
        self.mujoco_simulation.set_qvel("target_rotation", 0.0)
        self.mujoco_simulation.set_qvel("target_top_face", 0.0)
        self.mujoco_simulation.set_qvel("target_bottom_face", 0.0)

        self.mujoco_simulation.forward()

    @classmethod
    def _get_default_wrappers(cls):
        default_wrappers = super()._get_default_wrappers()

        default_wrappers.update(
            {
                "default_observation_noise_levels": {
                    "fingertip_pos": {"uncorrelated": 0.002, "additive": 0.001},
                    "hand_angle": {"additive": 0.1, "uncorrelated": 0.1},
                    "cube_pos": {"additive": 0.005, "uncorrelated": 0.001},
                    "cube_quat": {"additive": 0.1, "uncorrelated": 0.09},
                    "cube_face_angle": {"additive": 0.1, "uncorrelated": 0.1},
                },
                "default_no_noise_levels": {
                    "fingertip_pos": {},
                    "hand_angle": {},
                    "cube_pos": {},
                    "cube_quat": {},
                    "cube_face_angle": {},
                },
                "default_observation_delay_levels": {
                    "interpolators": {
                        "cube_quat": "QuatInterpolator",
                        "cube_face_angle": "RadianInterpolator",
                    },
                    "groups": {
                        # Uncomment below to enable observation delay randomization.
                        # "vision": {
                        #    "obs_names": ["cube_pos", "cube_quat"],
                        #    "mean": 3,
                        #    "std": 0.5,
                        # },
                        # "giiker": {
                        #    "obs_names": ["cube_face_angle"],
                        #    "mean": 1,
                        #    "std": 0.2,
                        # },
                        # "phasespace": {
                        #    "obs_names": ["fingertip_pos"],
                        #    "mean": 0.5,
                        #    "std": 0.1,
                        # }
                    },
                },
                "default_no_observation_delay_levels": {
                    "interpolators": {},
                    "groups": {},
                },
                "pre_obsnoise_randomizations": [
                    ["RandomizedActionLatency"],
                    ["RandomizedPerpendicularCubeSizeWrapper"],
                    ["RandomizedBodyInertiaWrapper"],
                    ["RandomizedTimestepWrapper"],
                    ["RandomizedRobotFrictionWrapper"],
                    ["RandomizedCubeFrictionWrapper"],
                    ["RandomizedGravityWrapper"],
                    ["RandomizedWindWrapper"],
                    ["RandomizedPhasespaceFingersWrapper"],
                    ["RandomizedRobotDampingWrapper"],
                    ["RandomizedRobotKpWrapper"],
                    ["RandomizedFaceDampingWrapper"],
                    ["RandomizedJointLimitWrapper"],
                    ["RandomizedTendonRangeWrapper"],
                ],
            }
        )

        return default_wrappers


make_simple_env = functools.partial(FacePerpendicularEnv.build, apply_wrappers=False)
make_env = FacePerpendicularEnv.build
