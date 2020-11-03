import functools
import typing

import attr
import numpy as np
import pycuber

import robogym.utils.rotation as rotation
from robogym.envs.dactyl.common.cube_env import (
    CubeEnv,
    CubeSimulationInterface,
    DactylCubeEnvConstants,
    DactylCubeEnvParameters,
)
from robogym.envs.dactyl.common.cube_manipulator import CubeManipulator
from robogym.envs.dactyl.common.mujoco_modifiers import PerpendicularCubeSizeModifier
from robogym.envs.dactyl.goals.face_cube_solver import FaceCubeSolverGoal
from robogym.envs.dactyl.goals.face_curriculum import FaceCurriculumGoal
from robogym.envs.dactyl.goals.face_free import FaceFreeGoal
from robogym.envs.dactyl.goals.fixed_fair_scramble import FixedFairScrambleGoal
from robogym.envs.dactyl.goals.full_unconstrained import FullUnconstrainedGoal
from robogym.envs.dactyl.goals.release_cube_solver import ReleaseCubeSolverGoal
from robogym.envs.dactyl.goals.unconstrained_cube_solver import UnconstrainedCubeSolver
from robogym.envs.dactyl.observation.cube import (
    GoalCubeRotObservation,
    MujocoCubePosObservation,
    MujocoCubeRotObservation,
)
from robogym.envs.dactyl.observation.full_perpendicular import (
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


@attr.s(auto_attribs=True)
class FullPerpendicularEnvParameters(DactylCubeEnvParameters):
    """ Parameters of the Dactyl Perpendicular env - possible to change for each episode """

    # How many steps with random action do we take when the environment is initialized
    n_random_initial_steps: int = 10

    # Multiplier of the cube size
    cube_size_multiplier: float = 1.0


@attr.s(auto_attribs=True)
class FullPerpendicularEnvConstants(DactylCubeEnvConstants):
    """ Parameters of the Dactyl Perpendicular env - set once and for all """

    # Number of mujoco simulation steps per environment step
    mujoco_substeps: int = 10

    # How many steps with zero action to take on env reset
    reset_initial_steps: int = 20

    # Threshold for success conditions
    success_threshold: dict = {"cube_quat": 0.4, "cube_face_angle": 0.2}

    max_timesteps_per_goal: int = 1600

    # What kind of goal generation we want for the environment
    goal_generation: str = "face_free"

    # Which directions do we rotate the faces
    goal_directions: typing.List[str] = ["cw", "ccw"]

    # Are faces always rotated to round angles
    round_target_face: bool = True

    # Probability of cube reorient vs face rotation
    p_face_flip: float = 0.5

    # How many times to scramble cube initially
    num_scramble_steps: int = 50

    # Whether to scramble face angles at the beginning of each episode
    scramble_face_angles: bool = True

    # Whether to randomize face angles at the beginning of each episode.
    randomize_face_angles: bool = True


class FullPerpendicularSimulation(CubeSimulationInterface):
    """
    Simulation of a shadow hand manipulating full perpendicular cube
    """

    @classmethod
    def _build_mujoco_cube_xml(cls, xml, cube_xml_path):
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .add_name_prefix("cube:")
            .set_named_objects_attr("cube:middle", tag="body", pos=[1.0, 0.87, 0.2])
            # Delete springs for now
            .remove_objects_by_prefix(prefix="cube:cubelet:spring:", tag="joint")
        )

        # Target
        xml.append(
            MujocoXML.parse(cube_xml_path)
            .add_name_prefix("target:")
            .set_named_objects_attr("target:middle", tag="body", pos=[1.0, 0.87, 0.2])
            # Delete springs for now
            .remove_objects_by_prefix(prefix="target:cubelet:spring:", tag="joint")
            # Disable collisions
            .set_objects_attr(tag="geom", group="2", conaffinity="0", contype="0")
        )

    def __init__(self, sim):
        super().__init__(sim)
        self.cube_model = CubeManipulator(prefix="cube:", sim=sim)
        self.target_model = CubeManipulator(prefix="target:", sim=sim)

        self.register_joint_group("cube_position", prefix="cube:cube:t")
        self.register_joint_group("cube_rotation", prefix="cube:cube:rot")
        self.register_joint_group("cube_drivers", prefix="cube:cubelet:driver:")
        self.register_joint_group("cube_cubelets", prefix="cube:cubelet:")

        self.register_joint_group("target_position", prefix="target:cube:t")
        self.register_joint_group("target_rotation", prefix="target:cube:rot")
        self.register_joint_group("target_drivers", prefix="target:cubelet:driver:")
        self.register_joint_group("target_cubelets", prefix="target:cubelet:")

        self.register_joint_group("cube_all_joints", prefix="cube:")
        self.register_joint_group("target_all_joints", prefix="target:")

        self.register_joint_group("hand_angle", prefix="robot0:")

    def clone_target_from_cube(self):
        """ Clone target internal state from cube state """
        self.set_qpos("target_cubelets", self.get_qpos("cube_cubelets"))

    def get_face_angles(self, target):
        """ Return "face angles" either from cube or from the target """
        assert target in {"cube", "target"}
        return self.get_qpos("{}_drivers".format(target))

    def align_target_faces(self):
        """ Align target orientation to given set of **straight** angles """
        self.target_model.soft_align_faces()

    def rotate_target_face(self, axis, side, angle):
        """ Rotate given face of the target by given angle """
        self.target_model.rotate_face(axis, side, angle)


class FullPerpendicularEnv(
    CubeEnv[
        FullPerpendicularEnvParameters,
        FullPerpendicularEnvConstants,
        FullPerpendicularSimulation,
    ]
):
    """
    A dactyl Rubik's cube environment that aims to replicate physically accurately
    perpendicular rotations
    """

    # Face angle is six numbers: one number for each cube face
    TARGET_ANGLE_SHAPE = 6

    FACE_GEOM_NAMES = [
        "cube:cubelet:neg_x",
        "cube:cubelet:pos_x",
        "cube:cubelet:neg_y",
        "cube:cubelet:pos_y",
        "cube:cubelet:neg_z",
        "cube:cubelet:pos_z",
    ]

    PYCUBER_ACTIONS = ["L", "L'", "R", "R'", "F", "F'", "B", "B'", "D", "D'", "U", "U'"]

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
        constants: FullPerpendicularEnvConstants,
        mujoco_simulation: CubeSimulationInterface,
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
        elif constants.goal_generation == "face_cube_solver":
            return FaceCubeSolverGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        elif constants.goal_generation == "release_cube_solver":
            return ReleaseCubeSolverGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        elif constants.goal_generation == "full_unconstrained":
            return FullUnconstrainedGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                goal_directions=constants.goal_directions,
                round_target_face=constants.round_target_face,
            )
        elif constants.goal_generation == "unconstrained_cube_solver":
            return UnconstrainedCubeSolver(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        elif constants.goal_generation == "fixed_fair_scramble":
            return FixedFairScrambleGoal(
                mujoco_simulation=mujoco_simulation,
                success_threshold=constants.success_threshold,
                face_geom_names=cls.FACE_GEOM_NAMES,
                num_scramble_steps=constants.num_scramble_steps,
            )
        else:
            raise RuntimeError(
                "Invalid 'goal_generation' constant '{}'".format(
                    constants.goal_generation
                )
            )

    @classmethod
    def build_simulation(cls, constants, parameters):
        return FullPerpendicularSimulation.build(
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
    def _scramble_cube(self):
        """ Scramble a cube randomly at the beginning of an episode """
        cube = pycuber.Cube()

        for i in range(self.constants.num_scramble_steps):
            action = self._random_state.choice(self.PYCUBER_ACTIONS)
            cube.perform_step(action)

        self.mujoco_simulation.cube_model.from_pycuber(cube)

    def _scramble_face_angles(self):
        """ Scramble face angles randomly without moving cubelets in any way """
        random_angles = self._random_state.choice([-2, -1, 0, 1, 2], size=6) * np.pi / 2
        self.mujoco_simulation.set_qpos("cube_drivers", random_angles)

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

        self._scramble_cube()

        if self.constants.scramble_face_angles:
            self._scramble_face_angles()

        if self.constants.randomize_face_angles:
            # Face angles
            random_face_angle = self._random_state.uniform(
                -np.pi / 4, np.pi / 4, size=2
            )
            # Face axes
            random_axis = self._random_state.randint(3)

            self.mujoco_simulation.cube_model.rotate_face(
                random_axis, 0, random_face_angle[0]
            )

            self.mujoco_simulation.cube_model.rotate_face(
                random_axis, 1, random_face_angle[1]
            )

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
        return "full-perpendicular"

    @property
    def face_joint_names(self):
        # Needed by some wrappers
        return [
            # Need to drop 'cube:' prefix
            x[5:]
            for x in self.mujoco_simulation.cube_model.joints
        ]

    ###############################################################################################
    # Fully internal methods
    def _render_callback(self, _sim, _viewer):
        """ Set a render callback """
        self.mujoco_simulation.set_qpos("target_position", np.array([0.15, 0, -0.03]))
        self.mujoco_simulation.set_qpos("target_rotation", self._goal["cube_quat"])
        self.mujoco_simulation.set_qpos("target_drivers", self._goal["cube_face_angle"])

        self.mujoco_simulation.set_qvel("target_all_joints", 0.0)

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


make_simple_env = functools.partial(FullPerpendicularEnv.build, apply_wrappers=False)
make_env = FullPerpendicularEnv.build
