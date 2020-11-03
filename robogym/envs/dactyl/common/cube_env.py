import abc
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, TypeVar

import attr
import numpy as np

from robogym.envs.dactyl.common import cube_utils
from robogym.envs.dactyl.common.cube_utils import DEFAULT_CAMERA_NAMES
from robogym.envs.dactyl.common.dactyl_cube_wrappers import apply_wrappers
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.mujoco.simulation_interface import (
    SimulationInterface,
    SimulationParameters,
)
from robogym.observation.dummy_vision import (
    DummyVisionGoalObservationProvider,
    DummyVisionObservationProvider,
)
from robogym.observation.goal import GoalObservationProvider
from robogym.observation.mujoco import MujocoObservationProvider, ObservationProvider
from robogym.robot.shadow_hand.mujoco.mujoco_shadow_hand import MuJoCoShadowHand
from robogym.robot_env import (
    RobotEnv,
    RobotEnvConstants,
    RobotEnvParameters,
    build_nested_attr,
)


@attr.s(auto_attribs=True)
class CubeSimulationParameters(SimulationParameters):
    """Simulation parameters for cube env."""

    # Cube appearance for mujoco and vision rendering.
    cube_appearance: str = "policy"

    # If True, hide target in simulation.
    hide_target: bool = False


@attr.s(auto_attribs=True)
class DactylCubeEnvParameters(RobotEnvParameters):
    """ Parameters of the Dactyl cube env - possible to change between episodes """

    # Standard deviation of initial cube position
    cube_position_wiggle_std: float = 0.005

    # Multiplier of the cube size
    cube_size_multiplier: float = 1.0

    simulation_params: CubeSimulationParameters = build_nested_attr(
        CubeSimulationParameters
    )


@attr.s(auto_attribs=True)
class DactylCubeEnvConstants(RobotEnvConstants):
    """ Parameters of the Dactyl cube env - set once and for all """

    # Maximum number of tries to place cube randomly before we give up for given episode
    max_pose_resets: int = 50

    #####################
    # Multi success related settings

    # How many successes needed to stop roll out.
    successes_needed: int = 50

    max_timesteps_per_goal: int = 400

    # Check if current goal is reachable in each step. This is disabled by default
    # as to get more reliably performance metrics. However, this should be enabled
    # for demos or when recording a video.
    check_goal_reachable: bool = False

    # Max number of steps before considering goal is unreachable.
    max_steps_goal_unreachable: int = 100

    success_pause_range_s: Tuple[float, float] = (0.0, 0.0)

    # Currently supported providers are:
    #
    # phasespace: Enable phasespace tracking for hand only.
    # phasespace/cube: Enable phasespace tracking for both hand and locked cube.
    # giiker: Enable giiker.
    # vision: Enable vision.
    # shadowhand: Enable shadowhand sensor.
    #

    # Example for observation_configs below.
    #
    # Locked cube:
    # {'cube_pos': 'phasespace', 'cube_quat': 'phasespace', 'fingertip_pos': 'phasespace'}
    #
    # Full cube:
    # {
    #     'cube_pos': 'vision',
    #     'cube_quat': 'vision',
    #     'cube_face_angle': 'giiker',
    #     'fingertip_pos': 'phasespace'
    # }
    #

    # How many steps with zero action to take on env reset
    reset_initial_steps: int = 20

    #####################
    # Wrapper settings

    # Args for vision wrapper.
    vision_args: Optional[dict] = None

    # If fix wrist joint position
    fixed_wrist: bool = False

    # If use relative goal wrapper.
    relative_goal_wrapper: bool = True

    # Reward for dropping the cube.
    drop_reward: float = -20.0

    # Minimum episode length.
    min_episode_length: int = -1

    # The cameras to be used when providing vision observations and/or goals.
    camera_names: List[str] = DEFAULT_CAMERA_NAMES

    # Width and height of images (only relevant for vision policies).
    image_size: int = 200


class CubeSimulationInterface(SimulationInterface):
    """
    Simulation interface for shadow hand manipulating some version of a cube.

    Should be subclassed to implement particular version of a cube that simulation will manipulate.
    """

    # Perpendicular cube model
    CUBE_XML = "rubik/rubik_perpendicular.xml"
    # XML with slightly different cube model for vision rendering
    VISION_CUBE_XML = "rubik/rubik_perpendicular_vision.xml"

    # XML for vision rendering with corner cutoff for center piece.
    VISION_CUBE_CORNER_CUTOFF_XML = "rubik/rubik_perpendicular_vision_corner_cutoff.xml"

    # Just a floor
    FLOOR_XML = "floor/basic_floor.xml"
    # Robot hand xml
    HAND_XML = "robot/shadowhand/main.xml"
    # XML with default light
    LIGHT_XML = "light/default.xml"
    # XML with default camera
    CAMERA_XML = "camera/default.xml"

    def __init__(self, sim):
        super().__init__(sim)
        self.enable_pid()
        self.shadow_hand = MuJoCoShadowHand(self)

    def is_cube_on_palm(self):
        """ Determines if cube is on palm """
        return cube_utils.on_palm(self.sim)

    @classmethod
    def _build_mujoco_cube_xml(cls, xml, cube_xml_path):
        """ Builc xml for the cube simulation """
        raise NotImplementedError

    @classmethod
    def build(
        cls,
        n_substeps: int = 10,
        simulation_params: CubeSimulationParameters = CubeSimulationParameters(),
    ):
        """
        Construct a CubeSimulationInterface object with perpendicular cube.
        """

        cube_xml_path, size_params = cls._get_model_xml_path_and_params(
            simulation_params.cube_appearance
        )

        xml = MujocoXML()
        xml.add_default_compiler_directive()

        cls._build_mujoco_cube_xml(xml, cube_xml_path)

        xml.append(
            MujocoXML.parse(cls.FLOOR_XML).set_named_objects_attr(
                "floor", tag="body", pos=[1, 1, 0]
            )
        )

        xml.append(
            MujocoXML.parse(cls.HAND_XML)
            .add_name_prefix("robot0:")
            .set_objects_attr(tag="size", **size_params)
            .set_named_objects_attr(
                "robot0:hand_mount",
                tag="body",
                pos=[1.0, 1.25, 0.15],
                euler=[np.pi / 2, 0, np.pi],
            )
            .remove_objects_by_name("robot0:annotation:outer_bound")
            # Remove hand base free joint so that hand is immovable
            .remove_objects_by_name("robot0:hand_base")
        )

        xml.append(MujocoXML.parse(cls.LIGHT_XML))

        simulation = cls(xml.build(nsubsteps=n_substeps))

        if simulation_params.hide_target:
            simulation.hide_target()

        return simulation

    @classmethod
    def _get_model_xml_path_and_params(cls, cube_appearance):
        assert cube_appearance in (
            "policy",
            "vision",
            "vision_corner_cutoff",
        ), f"Unexpected cube type: {cube_appearance}"

        if cube_appearance == "vision":
            cube_xml_path = cls.VISION_CUBE_XML
            max_contacts_params = dict(
                njmax=6000, nconmax=600, nuserdata=100, nuser_actuator=20
            )
        elif cube_appearance == "vision_corner_cutoff":
            cube_xml_path = cls.VISION_CUBE_CORNER_CUTOFF_XML
            max_contacts_params = dict(
                njmax=6000, nconmax=600, nuserdata=100, nuser_actuator=20
            )
        else:
            cube_xml_path = cls.CUBE_XML
            max_contacts_params = dict(
                njmax=2000, nconmax=200, nuserdata=100, nuser_actuator=20
            )

        return cube_xml_path, max_contacts_params

    @contextmanager
    def hide_target(self):
        """ Make target transparent (or invisible if we want to hide_target) """
        target_geom_ids = [
            self.sim.model.geom_name2id(name)
            for name in self.sim.model.geom_names
            if name.startswith("target")
        ]
        target_mat_ids = [self.sim.model.geom_matid[gid] for gid in target_geom_ids]
        target_site_ids = [
            self.sim.model.site_name2id(name)
            for name in self.sim.model.site_names
            if name.startswith("target")
        ]

        old_mat_rgba = self.sim.model.mat_rgba.copy()
        old_geom_rgba = self.sim.model.geom_rgba.copy()
        old_site_rgba = self.sim.model.site_rgba.copy()

        self.sim.model.mat_rgba[target_mat_ids, -1] = 0
        self.sim.model.geom_rgba[target_geom_ids, -1] = 0
        self.sim.model.site_rgba[target_site_ids, -1] = 0

        yield

        # Make target visible again.
        self.sim.model.mat_rgba[:] = old_mat_rgba
        self.sim.model.geom_rgba[:] = old_geom_rgba
        self.sim.model.site_rgba[:] = old_site_rgba


PType = TypeVar("PType", bound=DactylCubeEnvParameters)
CType = TypeVar("CType", bound=DactylCubeEnvConstants)
SType = TypeVar("SType", bound=CubeSimulationInterface)


class CubeEnv(RobotEnv[PType, CType, SType], abc.ABC):
    """
    Base class for dactyl cube environments.
    Locked, Face, Full, Perpendicular, with right subclass should handle all of them
    """

    def _build_observation_providers(self):
        """
        Initialize observation providers for the environment.
        """
        providers: Dict[str, ObservationProvider] = {
            "mujoco": MujocoObservationProvider(self.mujoco_simulation),
            "goal": GoalObservationProvider(self.goal_info),
        }

        if "dummy_vision" in self.constants.observation_providers:
            providers["dummy_vision"] = DummyVisionObservationProvider(
                camera_names=self.constants.camera_names,
                image_size=self.constants.image_size,
            )
            providers["goal_dummy_vision"] = DummyVisionGoalObservationProvider(
                get_goal=self.goal_info,
                goal_qpos_key="qpos_goal",
                camera_names=self.constants.camera_names,
                image_size=self.constants.image_size,
            )

        return providers

    def _has_episode_ended(self):
        """ Check if simulation is in state good enough to continue training """
        return not self.mujoco_simulation.is_cube_on_palm()

    def _setup_simulation_from_parameters(self):
        """
        Set all the simulation parameters from the current settings.

        You may override it or just leave it as it is for a very basic setup.
        """
        for param_name, modifier in self.modifiers:
            modifier(getattr(self.parameters, param_name))

    ###############################################################################################
    # Internal API - to be overridden - environment randomization
    def _randomize_cube_initial_position(self):
        """ Draw a random initial position for a cube """
        raise NotImplementedError("Override _randomize_cube_initial_position")

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        # Basically randomize cube position, until we get some random state where cube is still on
        # the palm of the hand
        for _ in range(self.constants.max_pose_resets):
            # Reset accumulated warnings
            self.warning_buffer.clear()

            # Reset to the initial state
            self.mujoco_simulation.reset()

            # Set all the simulation parameters
            self._setup_simulation_from_parameters()

            # Set derived constants of the simulation
            self.mujoco_simulation.set_constants()

            # Randomize cube position.
            self._randomize_cube_initial_position()

            if not self._has_episode_ended():
                break

    @classmethod
    def build_robot(cls, mujoco_simulation, physical):
        return mujoco_simulation.shadow_hand

    def apply_wrappers(self, **wrapper_params):
        """
        Apply wrappers to the environment.
        """
        self.constants: DactylCubeEnvConstants

        return apply_wrappers(
            self,
            randomize=self.constants.randomize,
            n_action_bins=self.constants.n_action_bins,
            fixed_wrist=self.constants.fixed_wrist,
            relative_goal_wrapper=self.constants.relative_goal_wrapper,
            drop_reward=self.constants.drop_reward,
            default_wrappers=self._get_default_wrappers(),
            min_episode_length=self.constants.min_episode_length,
            **wrapper_params,
        )

    @classmethod
    def _get_default_wrappers(cls):
        return {
            "post_obsnoise_randomizations": [
                ["FingersOccludedPhasespaceMarkers"],
                ["FingersFreezingPhasespaceMarkers"],
                ["CubeFreezingPhasespaceBody"],
                ["ActionNoiseWrapper"],
            ],
        }
