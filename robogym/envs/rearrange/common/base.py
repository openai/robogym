import abc
import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import attr
import numpy as np
import scipy
from gym.envs.robotics import utils

import robogym.utils.rotation as rotation
from robogym.envs.rearrange.common.utils import (
    load_all_materials,
    load_material_args,
    place_objects_in_grid,
    place_objects_with_no_constraint,
    sample_group_counts,
    stabilize_objects,
)
from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.goals.train_state import TrainStateGoal
from robogym.envs.rearrange.observation.common import (
    MujocoGoalImageObservationProvider,
    MujocoImageObservationProvider,
)
from robogym.envs.rearrange.simulation.base import (
    ObjectGroupConfig,
    RearrangeSimParameters,
    RearrangeSimulationInterface,
)
from robogym.mujoco.constants import MujocoEquality
from robogym.observation.common import SyncType
from robogym.observation.image import ImageObservation
from robogym.randomization.env import build_randomizable_param
from robogym.randomization.sim import (
    GenericSimRandomizer,
    GeomSolimpRandomizer,
    GeomSolrefRandomizer,
    GravityRandomizer,
    JointMarginRandomizer,
    PidRandomizer,
)
from robogym.robot.robot_interface import RobotControlParameters
from robogym.robot_env import ObservationMapValue as omv
from robogym.robot_env import (
    RobotEnv,
    RobotEnvConstants,
    RobotEnvParameters,
    build_nested_attr,
    get_generic_param_type,
)
from robogym.utils.env_utils import InvalidSimulationError
from robogym.wrappers.util import (
    BinSpacing,
    ClipRewardWrapper,
    DiscretizeActionWrapper,
    SmoothActionWrapper,
)

logger = logging.getLogger(__name__)

VISION_OBS = "vision_obs"
VISION_OBS_MOBILE = "vision_obs_mobile"
VISION_GOAL = "vision_goal"


@attr.s(auto_attribs=True)
class EncryptEnvConstants:
    enabled: bool = False

    use_dummy: bool = False

    input_shape: List[int] = [200, 200, 3]

    distortable_keys: List[str] = []

    encryption_prefix: str = "enc_"

    last_activation: str = "TANH"

    # Set hardcoded values for randomization parameters here.
    # This is useful for creating an evaluator with fixed parameters (i.e. not subject to ADR).
    param_values: Optional[Dict[str, Union[float, int]]] = None

    # Whether we should use different Encryption networks for each input key.
    use_separate_networks_per_key: bool = True


@attr.s(auto_attribs=True)
class VisionArgs:
    # The size of the rendered obs and goal images in pixel.
    image_size: int = 200

    # Whether rendering high-res images, only used in examine_vision.
    high_res_mujoco: bool = False

    # Names for static cameras.
    camera_names: List[str] = ["vision_cam_front"]

    # Camera used for mobile cameras. They are only used
    # for obs images.
    mobile_camera_names: List[str] = ["vision_cam_wrist"]

    def all_camera_names(self):
        """Returns all camera names specified by static and mobile camera lists."""
        return self.mobile_camera_names + self.camera_names


@attr.s(auto_attribs=True)
class RearrangeEnvConstants(RobotEnvConstants):
    mujoco_substeps: int = 40
    mujoco_timestep: float = 0.001

    # If this is set to true, new "masked_*" observation keys will be created with goal and
    # object states where objects outside the placement area will be zeroed out.
    mask_obs_outside_placement_area: bool = False

    # If use vision observation.
    vision: bool = False

    vision_args: VisionArgs = build_nested_attr(VisionArgs)

    # lambda range in exponential decay distribution for sampling duplicated object counts.
    sample_lam_low: float = 0.1
    sample_lam_high: float = 5.0

    #####################
    # Success/Rewards

    # Only the metrics present here will be used to compute goal distance
    success_threshold: dict = {"obj_pos": 0.04, "obj_rot": 0.2}

    # Max timesteps per object for each goal until timeout.
    # `constants.max_timesteps_per_goal` will be overwritten by this x num. object.
    max_timesteps_per_goal_per_obj: int = 200

    # Reward per object that is achieved if all distances are under `success_threshold`
    goal_reward_per_object: float = 1.0

    #####################
    # Goal settings
    goal_args: GoalArgs = build_nested_attr(GoalArgs)

    encrypt_env_constants: EncryptEnvConstants = build_nested_attr(EncryptEnvConstants)

    # Action spacing function to be used by the DiscretizeActionWrapper
    action_spacing: BinSpacing = attr.ib(
        default=BinSpacing.LINEAR,
        validator=attr.validators.in_(BinSpacing),
        converter=BinSpacing,
    )

    def has_mobile_cameras_enabled(self):
        return self.vision and len(self.vision_args.mobile_camera_names) > 0

    # This setting causes the env to ignore all actions and teleport directly to the goal.
    # This is useful for training a standalone goal classifier with a balanced distribution of
    # goal/non-goal images.
    teleport_to_goal: bool = False

    # Goal generation for env.
    # state: Use fully random object state (pos + rot) as goal.
    # image: Use real image as goal.
    # train: Use random object state controlled by ADR params as goal.
    goal_generation: str = attr.ib(
        default="state", validator=attr.validators.in_(["state", "image", "train"])
    )

    # Fix initial state
    # for debugging purpose, this option starts episode from fixed initial state again and
    # again. Initial state is determined by the `starting_seed`.
    use_fixed_seed: bool = False

    #####################
    # Self play settings.

    stabilize_objects: bool = True


@attr.s(auto_attribs=True)
class RearrangeRobotControlParameters(RobotControlParameters):
    """ Robot control parameters – defined as parameters since max_position_change can be
    subject to ADR randomization """

    # refer to RobotControlParameters.default_max_pos_change_for_solver to set a
    # reasonable default here.
    max_position_change: float = build_randomizable_param(0.1, low=0.01, high=2.5)


@attr.s(auto_attribs=True)
class RearrangeEnvParameters(RobotEnvParameters):
    robot_control_params: RearrangeRobotControlParameters = build_nested_attr(
        RearrangeRobotControlParameters
    )

    simulation_params: RearrangeSimParameters = build_nested_attr(
        RearrangeSimParameters
    )

    # If true, turn on debugging features like visualizing bounding boxes.
    debug: bool = False

    # the range of object rescaling ratios (in log scale)
    object_scale_low: float = build_randomizable_param(0.0, low=0.0, high=0.5)
    object_scale_high: float = build_randomizable_param(0.0, low=0.0, high=1.8)

    # List of materials available for this environment. If multiple materials are specified
    # we will randomly select material from the list for each object.
    # If the list is None, all materials under materials folder will be considered.
    # If the list is empty, no material specific mujoco args will be applied.
    material_names: Optional[List[str]] = ["default"]


PType = TypeVar("PType", bound=RearrangeEnvParameters)
CType = TypeVar("CType", bound=RearrangeEnvConstants)
SType = TypeVar("SType", bound=RearrangeSimulationInterface)


class RearrangeEnv(RobotEnv[PType, CType, SType], abc.ABC):
    """
    Base env setup for Rearrange.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cache the boundary of placement area per reset to speed up calculation of masked observation.
        self._placement_area_boundary = (
            self.mujoco_simulation.extract_placement_area_boundary()
        )

    def _build_observation_providers(self):
        providers = {}

        if self.constants.vision:
            image_size = self.constants.vision_args.image_size
            # Default to mujoco based rendering.
            providers.update(
                {
                    "image": MujocoImageObservationProvider(
                        self.mujoco_simulation,
                        self.constants.vision_args.camera_names,
                        image_size,
                    ),
                    "image_mobile": MujocoImageObservationProvider(
                        self.mujoco_simulation,
                        self.constants.vision_args.mobile_camera_names,
                        image_size,
                    ),
                    "goal_image": MujocoGoalImageObservationProvider(
                        self.mujoco_simulation,
                        self.constants.vision_args.camera_names,
                        image_size,
                        self.goal_info,
                        "qpos_goal",
                        hide_robot=self.constants.goal_args.goal_hide_robot,
                    ),
                }
            )

            if self.constants.vision_args.high_res_mujoco:
                providers.update(
                    {
                        "image_high_res": MujocoImageObservationProvider(
                            self.mujoco_simulation,
                            self.constants.vision_args.camera_names,
                            600,
                        ),
                        "image_mobile_high_res": MujocoImageObservationProvider(
                            self.mujoco_simulation,
                            self.constants.vision_args.mobile_camera_names,
                            600,
                        ),
                        "goal_image_high_res": MujocoGoalImageObservationProvider(
                            self.mujoco_simulation,
                            self.constants.vision_args.camera_names,
                            600,
                            self.goal_info,
                            "qpos_goal",
                            hide_robot=self.constants.goal_args.goal_hide_robot,
                        ),
                    }
                )

        return providers

    def _default_observation_map(self):
        obs_map = {}

        if self.constants.vision:
            obs_map.update(
                {
                    VISION_OBS: omv({"image": ImageObservation}),
                    VISION_OBS_MOBILE: omv({"image_mobile": ImageObservation}),
                    VISION_GOAL: omv({"goal_image": ImageObservation}),
                }
            )

        if self.constants.vision_args.high_res_mujoco:
            obs_map.update(
                {
                    VISION_OBS + "_high_res": omv({"image_high_res": ImageObservation}),
                    VISION_OBS_MOBILE
                    + "_high_res": omv({"image_mobile_high_res": ImageObservation}),
                    VISION_GOAL
                    + "_high_res": omv({"goal_image_high_res": ImageObservation}),
                }
            )

        return obs_map

    def _mask_goal_observation(
        self, obs: dict, goal_objects_in_placement_area: np.ndarray
    ) -> dict:
        """Create masked goal observation.
        """
        if not self.constants.mask_obs_outside_placement_area:
            return obs

        # Add goal observations (with optional masking).
        # 1.0 if an object is within the placement area.
        mask = goal_objects_in_placement_area.astype(np.float).reshape(-1, 1)
        assert mask.ndim == obs["goal_obj_pos"].ndim
        assert mask.shape[0] == obs["goal_obj_pos"].shape[0]

        obs["goal_placement_mask"] = mask.copy()

        # Mask all other goal-related observations. Note that We do not mask qpos_goal since
        # it contains the robot joints. qpos is only used for image rendering and policy should
        # not observe it.
        goal_obs_keys = [
            "goal_obj_pos",
            "goal_obj_rot",
            "rel_goal_obj_pos",
            "rel_goal_obj_rot",
        ]
        for k in goal_obs_keys:
            obs[f"masked_{k}"] = obs[k] * mask

        return obs

    def _mask_object_observation(self, obs: dict) -> dict:
        """Create masked object state observation.
        """
        if not self.constants.mask_obs_outside_placement_area:
            return obs

        assert self._placement_area_boundary is not None
        mask = self.mujoco_simulation.check_objects_in_placement_area(
            obs["obj_pos"].copy(),
            placement_area_boundary=self._placement_area_boundary,
            margin=self.constants.goal_args.mask_margin,
            soft=self.constants.goal_args.soft_mask,
        )
        # 1.0 if an object is in the placement area.
        mask = mask.astype(np.float).reshape(-1, 1)
        assert mask.ndim == obs["obj_pos"].ndim
        assert mask.shape[0] == obs["obj_pos"].shape[0]

        obs["placement_mask"] = mask.copy()

        obs_keys_to_mask = [
            "obj_pos",
            "obj_rot",
            "obj_rel_pos",
            "obj_vel_pos",
            "obj_vel_rot",
            "obj_gripper_contact",
            "obj_bbox_size",
            "obj_colors",
        ]
        for k in obs_keys_to_mask:
            obs[f"masked_{k}"] = obs[k] * mask

        return obs

    def _observe_simple(self):
        """
        Default observation for the environment. An observation can be added here if
        it meets one of the condition below:

        1. It's cheap to fetch on a per step basic and doesn't cause side effect even
            if not used by policy.
        2. It's shared across all variances of the env.
        """

        robot_obs = self.mujoco_simulation.robot.observe()

        obs = {
            "obj_pos": self.mujoco_simulation.get_object_pos(),
            "obj_rel_pos": self.mujoco_simulation.get_object_rel_pos(),
            "obj_vel_pos": self.mujoco_simulation.get_object_vel_pos(),
            "obj_rot": self.mujoco_simulation.get_object_rot(),
            "obj_vel_rot": self.mujoco_simulation.get_object_vel_rot(),
            "robot_joint_pos": robot_obs.joint_positions(),
            "gripper_pos": robot_obs.tcp_xyz(),
            "gripper_velp": robot_obs.tcp_vel(),
            "gripper_controls": robot_obs.gripper_controls(),
            "gripper_qpos": robot_obs.gripper_qpos(),
            "gripper_vel": robot_obs.gripper_vel(),
            "qpos": self.mujoco_simulation.qpos,
            "qpos_goal": self._goal["qpos_goal"].copy(),
            "goal_obj_pos": self._goal["obj_pos"].copy(),
            "goal_obj_rot": self._goal["obj_rot"].copy(),
            "is_goal_achieved": np.array([self._is_goal_achieved], np.int32),
            "rel_goal_obj_pos": self._goal_info_dict["rel_goal_obj_pos"].copy(),
            "rel_goal_obj_rot": self._goal_info_dict["rel_goal_obj_rot"].copy(),
            "obj_gripper_contact": self.mujoco_simulation.get_object_gripper_contact(),
            "obj_bbox_size": self.mujoco_simulation.get_object_bounding_box_sizes(),
            "obj_colors": self.mujoco_simulation.get_object_colors(),
            "safety_stop": np.array([robot_obs.is_in_safety_stop()]),
            "tcp_force": robot_obs.tcp_force(),
            "tcp_torque": robot_obs.tcp_torque(),
        }

        if self.constants.mask_obs_outside_placement_area:
            obs = self._mask_goal_observation(
                obs, self._goal["goal_objects_in_placement_area"].copy()
            )
            obs = self._mask_object_observation(obs)

        return obs

    @classmethod
    def build_simulation(cls, constants: CType, parameters: PType) -> SType:
        constants.max_timesteps_per_goal = (
            constants.max_timesteps_per_goal_per_obj
            * parameters.simulation_params.num_objects
        )

        simulation_type = get_generic_param_type(cls, 2, RearrangeSimulationInterface)
        return simulation_type.build(
            robot_control_params=parameters.robot_control_params,
            simulation_params=parameters.simulation_params,
            n_substeps=constants.mujoco_substeps,
            mujoco_timestep=constants.mujoco_timestep,
        )

    @classmethod
    def build_robot(cls, mujoco_simulation: SType, physical):
        return mujoco_simulation.robot

    @property
    def robot(self):
        return self.mujoco_simulation.robot

    ###############################################################################################
    # Methods to ensure we handle sim recreation properly.
    def _initialize_sim_state(self):
        if self.parameters.robot_control_params.is_joint_actuated():
            for idx, eq_type in enumerate(self.mujoco_simulation.mj_sim.model.eq_type):
                if eq_type == MujocoEquality.mjEQ_WELD.value:
                    self.mujoco_simulation.mj_sim.model.eq_active[idx] = 0
        else:
            utils.reset_mocap_welds(self.sim)
            self.sim.forward()
            tcp_pos = self.mujoco_simulation.mj_sim.data.get_body_xpos(
                "robot0:gripper_base"
            )
            tcp_quat = self.sim.data.get_body_xquat("robot0:gripper_base")
            self.mujoco_simulation.mj_sim.data.set_mocap_pos("robot0:mocap", tcp_pos)
            self.mujoco_simulation.mj_sim.data.set_mocap_quat("robot0:mocap", tcp_quat)

            for _ in range(10):
                self.mujoco_simulation.step()
        self.robot.reset()

    def _randomize_object_initial_positions(self):
        """
        Randomize initial position for each object.
        """
        object_pos, is_valid = self._generate_object_placements()

        if not is_valid:
            raise InvalidSimulationError("Object placement is invalid.")

        self.mujoco_simulation.set_object_pos(object_pos)

    def _randomize_object_initial_states(self):
        # It's important to re-scale and rotate objects before placement since this
        # will influence their placement.
        self._randomize_object_initial_rotations()
        self._randomize_object_initial_positions()

    def _randomize_robot_initial_position(self):
        action = self.action_space.sample()

        if self.parameters.n_random_initial_steps < 1:
            return
        for _ in range(self.parameters.n_random_initial_steps):
            self._set_action(action)
            self.mujoco_simulation.step()
        self._set_action(action * 0.0)
        for _ in range(100):
            # calling set_action each tick is necessary for the robot to reach stability with relative actions
            self._set_action(action * 0.0)
            self.mujoco_simulation.step()

    def _randomize_object_groups(self, dedupe_objects: bool = False):
        """
        :param dedupe_objects: if set to True, every object is different, no duplicated ones.
        """
        object_groups = self._sample_attributed_object_groups(dedupe_objects)
        self._set_object_groups(object_groups)

    def _set_object_groups(self, object_groups: List[ObjectGroupConfig]):
        self.parameters.simulation_params.object_groups = object_groups

    def _sample_attributed_object_groups(self, dedupe_objects: bool = False):
        object_groups = self._sample_random_object_groups(dedupe_objects)
        attrs_per_group = self._sample_group_attributes(len(object_groups))
        return self._set_group_attributes(object_groups, attrs_per_group)

    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        """
        :param dedupe_objects: if set to True, every object is different, no duplicated ones.
        """
        num_objects = self.mujoco_simulation.num_objects

        if dedupe_objects:
            group_counts = [1] * num_objects
        else:
            group_counts = sample_group_counts(
                self._random_state,
                num_objects,
                lam_low=self.constants.sample_lam_low,
                lam_high=self.constants.sample_lam_high,
            )

        assert len(self.parameters.simulation_params.object_groups) > 0
        obj_group_config_type = type(self.parameters.simulation_params.object_groups[0])

        num_groups = len(group_counts)

        object_groups = []
        obj_id = 0
        for i in range(num_groups):
            # Initialize with object counts
            obj_group = obj_group_config_type(count=group_counts[i])

            # Set up object ids
            obj_group.object_ids = list(range(obj_id, obj_id + group_counts[i]))
            obj_id += group_counts[i]

            object_groups.append(obj_group)
        return object_groups

    def _set_group_attributes(
        self, object_groups: List[ObjectGroupConfig], attrs_per_group: Dict[str, list]
    ) -> List[ObjectGroupConfig]:
        assert len(object_groups) > 0
        obj_group_config_type = type(object_groups[0])

        num_groups = len(object_groups)
        for i in range(num_groups):
            # Set up scale, color & material args.
            for attr_name, attr_values in attrs_per_group.items():
                assert hasattr(
                    object_groups[i], attr_name
                ), f"{obj_group_config_type} has no attribute {attr_name}."
                setattr(object_groups[i], attr_name, attr_values[i])

        return object_groups

    def _sample_group_attributes(self, num_groups: int) -> Dict[str, list]:
        attrs_per_group = {
            "material_args": self._sample_object_materials(num_groups),
            "color": self._sample_object_colors(num_groups),
            "scale": self._sample_object_size_scales(num_groups),
        }
        assert all(len(v) == num_groups for k, v in attrs_per_group.items())
        return attrs_per_group

    def _sample_object_materials(self, num_groups: int) -> List[Dict[str, Any]]:
        if not self.parameters.material_names:
            self.parameters.material_names = load_all_materials()

        material_args = {
            m: load_material_args(m) for m in set(self.parameters.material_names)
        }
        names = self._random_state.choice(
            self.parameters.material_names, size=num_groups
        )
        return [material_args[name] for name in names]

    def _sample_object_colors(
        self, num_groups: int
    ) -> List[Union[np.ndarray, list, tuple]]:
        random_colors = self._random_state.random((num_groups, 4))
        random_colors[:, -1] = 1.0
        return random_colors

    def _sample_object_size_scales(self, num_groups: int):
        return np.exp(
            self._random_state.uniform(
                low=-self.parameters.object_scale_low,
                high=self.parameters.object_scale_high,
                size=num_groups,
            )
        )

    def _apply_object_colors(self):
        obj_groups = self.mujoco_simulation.object_groups
        obj_colors = [g.color.copy() for g in obj_groups for _ in range(g.count)]
        self.mujoco_simulation.set_object_colors(obj_colors)

    def _apply_object_size_scales(self):
        obj_groups = self.mujoco_simulation.object_groups
        obj_scales = [g.scale for g in obj_groups for _ in range(g.count)]
        self.mujoco_simulation.rescale_object_sizes(obj_scales)

    def _sample_default_quaternions(self):
        num_objects = self.mujoco_simulation.num_objects
        quats = rotation.quat_from_angle_and_axis(
            angle=self._random_state.uniform(0.0, 2 * np.pi, size=num_objects),
            axis=np.array([[0, 0, 1.0]] * num_objects),
        )
        assert quats.shape == (num_objects, 4)
        return quats

    def _randomize_object_initial_rotations(self):
        """
        Randomize initial rotation for each object.
        """
        quats = self._sample_object_initial_rotations()
        self._set_object_initial_rotations(quats)

    def _sample_object_initial_rotations(self):
        return self._sample_default_quaternions()

    def _set_object_initial_rotations(self, quats: np.ndarray):
        self.mujoco_simulation.set_object_quat(quats)
        self.mujoco_simulation.set_target_quat(quats)
        self.mujoco_simulation.forward()

    def _randomize_camera(self):
        """
        An reimplementation of jitter mode of ORRB::CameraRandomizer::RunComponent
        https://github.com/openai/orrb/blob/master/unity/Assets/Scripts/Randomizers/CameraRandomizer.cs#L73

        It is slightly different as it follows the curriculum defined by fovy/pos/quat radius.
        """
        nc = len(self.mujoco_simulation.initial_values["camera_fovy"])

        fovy_delta = (
            self._random_state.uniform(-1.0, 1.0, nc)
            * self.mujoco_simulation.simulation_params.camera_fovy_radius
        )

        # pos delta is sampled from a sphere with pos_radius distance away from original pos.
        pos_delta = [np.zeros(3)] * nc
        for i in range(nc):
            vec = self._random_state.randn(3)
            vec /= np.linalg.norm(vec)
            pos_delta[i] = vec
        pos_delta = (
            np.array(pos_delta)
            * self.mujoco_simulation.simulation_params.camera_pos_radius
        )

        """
        quat delta is sampled from fixed quat_radius (in radian) rotation around a uniform sampled axis,
        adapted from original c# code here, except the uniform sampling:
        Vector3 axis = Random.rotationUniform * Vector3.up;
        camera_state.camera.transform.localRotation = camera_state.rot *
            Quaternion.AngleAxis(Random.Range(rot_min, rot_max) * Mathf.Rad2Deg, axis);
        """
        quat_delta = [np.zeros(4)] * nc
        up = np.array([0, 1, 0])
        angle = self.mujoco_simulation.simulation_params.camera_quat_radius
        for i in range(nc):
            uniform_quat = rotation.uniform_quat(self._random_state)
            axis = rotation.quat_rot_vec(uniform_quat, up)
            quat_delta[i] = rotation.quat_from_angle_and_axis(angle, axis)

        self.mujoco_simulation.reset_camera(fovy_delta, pos_delta, quat_delta)

    def _randomize_lighting(self):
        """
        Randomizes the position and direction of all lights, and randomizes the ambient and diffuse
        headlight intensities.
        """

        # Controls the fraction of valid positions for the light which are able to be sampled.
        range_fraction = self.mujoco_simulation.simulation_params.light_pos_range

        positions = []
        directions = []
        n_lights = len(self.mujoco_simulation.get_light_positions())
        for i in range(n_lights):
            # Randomly sample (x, y, z) coordinates for position independently, uniformly randomly
            # from their valid range (which is modulated by `range_fraction`). Given these
            # coordinates, we then normalize the resulting position vector and scale it such that
            # the light is always 4m away from the origin. You can view these coordinates as points
            # on a surface of the sphere centered at the origin with radius 4m; this surface
            # initially is just the point (0, 0, 4) and then expands outwards according to
            # `range_fraction`.

            x = self._random_state.uniform(
                -0.25 * range_fraction, 0.75 * range_fraction
            )
            y = range_fraction * self._random_state.uniform(-4.0, 4.0)
            z = self._random_state.uniform(4.0 - (range_fraction * 4.0), 4.0)

            raw_pos = np.array([x, y, z])
            pos_norm = np.linalg.norm(raw_pos)
            # Keep the light 4 m away from the origin.
            pos = (raw_pos / pos_norm) * 4.0

            # Direction is unit-norm of the negative position vector
            direction = -raw_pos / pos_norm

            positions.append(pos)
            directions.append(direction)

        # Randomize the intensity of diffuse and ambient headlights.
        diffuse_intensity = (
            self.mujoco_simulation.simulation_params.light_diffuse_intensity
        )
        ambient_intensity = (
            self.mujoco_simulation.simulation_params.light_ambient_intensity
        )

        self.mujoco_simulation.set_lighting(
            np.array(positions),
            np.array(directions),
            diffuse_intensity,
            ambient_intensity,
        )

    def reset_goal(
        self, update_seed=False, sync_type=SyncType.RESET_GOAL, raise_when_invalid=True
    ):
        obs = super().reset_goal(update_seed=update_seed, sync_type=sync_type)

        # Check if goal placement is valid here.
        if not self._goal["goal_valid"]:
            if raise_when_invalid:
                raise InvalidSimulationError(
                    self._goal.get("goal_invalid_reason", "Goal is invalid.")
                )
            else:
                logger.info(
                    "InvalidSimulationError: "
                    + self._goal.get("goal_invalid_reason", "Goal is invalid.")
                )

        return obs

    def _get_simulation_info(self) -> dict:
        object_positions = self.mujoco_simulation.get_object_pos()[
            : self.mujoco_simulation.num_objects
        ]
        objects_off_table = self.mujoco_simulation.check_objects_off_table(
            object_positions
        )

        simulation_info = super()._get_simulation_info()
        simulation_info["objects_off_table"] = objects_off_table

        if "vision_cam_wrist" in self.constants.vision_args.mobile_camera_names:
            wrist_cam_contacts = self.mujoco_simulation.get_wrist_cam_collisions()
            simulation_info["wrist_cam_contacts"] = wrist_cam_contacts

        return simulation_info

    def _get_simulation_reward_with_done(self, info: dict) -> Tuple[float, bool]:
        reward, done = super()._get_simulation_reward_with_done(info)

        # If there is a gripper to table contact, apply table_collision penalty
        table_penalty = self.parameters.simulation_params.penalty.get(
            "table_collision", 0.0
        )

        if self.mujoco_simulation.get_gripper_table_contact():
            reward -= table_penalty

        # Add a large negative penalty for "breaking" the wrist camera by hitting it
        # with the table or another object.
        if info.get("wrist_cam_contacts", {}).get("any", False):
            reward -= self.parameters.simulation_params.penalty.get(
                "wrist_collision", 0.0
            )

        if "objects_off_table" in info and info["objects_off_table"].any():
            # If any object is off the table, we will end the episode
            done = True
            # Add a penalty to letting an object go off the table
            reward -= self.parameters.simulation_params.penalty.get(
                "objects_off_table", 0.0
            )
        if any(self.observe()["safety_stop"]):
            reward -= self.parameters.simulation_params.penalty.get("safety_stop", 0.0)
        return reward, done

    def _generate_object_placements(self) -> Tuple[np.ndarray, bool]:
        """
        Generate placement for each object. Return object placement and a
        boolean indicating whether the placement is valid.
        """
        placement, is_valid = place_objects_in_grid(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            random_state=self._random_state,
            max_num_trials=self.mujoco_simulation.max_placement_retry,
        )

        if not is_valid:
            # Fall back to random placement, which works better for envs with more irregular
            # objects (e.g. ycb-8 with no mesh normalization).
            return place_objects_with_no_constraint(
                self.mujoco_simulation.get_object_bounding_boxes(),
                self.mujoco_simulation.get_table_dimensions(),
                self.mujoco_simulation.get_placement_area(),
                max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
                max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
                random_state=self._random_state,
            )
        else:
            return placement, is_valid

    def _calculate_num_success(self, goal_distance) -> float:
        """
        This method calculates the reward summed over all objects, where we only consider a
        success if all goal distances are under the threshold.

        NOTE: This method takes into account objects that are not being used and appear as a 0.
        in goal_distance so it should only be used in situations where you're subtracting the
        output of this with 2 different goal_distance values.
        """
        per_goal_successful = np.stack(
            [
                goal_distance[k] < self.constants.success_threshold[k]
                for k in self.constants.success_threshold
            ],
            axis=0,
        )  # Dimensions [metric, object]
        per_goal_successful = np.all(per_goal_successful, axis=0)  # Dimensions [object]
        return np.sum(per_goal_successful) * self.constants.goal_reward_per_object

    def _calculate_goal_distance_reward(
        self, previous_goal_distance, goal_distance
    ) -> float:
        previous_num_success = self._calculate_num_success(previous_goal_distance)
        num_success = self._calculate_num_success(goal_distance)
        return num_success - previous_num_success

    def _recreate_sim(self) -> None:
        self.mujoco_simulation.update(
            self.build_simulation(self.constants, self.parameters)
        )
        self.mujoco_simulation.mj_sim.render_callback = self._render_callback
        self._setup_simulation_from_parameters()
        self._initialize_sim_state()

    def _render_callback(self, _sim, _viewer):
        super()._render_callback(_sim, _viewer)

        if not self.parameters.debug:
            return

        # Debug visualization of object bounding boxes.
        bounding_boxes = (
            self.mujoco_simulation.get_object_bounding_boxes_in_table_coordinates()
        )
        for idx, (pos, size) in enumerate(bounding_boxes):
            name = f"object{idx}"
            _viewer.add_marker(
                size=size, pos=pos, rgba=np.array([0, 0.5, 1, 0.1]), label=name
            )

        # Debug visualization of target bounding boxes.
        bounding_boxes = (
            self.mujoco_simulation.get_target_bounding_boxes_in_table_coordinates()
        )
        for idx, (pos, size) in enumerate(bounding_boxes):
            name = f"target:object{idx}"
            _viewer.add_marker(
                size=size, pos=pos, rgba=np.array([0.5, 0, 1, 0.1]), label=name
            )

        # Debug visualization of the placement area.
        (
            table_pos,
            table_size,
            table_height,
        ) = self.mujoco_simulation.get_table_dimensions()
        placement_area = self.mujoco_simulation.get_placement_area()
        size = np.array(placement_area.size) / 2.0
        pos = np.array([placement_area.offset]) + table_pos - table_size + size
        _viewer.add_marker(
            size=size, pos=pos, rgba=np.array([1, 0.5, 0, 0.1]), label="placement_area"
        )

    def _reset(self):
        if self.constants.use_fixed_seed:
            self.seed(self.seed())

        # This needs to happen before sim creation because sim creation depends
        # on the mujoco args generated from material randomization.
        self._randomize_object_groups()
        self._recreate_sim()
        self._apply_object_size_scales()
        self._apply_object_colors()

        self._randomize_object_initial_states()
        self._randomize_camera()
        self._randomize_lighting()

        if self.constants.stabilize_objects:
            stabilize_objects(self.mujoco_simulation)

        self.mujoco_simulation.forward()

        self._randomize_robot_initial_position()

        # Update `max_timesteps_per_goal` directly in the tracker. Do NOT update
        # `self.constants.max_timesteps_per_goal` here since it won't propagate correctly
        # to `MultiGoalTracker`.
        assert hasattr(self.multi_goal_tracker, "max_timesteps_per_goal")

        self.multi_goal_tracker.max_timesteps_per_goal = (
            self.constants.max_timesteps_per_goal_per_obj
            * self.mujoco_simulation.num_objects
        )

        # Reset the placement area boundary, a tuple of (min_x, min_y, min_z, max_x, max_y, max_z).
        self._placement_area_boundary = (
            self.mujoco_simulation.extract_placement_area_boundary()
        )

    def _act(self, action):
        if self.constants.teleport_to_goal and self.t > 10:
            # Override the policy by teleporting directly to the goal state (used to generate
            # a balanced distribution for training a goal classifier). Add some noise to the
            # objects' states; we tune the scale of the noise so that with probability p=0.5,
            # all objects are within the success_threshold. This will result in our episodes
            # containing a ~p/(1+p) fraction of goal-achieved states.
            target_pos = self.mujoco_simulation.get_target_pos(pad=False)
            target_quat = self.mujoco_simulation.get_target_quat(pad=False)

            num_objects = target_pos.shape[0]
            num_randomizations = num_objects * (
                ("obj_pos" in self.constants.success_threshold)
                + ("obj_rot" in self.constants.success_threshold)
            )
            assert num_randomizations > 0
            success_prob = 0.5 ** (1 / num_randomizations)

            # Add Gaussian noise to x and y position.
            if "obj_pos" in self.constants.success_threshold:
                # Tune the noise so that the position is within success_threshold with
                # probability success_prob. Note for example that noise_scale -> 0 when
                # success_prob -> 1, and noise_scale -> infinity when success_prob -> 0.
                noise_scale = np.ones_like(target_pos)
                noise_scale *= self.constants.success_threshold["obj_pos"]
                noise_scale /= np.sqrt(-2 * np.log(1 - success_prob))
                noise_scale[:, 2] = 0.0  # Don't add noise to the z-axis
                target_pos = np.random.normal(loc=target_pos, scale=noise_scale)

            # Add Gaussian noise to rotation about z-axis.
            if "obj_rot" in self.constants.success_threshold:
                # Tune the noise so that the rotation is within success_threshold with
                # probability success_prob. Note for example that noise_scale -> 0 when
                # success_prob -> 1, and noise_scale -> infinity when success_prob -> 0.
                noise_scale = self.constants.success_threshold["obj_rot"]
                noise_scale /= scipy.special.ndtri(
                    success_prob + (1 - success_prob) / 2
                )
                noise_quat = rotation.quat_from_angle_and_axis(
                    angle=np.random.normal(
                        loc=np.zeros((num_objects,)), scale=noise_scale
                    ),
                    axis=np.array([[0, 0, 1.0]] * num_objects),
                )
                target_quat = rotation.quat_mul(target_quat, noise_quat)

            self.mujoco_simulation.set_object_pos(target_pos)
            self.mujoco_simulation.set_object_quat(target_quat)
            self.mujoco_simulation.forward()
        else:
            self._set_action(action)

    def apply_wrappers(self, **wrapper_params):
        env = SmoothActionWrapper(self, alpha=0.3)
        env = ClipRewardWrapper(env)
        env = DiscretizeActionWrapper(
            env,
            n_action_bins=self.constants.n_action_bins,
            bin_spacing=self.constants.action_spacing,
        )

        return env

    @classmethod
    def build_goal_generation(cls, constants: CType, mujoco_simulation: SType):
        if constants.goal_generation == "train":
            return TrainStateGoal(mujoco_simulation, args=constants.goal_args)
        else:
            return ObjectStateGoal(mujoco_simulation, args=constants.goal_args)

    @classmethod
    def build_observation_randomizers(cls, constants: CType):
        return []

    @classmethod
    def build_simulation_randomizers(cls, constants):
        return [
            GravityRandomizer(),
            JointMarginRandomizer(),
            GenericSimRandomizer(
                name="dof_frictionloss_robot",
                field_name="dof_frictionloss",
                dof_jnt_prefix="robot0:",
                apply_mode="uncoupled_mean_variance",
            ),
            GenericSimRandomizer(
                name="dof_damping_robot",
                field_name="dof_damping",
                dof_jnt_prefix="robot0:",
                apply_mode="uncoupled_mean_variance",
            ),
            GenericSimRandomizer(
                name="dof_armature_robot",
                field_name="dof_armature",
                dof_jnt_prefix="robot0:",
                apply_mode="uncoupled_mean_variance",
            ),
            GenericSimRandomizer(
                name="jnt_stiffness_robot",
                field_name="jnt_stiffness",
                jnt_prefix="robot0:",
                apply_mode="variance_mean_additive",
                coef=0.005,
                positive_only=True,
            ),
            GenericSimRandomizer(
                name="body_pos_robot",
                field_name="body_pos",
                body_prefix="robot0:",
                apply_mode="variance_additive",
                coef=0.02,
            ),
            PidRandomizer("pid_kp"),
            PidRandomizer("pid_ti"),
            PidRandomizer("pid_td"),
            PidRandomizer("pid_imax_clamp"),
            PidRandomizer("pid_error_deadband"),
            GenericSimRandomizer(
                name="actuator_forcerange",
                field_name="actuator_forcerange",
                apply_mode="uncoupled_mean_variance",
            ),
            GeomSolimpRandomizer(),
            GeomSolrefRandomizer(),
            GenericSimRandomizer(
                name="geom_margin",
                field_name="geom_margin",
                apply_mode="variance_additive",
                coef=0.0005,
            ),
            GenericSimRandomizer(
                name="geom_pos",
                field_name="geom_pos",
                apply_mode="variance_additive",
                coef=0.002,
            ),
            GenericSimRandomizer(
                name="geom_gap",
                field_name="geom_gap",
                apply_mode="max_additive",
                coef=0.01,
            ),
            GenericSimRandomizer(
                name="geom_friction",
                field_name="geom_friction",
                apply_mode="uncoupled_mean_variance",
            ),
            GenericSimRandomizer(
                name="body_mass",
                field_name="body_mass",
                apply_mode="uncoupled_mean_variance",
                zero_threshold=0.208,
            ),
            GenericSimRandomizer(
                name="body_inertia",
                field_name="body_inertia",
                apply_mode="variance_additive",
            ),
        ]
