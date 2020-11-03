from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import attr
import numpy as np
from gym.envs.robotics import rotations
from mujoco_py import const

from robogym.envs.rearrange.common.utils import (
    PlacementArea,
    geom_ids_of_body,
    get_all_vertices,
    mesh_vert_range_of_geom,
)
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.mujoco.simulation_interface import (
    SimulationInterface,
    SimulationParameters,
)
from robogym.randomization.env import build_randomizable_param
from robogym.robot.composite.ur_gripper_arm import build_composite_robot
from robogym.robot.robot_interface import RobotControlParameters
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface
from robogym.utils import rotation


class Meta(type):
    @classmethod
    def __prepare__(mcs, name, bases, **kwds):
        """
        A metaclass to force slots to be present in all subclasses. This
        is necessary to guarantee update always work correctly for all subclasses.

        See https://docs.python.org/2.5/ref/slots.html for details about __slots__
        """
        super_prepared = super().__prepare__(mcs, name, bases, **kwds)
        super_prepared["__slots__"] = ()
        return super_prepared


@attr.s(auto_attribs=True)
class ObjectGroupConfig:
    # Total number of objects with this config.
    count: int = 1

    object_ids: List[int] = [0]

    # Scale for object.
    scale: float = 1.0

    # Material args for each object. They are defined as mapping from tag name
    # to property values.
    material_args: Dict[str, Dict[str, Any]] = {}

    # color in RGBA
    color: np.ndarray = attr.ib(
        default=np.array([1.0, 1.0, 1.0, 0.2]), converter=np.array
    )  # type: ignore

    @color.validator
    def validate_color(self, _, value):
        assert isinstance(value, np.ndarray) and value.shape == (4,)


@attr.s(auto_attribs=True)
class RearrangeSimParameters(SimulationParameters):
    num_objects: int = build_randomizable_param(1, low=1, high=32)

    # The object size in half-size (as per Mujoco convention).
    object_size: float = build_randomizable_param(0.0254, low=0.01, high=0.1)

    # If not set, we use num_objects as max_num_objects.
    max_num_objects: Optional[int] = None

    # Max percent of table area to use.
    used_table_portion: float = build_randomizable_param(1.0, low=0.4, high=1.0)

    # Max number of times for placement retry. To place multiple objects, placement algorithm
    # place each object one by one. Whenever placing a new object is failed the placement
    # algorithm retry `max_placement_retry_per_object` times and if it still fails it start over
    # from the first object. The placement can start over for `max_placement_retry` times.
    max_placement_retry: int = 100
    max_placement_retry_per_object: int = 20

    # The `goal_distance_ratio` is used to decrease distance from object position to goal position
    # after uniformly sampling the goal. The decreased distance will be `goal_distance_ratio *
    # original_distance`. The distance cannot be decreased below `goal_distance_min`.
    goal_distance_ratio: float = build_randomizable_param(1.0, low=0.0, high=1.0)
    goal_distance_min: float = 0.06

    # The offset put on the relative positional distance between target and object. This can be
    # used to make the success threshold for object position different less strict. By adding
    # negative offset to relative positional distance, we can enjoy the effect of increasing the
    # success threshold by -goal_pos_offset.
    goal_pos_offset: float = build_randomizable_param(0.0, low=-0.04, high=0.0)

    # The weight put on the relative rotational distance between target and object. This can
    # be used to smoothly interpolate between not caring about rotational goals (when value = 0)
    # to fully caring about them (when value = 1).
    goal_rot_weight: float = build_randomizable_param(1.0, low=0.0, high=1.0)

    # Height of reach target object above the table, curriculum from high to low
    # helps policy learn to handle safety plane in z.
    target_height: float = build_randomizable_param(0.1, low=0.0, high=0.2)

    # Collection of penalties to impose on sim
    # Penalties are defined as positive numbers and are subtracted from
    # the reward.
    penalty: dict = dict(
        table_collision=0.0, objects_off_table=1.0, wrist_collision=0.0
    )

    # for camera view randomization (fov, pos and quat)
    camera_fovy_radius: float = build_randomizable_param(0.0, low=0, high=0.1)
    camera_pos_radius: float = build_randomizable_param(0.0, low=0, high=0.01)

    # about 5 degrees
    camera_quat_radius: float = build_randomizable_param(0.0, low=0, high=0.09)

    # Whether we should cast shadows. This is disabled by default because it slows down rendering
    # signficantly when using envs with custom meshes. We enable it for sim2real experiments.
    cast_shadows: bool = False

    # Lighting randomizations.
    light_pos_range: float = build_randomizable_param(0.0, low=0.0, high=0.8)
    light_ambient_intensity: float = build_randomizable_param(0.1, low=0.1, high=0.7)
    light_diffuse_intensity: float = build_randomizable_param(0.4, low=0.1, high=0.7)

    # Objects assigned with the same group id should be visually same (same shape, size, color).
    # We always assume group id starting from 0 and new ones are assigned incrementally.
    object_groups: List[ObjectGroupConfig] = None  # type: ignore

    def __attrs_post_init__(self):
        # Assign default value to object_groups if empty. Hard to use @object_groups.default here
        # as it depends on self.num_objects and the order is messed up in different subclasses.
        if self.object_groups is None:
            self.object_groups = [
                ObjectGroupConfig(count=1, object_ids=[obj_id])
                for obj_id in range(self.num_objects)
            ]


PType = TypeVar("PType", bound=RearrangeSimParameters)


class RearrangeSimulationInterface(
    ArmSimulationInterface, Generic[PType], metaclass=Meta
):
    """
    Creates a SimulationInterface with a rearrange-compatible robot-gripper and a
    table setup. Subclass this and implement make_objects_xml() to create other tasks.
    """

    # For rearrange task, we recreate simulation during environment reset, in order to avoid
    # environment component still referring to stale simulation instance, an update() method
    # is added to allow updating simulation in place. For update() to work properly
    # (called when underlying MjModel / XML is changed), the instance variables to be updated
    # must be registered somewhere (for simplicity, all instance variables are assumed to be
    # simulation dependent, and thus require updating). We use __slots__ to register the instance
    # variables. So you will need to add all simulation instance variables to slots,
    # or you'll get an error.
    __slots__ = ["robot", "simulation_params", "control_param", "initial_values"]

    def __init__(
        self,
        sim,
        robot_control_params: RobotControlParameters,
        simulation_params: PType,
    ):
        super().__init__(sim, robot_control_params=robot_control_params)

        self.simulation_params = simulation_params

        self.robot = build_composite_robot(
            robot_control_params=robot_control_params, simulation=self
        )

        for i in range(self.num_objects):
            self.register_joint_group(f"object{i}", prefix=[f"object{i}:"])
            # Verify the qpos and qvel arrays are the proper lengths (there was previously a bug
            # which caused this to not be the case when > 10 objects existed).
            assert len(self.get_qpos(f"object{i}")) == 7
            assert len(self.get_qvel(f"object{i}")) == 6

        self.initial_values = dict()
        self.initial_values["camera_fovy"] = self.mj_sim.model.cam_fovy.copy()
        self.initial_values["camera_pos"] = self.mj_sim.model.cam_pos.copy()
        self.initial_values["camera_quat"] = self.mj_sim.model.cam_quat.copy()

        self.set_object_colors()

        self.mj_sim.model.light_castshadow[:] = (
            1.0 if simulation_params.cast_shadows else 0.0
        )

    @property
    def num_objects(self):
        return self.simulation_params.num_objects

    @property
    def object_size(self):
        return self.simulation_params.object_size

    @property
    def goal_distance_ratio(self):
        return self.simulation_params.goal_distance_ratio

    @property
    def goal_distance_min(self):
        return self.simulation_params.goal_distance_min

    @property
    def goal_pos_offset(self):
        return self.simulation_params.goal_pos_offset

    @property
    def goal_rot_weight(self):
        return self.simulation_params.goal_rot_weight

    @property
    def max_num_objects(self):
        return (
            self.simulation_params.max_num_objects or self.simulation_params.num_objects
        )

    @property
    def used_table_portion(self):
        return self.simulation_params.used_table_portion

    @property
    def max_placement_retry(self):
        return self.simulation_params.max_placement_retry

    @property
    def max_placement_retry_per_object(self):
        return self.simulation_params.max_placement_retry_per_object

    @property
    def object_groups(self):
        return self.simulation_params.object_groups

    @property
    def num_groups(self):
        return len(self.object_groups)

    @classmethod
    def _sanity_check_object_groups(cls, sim_params: PType):
        # sanity check
        assert sim_params.object_groups
        assert (
            sum([g.count for g in sim_params.object_groups]) == sim_params.num_objects
        )
        assert [i for g in sim_params.object_groups for i in g.object_ids] == list(
            range(sim_params.num_objects)
        )

    @classmethod
    def build(
        cls,
        robot_control_params: RobotControlParameters,
        n_substeps: int = 20,
        mujoco_timestep: float = 0.002,
        simulation_params: Optional[
            PType
        ] = None,  # optional is required to keep parent signature compatibility
    ):
        assert (
            simulation_params is not None
        )  # we do not actually support calling without valid params

        xml = cls.make_xml(simulation_params, mujoco_timestep)
        xml = cls.make_robot_xml(xml, robot_control_params)

        return cls(
            xml.build(nsubsteps=n_substeps),
            robot_control_params=robot_control_params,
            simulation_params=simulation_params,
        )

    @classmethod
    def make_xml(cls, simulation_params: PType, mujoco_timestep: float):
        cls._sanity_check_object_groups(simulation_params)

        xml = cls.make_world_xml(
            simulation_params=simulation_params,
            contact_params={},
            mujoco_timestep=mujoco_timestep,
        )
        object_and_target_xmls = cls.make_objects_xml(xml, simulation_params)
        object_group_ids = [
            i
            for i, obj_group in enumerate(simulation_params.object_groups)
            for _ in range(obj_group.count)
        ]

        for group_id, (obj_xml, target_xml) in zip(
            object_group_ids, object_and_target_xmls
        ):
            obj_xml.set_objects_attrs(
                simulation_params.object_groups[group_id].material_args
            )
            xml.append(obj_xml)
            xml.append(target_xml)

        return xml

    @classmethod
    def make_world_xml(cls, *, mujoco_timestep: float, **kwargs):
        return super().make_world_xml(
            contact_params=dict(
                njmax=2000, nconmax=500, nuserdata=2000, nuser_actuator=16
            ),
            mujoco_timestep=mujoco_timestep,
        )

    @classmethod
    def make_objects_xml(
        cls, xml, simulation_params: PType
    ) -> List[Tuple[MujocoXML, MujocoXML]]:
        """
        Return list of (object xml, target xml) tuples.
        """
        return []

    def update(self, other: "RearrangeSimulationInterface"):
        """
        Update simulation state from other simulation instance. This is implemented
        by recursive set slot attributes along the ancestor chain.
        """
        self.mj_sim.set_stale()  # Mark current MjSim instance as stale.
        return self._update(other, self.__class__)

    def _update(self, other, current_class: Type[SimulationInterface]):
        """
        Helper function to recursively set slots.
        """
        for slot in current_class.__slots__:
            setattr(self, slot, getattr(other, slot))

        if current_class != SimulationInterface:
            parent_class = current_class.__bases__[0]
            assert issubclass(parent_class, SimulationInterface)
            self._update(other, parent_class)

    @contextmanager
    def hide_target(self, hide_robot=False):
        """
        A context manager in scope of which all target objects are hidden.
        """
        return self._hide_geoms(
            hide_targets=True, hide_objects=False, hide_robot=hide_robot
        )

    @contextmanager
    def hide_objects(self, hide_robot=False):
        """
        A context manager in scope of which all objects and target objects are hidden.
        """
        return self._hide_geoms(
            hide_targets=True, hide_objects=True, hide_robot=hide_robot
        )

    def _hide_geoms(self, hide_targets=False, hide_objects=False, hide_robot=False):
        sim = self.mj_sim

        geom_ids_to_hide = []

        # Hide sites.
        site_rgba = sim.model.site_rgba.copy()
        sim.model.site_rgba[:] = np.zeros_like(site_rgba)

        if hide_targets:
            # Hide targets
            target_ids = [
                target_id
                for i in range(self.num_objects)
                for target_id in geom_ids_of_body(sim, f"target:object{i}")
            ]

            assert len(target_ids) > 0

            geom_ids_to_hide += target_ids

        if hide_objects:
            # Hide targets
            object_ids = [
                target_id
                for i in range(self.num_objects)
                for target_id in geom_ids_of_body(sim, f"object{i}")
            ]

            assert len(object_ids) > 0

            geom_ids_to_hide += object_ids

        if hide_robot:
            robot_geom_ids = [
                sim.model.geom_name2id(name)
                for name in sim.model.geom_names
                if name.startswith("robot0:")
            ]
            geom_ids_to_hide += robot_geom_ids

        geom_rgba = sim.model.geom_rgba.copy()
        sim.model.geom_rgba[geom_ids_to_hide, -1] = 0.0

        yield

        # If sim becomes stale (e.g. because of reset while being yield), we don't have to reset
        # it to the original state
        if not sim.is_stale():
            # Restore sites and targets
            sim.model.geom_rgba[:] = geom_rgba
            sim.model.site_rgba[:] = site_rgba

    ##############################################################
    # Methods to get object related observations.

    def get_object_pos(self, pad=True) -> np.ndarray:
        """
        Get position for all objects.
        """
        return self._get_object_obs(self.mj_sim.data.get_body_xpos, pad=pad)

    def get_object_rel_pos(self) -> np.ndarray:
        """
        Get position for all objects relative to the gripper position.
        """
        gripper_pos = self.robot.observe().tcp_xyz()  # type: ignore
        return self._get_object_obs(
            lambda n: self.mj_sim.data.get_body_xpos(n) - gripper_pos
        )

    def get_object_quat(self, pad=True) -> np.ndarray:
        """
        Get rotation in quaternion for all objects.
        """
        return self._get_object_obs(
            lambda n: rotation.quat_normalize(
                rotations.mat2quat(self.mj_sim.data.get_body_xmat(n))
            ),
            pad=pad,
        )

    def get_object_rot(self, pad=True) -> np.ndarray:
        """
        Get rotation in euler angles for all objects.
        """
        return self._get_object_obs(
            lambda n: rotations.normalize_angles(
                rotations.mat2euler(self.mj_sim.data.get_body_xmat(n))
            ),
            pad=pad,
        )

    def get_object_vel_pos(self):
        """
        Get position velocity for all objects relative to tooltip velocity.
        """
        robot_obs = self.robot.observe()
        tooltip_vel = robot_obs.tcp_vel()

        return self._get_object_obs(
            lambda n: self.mj_sim.data.get_body_xvelp(n) - tooltip_vel
        )

    def get_object_vel_rot(self):
        """
        Get rotation velocity for all objects.
        """
        return self._get_object_obs(lambda n: self.mj_sim.data.get_body_xvelr(n))

    def get_target_pos(self, pad=True) -> np.ndarray:
        """
        Get target position for all objects.
        """
        return self._get_target_obs(self.mj_sim.data.get_body_xpos, pad=pad)

    def get_target_quat(self, pad=True) -> np.ndarray:
        """
        Get target rotation in quaternion for all objects.
        """
        return self._get_target_obs(
            lambda n: rotation.quat_normalize(
                rotations.mat2quat(self.mj_sim.data.get_body_xmat(n))
            ),
            pad=pad,
        )

    def get_target_rot(self, pad=True) -> np.ndarray:
        """
        Get target rotation in euler angles for all objects.
        """
        return self._get_target_obs(
            lambda n: rotations.mat2euler(self.mj_sim.data.get_body_xmat(n)), pad=pad
        )

    def get_object_bounding_box_sizes(self, pad=True) -> np.ndarray:
        return self._get_object_obs(lambda n: self._get_bounding_box(n)[1], pad=pad)

    def get_object_vertices(self, subdivide_threshold=None) -> List[np.ndarray]:
        """
        Get vertices for all objects.

        :param subdivide_threshold: If specified, subdivide mesh according to this threshold.
        """
        return [
            get_all_vertices(
                self.mj_sim, f"object{i}", subdivide_threshold=subdivide_threshold
            )
            for i in range(self.num_objects)
        ]

    def get_target_vertices(self, subdivide_threshold=None) -> List[np.ndarray]:
        """
        Get vertices for all objects.

        :param subdivide_threshold: If specified, subdivide mesh according to this threshold.
        """
        return [
            get_all_vertices(
                self.mj_sim,
                f"target:object{i}",
                subdivide_threshold=subdivide_threshold,
            )
            for i in range(self.num_objects)
        ]

    def get_object_damping(self) -> np.ndarray:
        """
        Get dumping value for each object
        """

        def _get_object_damping(object_name):
            joint_name = f"{object_name}:joint"
            joint_id = self.mj_sim.model.joint_name2id(joint_name)
            dof_ids = [
                idx
                for idx in range(self.mj_sim.model.nv)
                if self.mj_sim.model.dof_jntid[idx] == joint_id
            ]

            return self.mj_sim.model.dof_damping[dof_ids]

        return self._get_object_obs(_get_object_damping, pad=False)

    def get_gripper_geom_ids(self) -> List[int]:
        """
        Get gripper geom ids for the two gripper contacts
        """

        def _get_geom_id(name):
            return self.mj_sim.model.geom_name2id(name)

        # Looking for UR gripper
        l_finger_geom_id = _get_geom_id("robot0:left_contact_v")
        r_finger_geom_id = _get_geom_id("robot0:right_contact_v")

        return [l_finger_geom_id, r_finger_geom_id]

    def get_wrist_cam_collisions(self):
        """
        Get geometries contacted by the wrist camera. Group these geometries into
        categories of bodies - namely the environment table_collision_plane, the robot itself, or
        other objects.
        """

        geom_id = self.mj_sim.model.geom_name2id("robot0:wrist_cam_collision_sphere")
        contacts = {"table_collision_plane": False, "robot": False, "object": False}

        def _map_contact_to_group(geom: Optional[str]):
            """
            Map the name of a geometry to the type of contact (table_collision_plane, robot, or object).
            Note that in holdout envs many objects don't have names (represented as None).
            """
            if geom == "table_collision_plane":
                return geom
            elif geom is not None and geom.startswith("robot0:"):
                return "robot"
            return "object"

        for i in range(self.mj_sim.data.ncon):
            c = self.mj_sim.data.contact[i]
            geoms = [c.geom1, c.geom2]

            if geom_id in geoms:
                geoms.remove(geom_id)
                geom_name = self.mj_sim.model.geom_id2name(geoms[0])
                contacts[_map_contact_to_group(geom_name)] = True

        contacts["any"] = any(contacts.values())
        return contacts

    def get_object_gripper_contact(
        self,
        other_geom_ids: Optional[List[int]] = None,
        dist_cutoff: float = 1.0e-5,
        pad: bool = True,
    ) -> np.ndarray:
        """
        Check whether each object has contact with the provided geom ids.

        :param other_geom_ids: If none, use the left and right gripper by default.
        :param dist_cutoff: Only when the contact penetration is smaller than this threshold,
            the contact is counted. This threshold should be small.
            Note: The distances this is compared to are all negative; the default setting of
            1e-5 means we include all contacts (is equivalent to setting this to 0).
        :param pad: whether pad the results to have the length same as max_num_objects.
        :return: a numpy array of shape [num objects, len(other_geom_ids)], in which each
            value is binary, 1 meaning having contact and 0 no contact.
        """
        if other_geom_ids is None:
            other_geom_ids = self.get_gripper_geom_ids()

        contact_dict: Dict[int, set] = {geom_id: set() for geom_id in other_geom_ids}

        # Only mjData.ncon elements are in data.contact at a given time.
        # The rest is left over from previous iterations and is not used.
        for i in range(self.mj_sim.data.ncon):
            c = self.mj_sim.data.contact[i]
            if c.dist < dist_cutoff:
                if c.geom1 in other_geom_ids:
                    contact_dict[c.geom1].add(c.geom2)
                if c.geom2 in other_geom_ids:
                    contact_dict[c.geom2].add(c.geom1)

        def _get_object_contact(object_name):
            geom_ids = geom_ids_of_body(self.mj_sim, object_name)
            return [
                float(any(i in contact_dict[other_id] for i in geom_ids))
                for other_id in other_geom_ids
            ]

        return self._get_object_obs(_get_object_contact, pad=pad)

    def get_light_positions(self) -> np.ndarray:
        return self.mj_sim.model.light_pos

    def get_object_colors(self, pad=True) -> np.ndarray:
        """Get object colors.
        This logic works, assuming we only assign a single color to one object.
        """
        return self._get_object_obs(
            lambda n: self.mj_sim.model.geom_rgba[geom_ids_of_body(self.mj_sim, n)[0]],
            pad=pad,
        )

    ##############################################################
    # Methods related to objects state.

    def set_target_pos(self, target_positions: np.ndarray):
        assert target_positions.shape == (self.num_objects, 3), (
            f"Incorrect target_positions.shape {target_positions.shape}, "
            f"which should be {(self.num_objects, 3)}."
        )

        for i in range(self.num_objects):
            target_id = self.mj_sim.model.body_name2id(f"target:object{i}")
            self.mj_sim.model.body_pos[target_id][:] = target_positions[i]

    def set_target_quat(self, target_quats: np.ndarray):
        assert target_quats.shape == (self.num_objects, 4), (
            f"Incorrect target_quats.shape {target_quats.shape}, "
            f"which should be {(self.num_objects, 4)}."
        )

        target_quats = rotation.quat_normalize(target_quats)
        for i in range(self.num_objects):
            target_id = self.mj_sim.model.body_name2id(f"target:object{i}")
            self.mj_sim.model.body_quat[target_id][:] = target_quats[i]

    def set_target_rot(self, target_rots: np.ndarray):
        assert target_rots.shape == (self.num_objects, 3), (
            f"Incorrect target_rots.shape {target_rots.shape}, "
            f"which should be {(self.num_objects, 3)}."
        )
        self.set_target_quat(rotation.euler2quat(target_rots))

    def set_object_pos(self, object_positions: np.ndarray):
        assert object_positions.shape == (self.num_objects, 3), (
            f"Incorrect object_positions.shape {object_positions.shape}, "
            f"which should be {(self.num_objects, 3)}."
        )

        for i in range(self.num_objects):
            joint_name = f"object{i}:joint"
            joint_qpos = self.mj_sim.data.get_joint_qpos(joint_name)
            joint_qpos[:3] = object_positions[i]
            self.mj_sim.data.set_joint_qpos(joint_name, joint_qpos)

    def set_object_quat(self, object_quats: np.ndarray):
        assert object_quats.shape == (self.num_objects, 4), (
            f"Incorrect object_quats.shape {object_quats.shape}, "
            f"which should be {(self.num_objects, 4)}."
        )

        object_quats = rotation.quat_normalize(object_quats)
        for i in range(self.num_objects):
            joint_name = f"object{i}:joint"
            joint_qpos = self.mj_sim.data.get_joint_qpos(joint_name)
            joint_qpos[3:] = object_quats[i]
            self.mj_sim.data.set_joint_qpos(joint_name, joint_qpos)

    def set_object_rot(self, object_rots: np.ndarray):
        assert object_rots.shape == (self.num_objects, 3), (
            f"Incorrect object_rots.shape {object_rots.shape}, "
            f"which should be {(self.num_objects, 3)}."
        )
        self.set_object_quat(rotation.euler2quat(object_rots))

    def rescale_object_sizes(self, object_scales):
        assert (
            len(object_scales) == self.num_objects
        ), f"Incorrect number of scales: {len(object_scales)}, should be {self.num_objects}."
        for i in range(self.num_objects):
            object_scale = object_scales[i]
            self._rescale_object(f"object{i}", object_scale)
            self._rescale_object(f"target:object{i}", object_scale)

    def _rescale_object(self, object_name, object_scale):
        geom_ids = geom_ids_of_body(self.mj_sim, object_name)
        for geom_id in geom_ids:
            self.mj_sim.model.geom_pos[geom_id, :] *= object_scale
            if self.mj_sim.model.geom_type[geom_id] == const.GEOM_MESH:
                self.mj_sim.model.mesh_vert[
                    mesh_vert_range_of_geom(self.mj_sim, geom_id)
                ] *= object_scale
            else:
                self.mj_sim.model.geom_size[geom_id, :] *= object_scale

    def set_object_colors(self, colors=None):
        """
        Set color for each object and its corresponding target. If colors are not
        provided, use default color in current model.
        """
        assert (
            colors is None or len(colors) == self.num_objects
        ), f"Incorrect number of colors: {len(colors)}, should be {self.num_objects}."

        for i in range(self.num_objects):
            object_geom_ids = geom_ids_of_body(self.mj_sim, f"object{i}")
            target_geom_ids = geom_ids_of_body(self.mj_sim, f"target:object{i}")

            if colors is not None:
                color = list(colors[i])

                self.mj_sim.model.geom_rgba[object_geom_ids, :] = color
                self.mj_sim.model.geom_rgba[target_geom_ids, :] = color

            self.mj_sim.model.geom_rgba[target_geom_ids, -1] = 0.2

    def set_object_damping(self, damping: Union[float, np.ndarray]):
        if isinstance(damping, float):
            damping = np.full(self.num_objects, damping)

        assert len(damping) == self.num_objects, (
            f"Incorrect number of objects: {len(damping)}, "
            f"should be {self.num_objects}."
        )

        for i in range(self.num_objects):
            joint_name = f"object{i}:joint"
            joint_id = self.mj_sim.model.joint_name2id(joint_name)
            dof_ids = [
                idx
                for idx in range(self.mj_sim.model.nv)
                if self.mj_sim.model.dof_jntid[idx] == joint_id
            ]
            self.mj_sim.model.dof_damping[dof_ids] = damping[i]

        self.forward()

    def set_lighting(
        self,
        positions: np.ndarray,
        directions: np.ndarray,
        headlight_diffuse: float,
        headlight_ambient: float,
    ):
        n_lights = len(self.mj_sim.model.light_pos)
        assert n_lights == len(positions) == len(directions)

        self.mj_sim.model.light_pos[:] = positions
        self.mj_sim.model.light_dir[:] = directions

        self.mj_sim.model.vis.headlight.diffuse[:] = headlight_diffuse
        self.mj_sim.model.vis.headlight.ambient[:] = headlight_ambient

    def reset_camera(self, fov_delta, pos_delta, quat_delta):
        nc = len(self.initial_values["camera_fovy"])
        for i in range(nc):
            self.mj_sim.model.cam_fovy[i] = (
                self.initial_values["camera_fovy"][i] + fov_delta[i]
            )
            self.mj_sim.model.cam_pos[i] = (
                self.initial_values["camera_pos"][i] + pos_delta[i]
            )
            self.mj_sim.model.cam_quat[i] = rotations.quat_mul(
                self.initial_values["camera_quat"][i], quat_delta[i]
            )

    ##############################################################
    # Methods related to object placement.
    def check_objects_off_table(
        self, object_pos: np.ndarray, hor_margin: float = 0.0
    ) -> np.ndarray:
        """
        Calculate, per object, if object is off the table
        :param object_pos: matrix of object positions (num_objects, 3)
        :param hor_margin: how much margin of error we consider in the horizontal projection
            (x,y) before saying whether an object is out of the table
        :return: np.ndarray of shape (num_objects) with bolean values if objects are off the table
        """
        assert object_pos.shape == (self.num_objects, 3)

        table_pos, table_size, table_height = self.get_table_dimensions()
        min_x, min_y, _ = table_pos - table_size
        max_x, max_y, _ = table_pos + table_size

        # Stack per dimension instead of per object
        x_pos, y_pos, z_pos = np.stack(object_pos, axis=-1)

        return np.logical_or.reduce(
            (
                z_pos < table_height * 0.75,
                x_pos < min_x - hor_margin,
                x_pos > max_x + hor_margin,
                y_pos < min_y - hor_margin,
                y_pos > max_y + hor_margin,
            )
        )

    def extract_placement_area_boundary(self) -> tuple:
        """Extract the boundary of the placement area in 3d space.
        """
        table_pos, table_size, table_height = self.get_table_dimensions()
        placement_area = self.get_placement_area()
        size = np.array(placement_area.size) / 2
        pos = np.array(placement_area.offset) + table_pos - table_size + size

        min_x, min_y, min_z = pos - size
        max_x, max_y, max_z = pos + size

        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def check_objects_in_placement_area(
        self,
        object_pos: np.ndarray,
        margin: float = 0.02,
        soft: bool = False,
        placement_area_boundary: Optional[tuple] = None,
    ) -> np.array:
        """
        Calculate, per object, if object is within the placement area. If yes it is 1.0 otherwise 0.0.

        :param object_pos: matrix of object positions of shape (max_num_objects, 3) or (num_objects, 3).
        :param margin: how much margin of error we consider in all three dimensions.
        :param soft: if true, we will label an object within the margin area stochastically.
        :param placement_area_boundary: min & max value in each dimension in 3D space. If not provided,
            we would calculate it on the fly.

        :return: np.ndarray of shape (object_pos.shape[0], ) with boolean values if objects are within
            the placement area. Returns True per object if it is in the placement area.
        """
        if object_pos.shape[0] == self.max_num_objects:
            assert object_pos.shape == (self.max_num_objects, 3)
            object_pos = object_pos[: self.num_objects]
            pad = True
        else:
            assert object_pos.shape == (self.num_objects, 3)
            pad = False

        if placement_area_boundary is None:
            placement_area_boundary = self.extract_placement_area_boundary()
        min_x, min_y, min_z, max_x, max_y, max_z = placement_area_boundary
        val_min = np.array([min_x, min_y, min_z])
        val_max = np.array([max_x, max_y, max_z])

        # Compute the distance per dimension to the placement boundary if out of the area.
        dist = np.maximum(object_pos - val_max, val_min - object_pos)
        dist = np.maximum(dist, np.zeros_like(dist))
        assert dist.shape == (self.num_objects, 3) and np.all(dist >= 0.0)
        max_dist = np.max(dist, axis=-1)
        assert max_dist.shape == (self.num_objects,)

        if soft:
            mask_proba = np.clip(max_dist / margin, 0.0, 1.0)
            mask = np.random.random() > mask_proba
        else:
            mask = max_dist < margin

        if pad:
            padding = np.array(
                [True] * (self.max_num_objects - self.num_objects), dtype=np.bool
            )
            mask = np.concatenate([mask, padding])
            assert mask.shape == (self.max_num_objects,)
        else:
            assert mask.shape == (self.num_objects,)

        return mask

    @classmethod
    def compute_table_dimension(cls, table_pos, table_size):
        table_height = table_size[-1] + table_pos[-1]
        return table_pos, table_size, table_height

    @classmethod
    def get_table_dimensions_from_xml(cls, xml):
        """Get dimensions of the table in the world."""
        # note that this makes a few assumptions on the base xml returned
        # by make_world_xml. If that XML changes, then this might break...
        table_geom_el = xml.root_element.find(".//geom[@name='table']")
        assert table_geom_el is not None, "Couldn't find table geom in XML"
        table_size = np.fromstring(table_geom_el.attrib["size"], sep=" ")

        table_body_el = xml.root_element.find(".//body[@name='table']")
        assert table_body_el is not None, "Couldn't find table body in XML"
        table_pos = np.fromstring(table_body_el.attrib["pos"], sep=" ")
        return cls.compute_table_dimension(table_pos, table_size)

    def get_table_dimensions(self):
        """
        Returns the dimensions of the table: First position, then size
        (as half-size, as per MuJoCo contentions) and finally the table height explicitly.
        """
        table_geom_id = self.mj_sim.model.geom_name2id("table")
        table_size = self.mj_sim.model.geom_size[table_geom_id].copy()
        table_body_id = self.mj_sim.model.body_name2id("table")
        table_pos = self.mj_sim.model.body_pos[table_body_id].copy()
        return self.compute_table_dimension(table_pos, table_size)

    def _get_bounding_box(self, object_name: str) -> Tuple:
        """
        Returns the bounding box for one objects as a tuple of (positive, half size),
        where both positive and half size are np.array of shape (3,).
        """
        raise NotImplementedError()

    def get_object_bounding_boxes(self) -> np.ndarray:
        """
        Returns bounding boxes for all the objects as an np.array of shape (num_objects, 2, 3).
        where [:, 0, :] contains the center position of the bounding box in Cartesian space
        relative to the body's frame of reference and where [:, 1, :] contains the half-width,
        half-height, and half-depth of the object (as per Mujoco convention).

        The reason why we return a position here is that more complicated bodies will have
        a bounding box that is not necessarily at the center of the body itself. This is the
        case for objects that consist of meshes, for example.

        This method needs to be implemented by the concrete simulation since Mujoco does not
        support computing bounding boxes itself.
        """
        return self._get_object_obs(self._get_bounding_box, pad=False)

    def get_target_bounding_boxes(self) -> np.ndarray:
        """
        Same as get_object_bounding_boxes(), but returns the bounding boxes for target objects.
        """
        return self._get_target_obs(self._get_bounding_box, pad=False)

    def get_object_bounding_boxes_in_table_coordinates(self) -> np.ndarray:
        """
        The default bounding box has position readings in its own coordinate system.
        The method shifts the positions into the table environment.
        """
        bboxes = self.get_object_bounding_boxes().copy()  # shape: (num_objects, 2, 3)
        bboxes[:, 0, :] += self.get_object_pos(pad=False).copy()
        return bboxes

    def get_target_bounding_boxes_in_table_coordinates(self) -> np.ndarray:
        """
        Same as get_shifted_object_bounding_boxes(), but works for target objects.
        """
        bboxes = self.get_target_bounding_boxes().copy()  # shape: (num_objects, 2, 3)
        bboxes[:, 0, :] += self.get_target_pos(pad=False).copy()
        return bboxes

    @classmethod
    def get_table_setting(
        cls, table_size: np.ndarray, num_objects: int, used_table_portion: float
    ):
        # place blocks and the targets randomly on the table
        # For curriculum, we use small centric region of the table to place objects. This
        # centric region is set by "used_table_portion".
        table_size_x, table_size_y = table_size[:2] * 2
        minimum_table_portion = num_objects * 0.1
        used_table_portion = np.clip(used_table_portion, minimum_table_portion, 1.0)
        return table_size_x, table_size_y, used_table_portion

    def get_placement_area(self) -> PlacementArea:
        table_pos, table_size, table_height = self.get_table_dimensions()

        table_size_x, table_size_y, used_table_portion = self.get_table_setting(
            table_size, self.num_objects, self.used_table_portion
        )
        assert used_table_portion <= 1.0

        place_size_x = 0.5 * table_size_x * used_table_portion
        place_size_y = 0.38 * table_size_y * used_table_portion
        place_size_z = 0.26
        offset_x = 0.5 * table_size_x - place_size_x / 2.0
        offset_y = 0.44 * table_size_y - place_size_y / 2.0
        offset_z = 2 * table_size[2]

        return PlacementArea(
            offset=(offset_x, offset_y, offset_z),
            size=(place_size_x, place_size_y, place_size_z),
        )

    ##############################################################
    # Fully internal methods. Maybe overloaded by sub classes.

    def _get_object_obs_with_prefix(self, obs_func, *, prefix, pad) -> np.ndarray:
        obs = np.asarray(
            [obs_func(f"{prefix}object{i}") for i in range(self.num_objects)]
        )
        if pad:
            assert len(obs.shape) == 2, f"Unexpected obs shape {obs.shape}"
            padded_obs = np.zeros((self.max_num_objects, obs.shape[-1]))
            padded_obs[: obs.shape[0]] = obs
            obs = padded_obs
        return obs

    def _get_object_obs(self, obs_func, pad=True) -> np.ndarray:
        return self._get_object_obs_with_prefix(obs_func, prefix="", pad=pad)

    def _get_target_obs(self, obs_func, pad=True) -> np.ndarray:
        return self._get_object_obs_with_prefix(obs_func, prefix="target:", pad=pad)

    @contextmanager
    def override_object_state(self, obj_pos, obj_rot):
        """
        A context manager that temporarily overrides object state.
        """

        cached_obj_pos = self.get_object_pos(pad=False)
        cached_obj_quat = self.get_object_quat(pad=False)

        self.set_object_pos(obj_pos)
        self.set_object_rot(obj_rot)
        self.forward()

        yield

        self.set_object_pos(cached_obj_pos)
        self.set_object_quat(cached_obj_quat)
        self.forward()

    def get_state(self):
        """ Returns a copy of the simulator state. """
        data = self.mj_sim.data
        return ExtendedSimState(
            time=data.time,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            qacc=data.qacc.copy(),
            ctrl=data.ctrl.copy(),
            actuator_force=data.actuator_force.copy(),
            sensordata=data.sensordata.copy(),
            udd_state={},
        )


class ExtendedSimState(
    namedtuple(
        "MjSimState", "time qpos qvel qacc ctrl actuator_force sensordata udd_state"
    )
):
    pass
