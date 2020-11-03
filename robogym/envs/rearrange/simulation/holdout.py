from typing import Any, Dict, List, Type, TypeVar

import attr
import numpy as np

from robogym.envs.rearrange.common.utils import get_mesh_bounding_box
from robogym.envs.rearrange.simulation.base import (
    RearrangeSimParameters,
    RearrangeSimulationInterface,
)
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.randomization.env import build_randomizable_param


@attr.s(auto_attribs=True)
class ObjectConfig:
    # xml path relative to robogym/assets/xmls folder
    xml_path: str = attr.ib()

    # Map between tag name to property values.
    # for example
    # {
    #     "geom": {
    #         "size": 0.01,
    #         "rgba": [0.5, 0.5, 0.5, 1.0],
    #     }
    # }
    tag_args: Dict[str, Dict[str, Any]] = {}

    # Same format as tag_args. Can be used to
    # specify material related physical properties.
    material_args: Dict[str, Dict[str, Any]] = {}


OType = TypeVar("OType", bound=ObjectConfig)


def convert_object_configs(object_config_type: Type[OType], data: Any) -> List[OType]:
    object_configs = []
    for config in data:
        if isinstance(config, object_config_type):
            object_configs.append(config)
        else:
            assert isinstance(config, dict)
            object_configs.append(object_config_type(**config))

    return object_configs


@attr.s(auto_attribs=True)
class SceneObjectConfig(ObjectConfig):
    """
    Configs for static object in the scene. These observes will not be part of
    state observation and goal and will by default have collision disabled.
    """

    # Position (x, y, z) of the object on the table which is defined as:
    # x: X position in range of [0, 1] relative to top left corner (-x, -y) of placement area.
    # y: Y position in range of [0, 1] relative to top left corner (-x, -y) of placement area .
    # z: Z position in meters relative to table surface.
    pos: List[float] = [0.0, 0.0, 0.0]

    # Rotation of the object.
    quat: List[float] = [1.0, 0.0, 0.0, 0.0]


def convert_scene_object_config(data: Any) -> List[SceneObjectConfig]:
    return convert_object_configs(SceneObjectConfig, data)


@attr.s(auto_attribs=True)
class TaskObjectConfig(ObjectConfig):
    """
    Configs for objects which are part of the task.
    """

    # Total number of this object in the env.
    count: int = 1


def convert_task_object_config(data: Any) -> List[TaskObjectConfig]:
    return convert_object_configs(TaskObjectConfig, data)


@attr.s(auto_attribs=True)
class HoldoutRearrangeSimParameters(RearrangeSimParameters):
    scene_object_configs: List[SceneObjectConfig] = attr.ib(
        default=[], converter=convert_scene_object_config
    )

    task_object_configs: List[TaskObjectConfig] = attr.ib(
        default=[TaskObjectConfig(xml_path="object/tests/block.xml")],
        converter=convert_task_object_config,
    )

    shared_settings: str = attr.ib(default="", converter=str)

    num_objects: int = build_randomizable_param(low=1, high=32)

    @num_objects.default
    def default_num_objects(self):
        return self._num_objects()

    @num_objects.validator
    def validate_num_objects(self, _, value):
        if value > self._num_objects():
            raise ValueError(
                f"Too many objects {value}, only {self._num_objects()} "
                f"objects are defined in config"
            )

    def _num_objects(self):
        return sum(o.count for o in self.task_object_configs)


class HoldoutRearrangeSim(RearrangeSimulationInterface[HoldoutRearrangeSimParameters]):
    """
    Simulation class for holdout envs.
    """

    @classmethod
    def _make_object_xml(cls, object_id: int, config: ObjectConfig):
        object_name = f"object{object_id}"
        xml = (
            MujocoXML.parse(config.xml_path)
            .set_objects_attr(tag="joint", name="joint")
            .add_name_prefix(f"{object_name}:", exclude_attribs=["class"])
            .set_objects_attr(tag="body", name=object_name)
        )

        xml.set_objects_attrs(config.tag_args)
        xml.set_objects_attrs(config.material_args)

        return xml

    @classmethod
    def _make_scene_object_xml(
        cls, object_id: int, config: SceneObjectConfig, table_dims
    ):
        xml = cls._make_object_xml(object_id, config)
        table_pos, table_size, table_height = table_dims
        pos = np.array(config.pos)
        x, y, _ = table_pos + table_size * (pos * 2 - 1)
        z = table_height + pos[-1]

        xml = (
            xml.add_name_prefix("scene:", exclude_attribs=["class"])
            .set_objects_attr(tag="body", pos=[x, y, z])
            .set_objects_attr(tag="body", quat=config.quat)
            .set_objects_attr(tag="geom", contype=0, conaffinity=0)
        )

        return xml

    @classmethod
    def make_objects_xml(cls, xml, simulation_params):
        xmls = []
        num_objects = simulation_params.num_objects
        object_id = 0
        for object_config in simulation_params.task_object_configs[:num_objects]:
            for _ in range(object_config.count):
                obj_xml = cls._make_object_xml(object_id, object_config)
                target_xml = (
                    cls._make_object_xml(object_id, object_config)
                    .remove_objects_by_tag("joint")
                    .remove_objects_by_tag("mesh")  # Target can reuse same mesh.
                    .remove_objects_by_tag(
                        "material"
                    )  # Target can reuse same material.
                    .add_name_prefix(
                        "target:", exclude_attribs=["mesh", "material", "class"]
                    )
                    .set_objects_attr(tag="geom", contype=0, conaffinity=0)
                )

                xmls.append((obj_xml, target_xml))
                object_id += 1

        return xmls

    @classmethod
    def make_world_xml(
        cls, *, mujoco_timestep: float, simulation_params=None, **kwargs
    ):
        if simulation_params is None:
            simulation_params = HoldoutRearrangeSimParameters()
        world_xml = super().make_world_xml(
            simulation_params=simulation_params, mujoco_timestep=mujoco_timestep
        )
        table_dims = cls.get_table_dimensions_from_xml(world_xml)

        for object_id, object_config in enumerate(
            simulation_params.scene_object_configs
        ):
            world_xml.append(
                cls._make_scene_object_xml(object_id, object_config, table_dims,)
            )
        if simulation_params.shared_settings:
            world_xml.append(MujocoXML.parse(simulation_params.shared_settings))
        return world_xml

    def _get_bounding_box(self, object_name):
        return get_mesh_bounding_box(self.mj_sim, object_name)
