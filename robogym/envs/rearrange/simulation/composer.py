import abc
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import attr
import numpy as np

from robogym.envs.rearrange.common.utils import (
    find_stls,
    get_mesh_bounding_box,
    make_composed_mesh_object,
    make_target,
)
from robogym.envs.rearrange.simulation.base import (
    ObjectGroupConfig,
    RearrangeSimParameters,
    RearrangeSimulationInterface,
)
from robogym.mujoco.mujoco_xml import ASSETS_DIR, MujocoXML
from robogym.randomization.env import build_randomizable_param

logger = logging.getLogger(__file__)


class Composer:
    def __init__(self, random_state: np.random.RandomState = None):
        if random_state is None:
            random_state = np.random.RandomState()
        self._random_state = random_state

    def reset(self):
        pass

    @abc.abstractmethod
    def sample(self, name: str, num_geoms: int, object_size: float) -> MujocoXML:
        """Samples a single, composed object with the given name, number of geoms, and total
        object size.

        :param name: The name of the object
        :param num_geoms: The number of geoms to be composed
        :param object_size: In half-size, as per Mujoco convenient
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_bounding_box(self, sim, object_name) -> Tuple[float, float]:
        """Computes the bounding box for a single, composed object as identified by its name.
        """
        raise NotImplementedError


class RandomMeshComposer(Composer):
    PRIMITIVES_CACHE: Dict[str, List[str]] = {}

    YCB_ASSET_PATH = os.path.join(ASSETS_DIR, "stls", "ycb")
    GEOM_ASSET_PATH = os.path.join(ASSETS_DIR, "stls", "geom")

    def __init__(self, random_state=None, mesh_path: Optional[Union[list, str]] = None):
        super().__init__(random_state=random_state)

        if mesh_path is None:
            # Default to both YCB meshes and geoms
            self._mesh_paths = [self.YCB_ASSET_PATH, self.GEOM_ASSET_PATH]
        elif isinstance(mesh_path, str):
            self._mesh_paths = [mesh_path]
        else:
            assert isinstance(mesh_path, list)
            self._mesh_paths = mesh_path

        # Find meshes (and cache for performance).
        for path in self._mesh_paths:
            if path not in self.PRIMITIVES_CACHE:
                self.PRIMITIVES_CACHE[path] = find_stls(path)

        self._meshes = {p: self.PRIMITIVES_CACHE[p] for p in self._mesh_paths}
        assert (
            sum(map(len, self._meshes.values())) > 0
        ), f"Did not find any .stl files in {self._mesh_paths}"

    def _sample_primitives(self, num_geoms: int) -> List[str]:
        chosen_primitives = []
        for _ in range(num_geoms):
            chosen_path = self._random_state.choice(self._mesh_paths)
            chosen_primitives.append(
                self._random_state.choice(self._meshes[chosen_path])
            )
        return chosen_primitives

    def sample(self, name: str, num_geoms: int, object_size: float) -> MujocoXML:
        chosen_primitives = self._sample_primitives(num_geoms)
        chosen_attachment = self._random_state.choice(["random", "last"])
        return make_composed_mesh_object(
            name,
            chosen_primitives,
            self._random_state,
            mesh_size_range=(0.01, 0.1),
            object_size=object_size,
            attachment=chosen_attachment,
        )

    def get_bounding_box(self, sim, object_name) -> Tuple[float, float]:
        pos, size = get_mesh_bounding_box(sim, object_name)
        return pos, size


def composer_converter(s: str) -> Composer:
    """ Converts string -> Composer for a predefined set of composers.
    """
    if s == "geom":
        return RandomMeshComposer(mesh_path=RandomMeshComposer.GEOM_ASSET_PATH)
    elif s == "mesh":
        return RandomMeshComposer(mesh_path=RandomMeshComposer.YCB_ASSET_PATH)
    elif s == "mixed":
        return RandomMeshComposer()
    else:
        raise ValueError()


@attr.s(auto_attribs=True)
class ComposerObjectGroupConfig(ObjectGroupConfig):
    num_geoms: int = 1


@attr.s(auto_attribs=True)
class ComposerRearrangeSimParameters(RearrangeSimParameters):
    # Overwrite the object group config type.
    object_groups: List[ComposerObjectGroupConfig] = None  # type: ignore

    def __attrs_post_init__(self):
        if self.object_groups is None:
            self.object_groups = [
                ComposerObjectGroupConfig(count=1, object_ids=[obj_id])
                for obj_id in range(self.num_objects)
            ]

    # Num. of geoms to use to compose one object.
    num_max_geoms: int = build_randomizable_param(1, low=1, high=20)

    # The type of object composer to be used.
    composer: Composer = attr.ib(  # type: ignore
        default="mixed", converter=composer_converter,  # type: ignore
    )


class ComposerRearrangeSim(
    RearrangeSimulationInterface[ComposerRearrangeSimParameters]
):
    """
    Move around a randomly generated object of different colors on the table.
    Each object is composed by multiple connected geoms.
    """

    @property
    def object_size(self):
        return self.simulation_params.object_size

    @property
    def num_max_geoms(self):
        return self.simulation_params.num_max_geoms

    @classmethod
    def make_objects_xml(cls, xml, simulation_params: ComposerRearrangeSimParameters):
        composer = simulation_params.composer
        composer.reset()

        xmls = []
        for obj_group in simulation_params.object_groups:
            first_obj_id = obj_group.object_ids[0]
            object_xml = composer.sample(
                f"object{first_obj_id}",
                obj_group.num_geoms,
                simulation_params.object_size,
            )
            xmls.append((object_xml, make_target(object_xml)))

            for j in obj_group.object_ids[1:]:
                copied_object_xml = deepcopy(object_xml)
                copied_object_xml.replace_name(f"object{first_obj_id}", f"object{j}")
                xmls.append((copied_object_xml, make_target(copied_object_xml)))

        assert len(xmls) == simulation_params.num_objects
        return xmls

    def _get_bounding_box(self, object_name):
        composer = self.simulation_params.composer
        return composer.get_bounding_box(self.mj_sim, object_name)
