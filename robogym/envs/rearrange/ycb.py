import logging
import os
from typing import Dict, List

import attr

from robogym.envs.rearrange.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
)
from robogym.envs.rearrange.common.utils import find_meshes_by_dirname
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim

logger = logging.getLogger(__name__)


def find_ycb_meshes() -> Dict[str, list]:
    return find_meshes_by_dirname("ycb")


def extract_object_name(mesh_files: List[str]) -> str:
    """
    Given a list of mesh file paths, this method returns an consistent name for the object

    :param mesh_files: List of paths to mesh files on disk for the object
    :return: Consistent name for the object based on the mesh file paths
    """
    dir_names = sorted(set([os.path.basename(os.path.dirname(p)) for p in mesh_files]))
    if len(dir_names) != 1:
        logger.warning(
            f"Multiple directory names found: {dir_names} for object: {mesh_files}."
        )

    return dir_names[0]


@attr.s(auto_attribs=True)
class YcbRearrangeEnvConstants(MeshRearrangeEnvConstants):
    # Whether to sample meshes with replacement
    sample_with_replacement: bool = True


class YcbRearrangeEnv(
    MeshRearrangeEnv[
        MeshRearrangeEnvParameters, YcbRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    MESH_FILES = find_ycb_meshes()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cached_object_names: Dict[str, str] = {}

    def _recreate_sim(self) -> None:
        # Call super to recompute `self.parameters.simulation_params.mesh_files`.
        super()._recreate_sim()

        # Recompute object names from new mesh files
        self._cached_object_names = {}
        for obj_group in self.mujoco_simulation.object_groups:
            mesh_obj_name = extract_object_name(obj_group.mesh_files)
            for i in obj_group.object_ids:
                self._cached_object_names[f"object{i}"] = mesh_obj_name

    def _sample_object_meshes(self, num_groups: int) -> List[List[str]]:
        if self.parameters.mesh_names is not None:
            candidates = [
                files
                for dir_name, files in self.MESH_FILES.items()
                if dir_name in self.parameters.mesh_names
            ]
        else:
            candidates = list(self.MESH_FILES.values())

        assert len(candidates) > 0, f"No mesh file for {self.parameters.mesh_names}."
        candidates = sorted(candidates)
        replace = self.constants.sample_with_replacement
        indices = self._random_state.choice(
            len(candidates), size=num_groups, replace=replace
        )

        return [candidates[i] for i in indices]

    def _get_simulation_info(self) -> dict:
        simulation_info = super()._get_simulation_info()
        simulation_info.update(self._cached_object_names)

        return simulation_info


make_env = YcbRearrangeEnv.build
