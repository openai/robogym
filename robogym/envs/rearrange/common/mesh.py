import abc
from typing import List, Optional, TypeVar

import attr
import numpy as np

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.common.utils import get_combined_mesh
from robogym.envs.rearrange.simulation.mesh import (
    MeshRearrangeSim,
    MeshRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class MeshRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: MeshRearrangeSimParameters = build_nested_attr(
        MeshRearrangeSimParameters
    )

    # A list of mesh folders to sample from in _sample_object_meshes(). If None, we
    # just sample from all the available meshes.
    mesh_names: Optional[List[str]] = None


@attr.s(auto_attribs=True)
class MeshRearrangeEnvConstants(RearrangeEnvConstants):
    use_grey_colors: bool = False

    # If true, the dimension with max size will be normalized to mesh_size parameter
    # defined below. This overwrite the global scaling factor `mesh_scale`.
    normalize_mesh: bool = False

    # Half size for the dimension with max size after normalization.
    normalized_mesh_size: float = 0.05


CType = TypeVar("CType", bound=MeshRearrangeEnvConstants)
PType = TypeVar("PType", bound=MeshRearrangeEnvParameters)
SType = TypeVar("SType", bound=MeshRearrangeSim)


class MeshRearrangeEnv(RearrangeEnv[PType, CType, SType], abc.ABC):
    def _sample_group_attributes(self, num_groups: int):
        attrs_dict = super()._sample_group_attributes(num_groups)
        attrs_dict["mesh_files"] = self._sample_object_meshes(num_groups)
        return attrs_dict

    def _sample_object_colors(self, num_groups: int):
        # Overwrite colors if needed.
        if self.constants.use_grey_colors:
            return [[0.5, 0.5, 0.5, 1]] * num_groups
        return super()._sample_object_colors(num_groups)

    @abc.abstractmethod
    def _sample_object_meshes(self, num_groups: int) -> List[List[str]]:
        """Sample `num_groups` sets of mesh_files and return.
        """
        pass

    def _recreate_sim(self):
        num_objects = self.mujoco_simulation.num_objects
        # If we have 10 or more objects, downscale all objects so that the total object area is
        # roughly equal to the area occupied by 10 "normal" objects.
        global_scale = 1.0 if num_objects < 10 else (10.0 / num_objects) ** 0.5

        if self.constants.normalize_mesh:
            # Use trimesh to measure the size of each object group's mesh so that we can compute
            # the appropriate normalizing rescaling factor.
            new_obj_group_scales = []
            for obj_group in self.parameters.simulation_params.object_groups:
                mesh = get_combined_mesh(obj_group.mesh_files)
                size = mesh.extents
                # The size is the "full size", so divide by 2 to get the half-size since this is
                # what MuJoCo uses and the normalized_mesh_size param expects.
                max_size = np.max(size) / 2.0

                new_obj_group_scales.append(
                    self.constants.normalized_mesh_size / max_size
                )
        else:
            new_obj_group_scales = np.ones(
                len(self.parameters.simulation_params.object_groups)
            )

        # For meshes, we cannot resize 'mesh' geoms directly. We need to change the scale
        # when loading the mesh assets into XML.
        for i, obj_group in enumerate(self.parameters.simulation_params.object_groups):
            orig_group_scale = obj_group.scale
            if num_objects >= 10:
                # If we have more than 10 objects, then we don't allow upscaling the object since
                # this could result in too little space to place objects.
                orig_group_scale = min(orig_group_scale, 1.0)

            obj_group.scale = orig_group_scale * new_obj_group_scales[i] * global_scale

        return super()._recreate_sim()

    def _apply_object_size_scales(self):
        # We need to apply rescaling for mesh objects when creating the XML, so we do
        # nothing here after sim has been created.
        pass
