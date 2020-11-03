import logging
from typing import List

import attr
import numpy as np

from robogym.envs.rearrange.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
    MeshRearrangeSimParameters,
)
from robogym.envs.rearrange.goals.object_state_fixed import ObjectFixedStateGoal
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.ycb import find_ycb_meshes
from robogym.robot_env import build_nested_attr
from robogym.utils.rotation import quat_from_angle_and_axis

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class TableSettingRearrangeEnvParameters(MeshRearrangeEnvParameters):
    simulation_params: MeshRearrangeSimParameters = build_nested_attr(
        MeshRearrangeSimParameters, default=dict(num_objects=5)
    )


class TableSettingRearrangeEnv(
    MeshRearrangeEnv[
        TableSettingRearrangeEnvParameters, MeshRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    MESH_FILES = find_ycb_meshes()

    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups: int):
        assert num_groups == 5
        return [[0.99, 0.44, 0.35, 1.0]] + [[0.506, 0.675, 0.75, 1.0]] * 4

    def _sample_object_size_scales(self, num_groups: int):
        assert num_groups == 5
        return [0.6, 0.53, 0.63, 0.6, 0.6]

    def _sample_object_meshes(self, num_groups: int):
        """Add one plate, 2 forks, 1 spoon and 1 knife."""
        return [
            self.MESH_FILES[name]
            for name in ["029_plate", "030_fork", "030_fork", "032_knife", "031_spoon"]
        ]

    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        return ObjectFixedStateGoal(
            mujoco_simulation,
            args=constants.goal_args,
            relative_placements=np.array(
                [
                    [0.6, 0.5],  # "029_plate"
                    [0.6, 0.68],  # "030_fork"
                    [0.6, 0.75],  # "030_fork"
                    [0.6, 0.36],  # "032_knife"
                    [0.6, 0.28],  # "031_spoon"
                ]
            ),
            init_quats=np.array(
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    # We need to rotate the spoon a little bit counter-clock-wise to be aligned with others.
                    quat_from_angle_and_axis(0.38, np.array([0, 0, 1.0])),
                ]
            ),
        )


make_env = TableSettingRearrangeEnv.build
