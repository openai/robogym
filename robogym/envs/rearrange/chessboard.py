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
from robogym.envs.rearrange.common.utils import find_meshes_by_dirname
from robogym.envs.rearrange.goals.object_state_fixed import ObjectFixedStateGoal
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.chessboard import ChessboardRearrangeSim
from robogym.robot_env import build_nested_attr

logger = logging.getLogger(__name__)

CHESS_CHARS = ["rook", "queen", "bishop", "knight"]


@attr.s(auto_attribs=True)
class ChessboardRearrangeEnvParameters(MeshRearrangeEnvParameters):
    simulation_params: MeshRearrangeSimParameters = build_nested_attr(
        MeshRearrangeSimParameters,
        default=dict(num_objects=len(CHESS_CHARS), mesh_scale=0.4),
    )


class ChessboardRearrangeEnv(
    MeshRearrangeEnv[
        ChessboardRearrangeEnvParameters,
        MeshRearrangeEnvConstants,
        ChessboardRearrangeSim,
    ]
):
    MESH_FILES = find_meshes_by_dirname("chess")

    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups: int):
        assert num_groups == 4
        return [[0.267, 0.165, 0.133, 1.0]] * num_groups

    def _sample_object_meshes(self, num_groups: int):
        assert num_groups == 4
        return [self.MESH_FILES[name] for name in CHESS_CHARS[:num_groups]]

    def _reset(self):
        super()._reset()

        # Scale chessboard properly
        (x, y, _), (w, h, _), z = self.mujoco_simulation.get_table_dimensions()
        placement_area = self.mujoco_simulation.get_placement_area()
        px = placement_area.offset[0]
        py = placement_area.offset[1]
        pw = placement_area.size[0]
        ph = placement_area.size[1]

        # Move board to center of placement area.
        sim = self.mujoco_simulation.mj_sim
        body_id = sim.model.body_name2id("chessboard")
        sim.model.body_pos[body_id][:] = [x - w + px + pw / 2, y - h + py + ph / 2, z]

        self.mujoco_simulation.forward()

    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        pw, ph, _ = mujoco_simulation.get_placement_area().size
        sim = mujoco_simulation.mj_sim
        geom_id = sim.model.geom_name2id("chessboard")
        cw, ch = sim.model.geom_size[geom_id, :2]
        num_objects = mujoco_simulation.num_objects
        placements = np.zeros((num_objects, 2))
        placements[:, 0] = 1 - 1.0 / num_objects / 2
        placements[:, 1] = (
            np.linspace(
                ph / 2 - ch + ch / num_objects,
                ph / 2 + ch - ch / num_objects,
                num_objects,
            )
            / ph
        )

        return ObjectFixedStateGoal(
            mujoco_simulation, args=constants.goal_args, relative_placements=placements
        )


make_env = ChessboardRearrangeEnv.build
