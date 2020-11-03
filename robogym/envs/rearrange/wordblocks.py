import logging
from typing import List

import attr
import numpy as np

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.common.utils import update_object_body_quat
from robogym.envs.rearrange.goals.object_state_fixed import ObjectFixedStateGoal
from robogym.envs.rearrange.simulation.base import (
    ObjectGroupConfig,
    RearrangeSimParameters,
)
from robogym.envs.rearrange.simulation.wordblocks import WordBlocksSim
from robogym.robot_env import build_nested_attr
from robogym.utils.rotation import quat_from_angle_and_axis

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class WordBlocksEnvConstants(RearrangeEnvConstants):
    rainbow_mode: bool = False


@attr.s(auto_attribs=True)
class WordBlocksEnvParameters(RearrangeEnvParameters):
    simulation_params: RearrangeSimParameters = build_nested_attr(
        RearrangeSimParameters, default=dict(num_objects=6)
    )


class WordBlocksEnv(
    RearrangeEnv[WordBlocksEnvParameters, WordBlocksEnvConstants, WordBlocksSim]
):
    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups: int):
        if self.constants.rainbow_mode:
            colors = [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.647, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.502, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.294, 0.0, 0.51, 1.0],
            ]
        else:
            colors = [[0.702, 0.522, 0.212, 1.0]] * 6

        return colors

    def _reset(self):
        super()._reset()

        # rotate A & I block a bit.
        new_quat = quat_from_angle_and_axis(0.38, np.array([0, 0, 1.0]))
        update_object_body_quat(
            self.mujoco_simulation.mj_sim, "target:object4", new_quat
        )
        update_object_body_quat(
            self.mujoco_simulation.mj_sim, "target:object5", new_quat
        )

    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        return ObjectFixedStateGoal(
            mujoco_simulation,
            args=constants.goal_args,
            relative_placements=np.array(
                [
                    [0.5, 0.05],
                    [0.5, 0.2],
                    [0.5, 0.35],
                    [0.5, 0.65],
                    [0.5, 0.8],
                    [0.5, 0.95],
                ]
            ),
        )


make_env = WordBlocksEnv.build
