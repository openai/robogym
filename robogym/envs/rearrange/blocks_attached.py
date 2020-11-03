from typing import TypeVar

import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.goals.attached_block_state import AttachedBlockStateGoal
from robogym.envs.rearrange.simulation.blocks import (
    BlockRearrangeSim,
    BlockRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class AttachedBlockRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: BlockRearrangeSimParameters = build_nested_attr(
        BlockRearrangeSimParameters, default=dict(num_objects=8)
    )


PType = TypeVar("PType", bound=AttachedBlockRearrangeEnvParameters)
CType = TypeVar("CType", bound=RearrangeEnvConstants)
SType = TypeVar("SType", bound=BlockRearrangeSim)


class AttachedBlockRearrangeEnv(RearrangeEnv[PType, CType, SType]):
    @classmethod
    def build_goal_generation(cls, constants: CType, mujoco_simulation: SType):
        return AttachedBlockStateGoal(mujoco_simulation, args=constants.goal_args)


make_env = AttachedBlockRearrangeEnv.build
