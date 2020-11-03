import logging

import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.goals.object_stack_goal import ObjectStackGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimParameters
from robogym.envs.rearrange.simulation.blocks import (
    BlockRearrangeSim,
    BlockRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class BlockStackEnvConstants(RearrangeEnvConstants):
    # whether block stacked in the fixed order or random order.
    fixed_order: bool = False


@attr.s(auto_attribs=True)
class BlockStackEnvParameters(RearrangeEnvParameters):
    simulation_params: RearrangeSimParameters = build_nested_attr(
        BlockRearrangeSimParameters, default=dict(num_objects=2)
    )


class BlockStackEnv(
    RearrangeEnv[BlockStackEnvParameters, BlockStackEnvConstants, BlockRearrangeSim]
):
    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        return ObjectStackGoal(
            mujoco_simulation, constants.goal_args, constants.fixed_order
        )


make_env = BlockStackEnv.build
