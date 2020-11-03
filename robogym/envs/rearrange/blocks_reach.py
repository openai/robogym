import attr
import numpy as np

from robogym.envs.rearrange.blocks import BlockRearrangeEnvParameters, BlockRearrangeSim
from robogym.envs.rearrange.common.base import RearrangeEnv, RearrangeEnvConstants
from robogym.envs.rearrange.goals.object_reach_goal import (
    DeterministicReachGoal,
    ObjectReachGoal,
)


@attr.s(auto_attribs=True)
class BlocksReachEnvConstants(RearrangeEnvConstants):
    # Goal generation for env.
    # det-state: Use deterministic goals
    # state: Use random state goals
    goal_generation: str = attr.ib(
        default="state", validator=attr.validators.in_(["state", "det-state"])
    )


class BlocksReachEnv(
    RearrangeEnv[
        BlockRearrangeEnvParameters, BlocksReachEnvConstants, BlockRearrangeSim,
    ]
):
    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        if constants.goal_generation == "det-state":
            return DeterministicReachGoal(mujoco_simulation, args=constants.goal_args)
        else:
            return ObjectReachGoal(mujoco_simulation, args=constants.goal_args)

    def _calculate_goal_distance_reward(self, previous_goal_distance, goal_distance):
        return np.sum(previous_goal_distance["obj_pos"] - goal_distance["obj_pos"])


make_env = BlocksReachEnv.build
