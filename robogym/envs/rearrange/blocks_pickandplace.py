from robogym.envs.rearrange.blocks import BlockRearrangeEnv
from robogym.envs.rearrange.common.base import RearrangeEnvConstants
from robogym.envs.rearrange.goals.pickandplace import PickAndPlaceGoal
from robogym.envs.rearrange.simulation.blocks import BlockRearrangeSim


class BlocksPickAndPlaceEnv(BlockRearrangeEnv):
    @classmethod
    def build_goal_generation(
        cls, constants: RearrangeEnvConstants, mujoco_simulation: BlockRearrangeSim
    ):
        return PickAndPlaceGoal(mujoco_simulation, constants.goal_args)


make_env = BlocksPickAndPlaceEnv.build
