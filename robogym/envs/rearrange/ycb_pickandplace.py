from robogym.envs.rearrange.common.base import RearrangeEnvConstants
from robogym.envs.rearrange.goals.pickandplace import PickAndPlaceGoal
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.ycb import YcbRearrangeEnv


class YcbPickAndPlaceEnv(YcbRearrangeEnv):
    @classmethod
    def build_goal_generation(
        cls, constants: RearrangeEnvConstants, mujoco_simulation: MeshRearrangeSim
    ):
        return PickAndPlaceGoal(mujoco_simulation, constants.goal_args)


make_env = YcbPickAndPlaceEnv.build
