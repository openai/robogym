import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.goals.dominos import DominoStateGoal
from robogym.envs.rearrange.goals.object_state import GoalArgs
from robogym.envs.rearrange.goals.train_state import TrainStateGoal
from robogym.envs.rearrange.simulation.dominos import (
    DominosRearrangeSim,
    DominosRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class DominosRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: DominosRearrangeSimParameters = build_nested_attr(
        DominosRearrangeSimParameters,
    )


@attr.s(auto_attribs=True)
class DominosRearrangeEnvConstants(RearrangeEnvConstants):
    # If set then we will setup the goal as a `DominoStateGoal` where we try to place
    # dominos in an arc. Otherise, use the same goal logic as `TrainStateGoal`.
    is_holdout: bool = False

    goal_args: GoalArgs = build_nested_attr(
        GoalArgs, default=dict(rot_dist_type="mod180")
    )


class DominosRearrangeEnv(
    RearrangeEnv[
        DominosRearrangeEnvParameters,
        DominosRearrangeEnvConstants,
        DominosRearrangeSim,
    ]
):
    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        if constants.is_holdout:
            return DominoStateGoal(mujoco_simulation, args=constants.goal_args)
        else:
            return TrainStateGoal(mujoco_simulation, args=constants.goal_args)


make_env = DominosRearrangeEnv.build
