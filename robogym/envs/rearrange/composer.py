import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.goals.object_state import GoalArgs
from robogym.envs.rearrange.simulation.composer import (
    ComposerRearrangeSim,
    ComposerRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class ComposerRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: ComposerRearrangeSimParameters = build_nested_attr(
        ComposerRearrangeSimParameters
    )


@attr.s(auto_attribs=True)
class ComposerRearrangeEnvConstants(RearrangeEnvConstants):
    goal_args: GoalArgs = build_nested_attr(GoalArgs, default={"stabilize_goal": True})

    goal_generation: str = "train"


class ComposerRearrangeEnv(
    RearrangeEnv[
        ComposerRearrangeEnvParameters,
        ComposerRearrangeEnvConstants,
        ComposerRearrangeSim,
    ]
):
    def _sample_group_attributes(self, num_groups: int):
        attrs_dict = super()._sample_group_attributes(num_groups)
        attrs_dict["num_geoms"] = self._random_state.randint(
            low=1,
            high=self.parameters.simulation_params.num_max_geoms + 1,
            size=num_groups,
        )
        return attrs_dict


make_env = ComposerRearrangeEnv.build
