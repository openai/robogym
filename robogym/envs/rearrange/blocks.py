import attr

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)
from robogym.envs.rearrange.simulation.blocks import (
    BlockRearrangeSim,
    BlockRearrangeSimParameters,
)
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class BlockRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: BlockRearrangeSimParameters = build_nested_attr(
        BlockRearrangeSimParameters
    )


class BlockRearrangeEnv(
    RearrangeEnv[BlockRearrangeEnvParameters, RearrangeEnvConstants, BlockRearrangeSim]
):
    OBJECT_COLORS = (
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 1, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 1, 1, 1),
    )

    def _sample_object_colors(self, num_groups: int):
        return self._random_state.permutation(self.OBJECT_COLORS)[:num_groups]


make_env = BlockRearrangeEnv.build
