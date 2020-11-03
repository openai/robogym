import attr
import numpy as np

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
class BlockTrainRearrangeEnvConstants(RearrangeEnvConstants):
    # If true, we randomize width, height, depth of each block object independently.
    # If false, we use regular-shaped cube blocks with weight == height == depth.
    use_cuboid: bool = False

    goal_generation: str = "train"


@attr.s(auto_attribs=True)
class BlockTrainRearrangeEnvParameters(RearrangeEnvParameters):
    simulation_params: BlockRearrangeSimParameters = build_nested_attr(
        BlockRearrangeSimParameters
    )


class BlockTrainRearrangeEnv(
    RearrangeEnv[
        BlockTrainRearrangeEnvParameters,
        BlockTrainRearrangeEnvConstants,
        BlockRearrangeSim,
    ]
):
    def _apply_object_size_scales(self):
        if not self.constants.use_cuboid:
            super()._apply_object_size_scales()
            return

        # Randomize width, length and height of each block.
        object_scales_by_group = np.exp(
            self._random_state.uniform(
                low=-self.parameters.object_scale_low,
                high=self.parameters.object_scale_high,
                size=(self.mujoco_simulation.num_groups, 3),
            )
        )
        object_scales = np.array(
            [
                object_scales_by_group[i].copy()
                for i, obj_group in enumerate(self.mujoco_simulation.object_groups)
                for _ in range(obj_group.count)
            ]
        )
        assert object_scales.shape == (self.mujoco_simulation.num_objects, 3)
        self.mujoco_simulation.rescale_object_sizes(object_scales)


make_env = BlockTrainRearrangeEnv.build
