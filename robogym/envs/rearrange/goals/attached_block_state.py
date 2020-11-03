from typing import Tuple

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_targets_with_fixed_position
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal


class AttachedBlockStateGoal(ObjectStateGoal):
    """
    This is a goal state generator for the advanced blocks environment, where the
    goal configuration of the blocks are highly structured and blocks are tightly attached to each
    other.
    """

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        # Set all the goals to the initial rotation first.
        self.mujoco_simulation.set_target_quat(np.array([[1, 0, 0, 0]] * 8))
        self.mujoco_simulation.forward()

        # Set position of blocks
        block_size = self.mujoco_simulation.simulation_params.object_size
        width, height, _ = self.mujoco_simulation.get_placement_area().size

        # Note that block_size and rel_w, rel_h are all half of the block size
        rel_w, rel_h = block_size / width, block_size / height

        # offset for making blocks to be attached to each other
        offset_w, offset_h = rel_w * 2, rel_h * 2

        # Expected configuration
        #       [ ][ ]
        #    [ ][ ][ ][ ]
        #       [ ][ ]
        block_config = random_state.permutation(
            [
                [offset_w, 0],
                [offset_w * 2, 0],
                [0, offset_h],
                [offset_w, offset_h],
                [offset_w * 2, offset_h],
                [offset_w * 3, offset_h],
                [offset_w, offset_h * 2],
                [offset_w * 2, offset_h * 2],
            ]
        )

        # Now randomly place the overall config in the placement area
        config_w, config_h = block_config.max(axis=0)
        margin_w, margin_h = 1.0 - config_w - rel_w, 1.0 - config_h - rel_h

        ori_x, ori_y = random_state.uniform(
            low=(rel_w, rel_h), high=(margin_w, margin_h)
        )

        # Randomize the position of the entire block configuration.
        block_config += np.array([[ori_x, ori_y]])

        # Then place the objects as designed.
        return place_targets_with_fixed_position(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            block_config,
        )
