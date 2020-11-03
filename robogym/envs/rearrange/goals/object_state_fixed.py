from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_targets_with_fixed_position
from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface


class ObjectFixedStateGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs],
        relative_placements: np.ndarray,
        init_quats: Optional[np.ndarray] = None,
    ):
        """
        Always return fixed goal state for each object.

        :param relative_placements: the relative position of each object, relative to the placement area.
            Each dimension is a ratio between [0, 1].
        :param init_quats: the desired quat of each object.
        """
        super().__init__(mujoco_simulation, args)

        if init_quats is None:
            init_quats = np.array([[1, 0, 0, 0]] * self.mujoco_simulation.num_objects)

        assert relative_placements.shape == (self.mujoco_simulation.num_objects, 2)
        assert init_quats.shape == (self.mujoco_simulation.num_objects, 4)

        self.relative_placements = relative_placements
        self.init_quats = init_quats

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:

        # Set all the goals to the initial rotation first.
        quats = np.array(self.init_quats)
        self.mujoco_simulation.set_target_quat(quats)
        self.mujoco_simulation.forward()

        # Then place the objects as designed.
        return place_targets_with_fixed_position(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            self.relative_placements,
        )
