from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface


class PickAndPlaceGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs] = GoalArgs(),
        height_range: Tuple[float, float] = (0.05, 0.25),
    ):

        super().__init__(mujoco_simulation, args)
        self.height_range = height_range

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        placement, is_valid = super()._sample_next_goal_positions(random_state)
        placement = move_one_object_to_the_air(
            current_placement=placement,
            height_range=self.height_range,
            random_state=random_state,
        )
        return placement, is_valid


def move_one_object_to_the_air(
    current_placement: np.ndarray,
    height_range: Tuple[float, float],
    random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Modify current_placement to move one object to the air.

    :param current_placement: np.ndarray of size (num_objects, 3) where columns are x, y, z
        coordinates of objects relative to the world frame.
    :param height_range: One object is moved along z direction to have height from table in a
        range (min_height, max_height). Height is randomly sampled.
    :param random_state: numpy RandomState to use for sampling
    :return: modified object placement. np.ndarray of (num_objects, 3) where columns are x, y, z
        coordinates of objects relative to the world frame.
    """
    n_objects = current_placement.shape[0]
    min_h, max_h = height_range

    height = random_state.uniform(low=min_h, high=max_h)
    target_i = random_state.randint(n_objects)

    current_placement[target_i, -1] += height
    return current_placement
