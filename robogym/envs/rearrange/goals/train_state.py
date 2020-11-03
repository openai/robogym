import logging
from typing import Tuple

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_targets_with_goal_distance_ratio
from robogym.envs.rearrange.goals.object_state import ObjectStateGoal

logger = logging.getLogger(__name__)


def move_one_object_to_the_air_with_restrictions(
    current_placement: np.ndarray,
    height_range: Tuple[float, float],
    object_size: float,
    random_state: np.random.RandomState,
    pickup_proba: float = 0.0,
    stacking_proba: float = 0.0,
    goal_distance_ratio: float = 1.0,
) -> np.ndarray:
    """
    Modify current_placement to move one object to the air.
    Set height w.r.t goal_distance_ratio too.
    With probability `pickup_proba`, we will randomly move one object up into the air.
    Otherwise do nothing.
    The pickup height is discounted by `goal_distance_ratio`.
    """
    assert 0.0 <= pickup_proba + stacking_proba <= 1.0
    if pickup_proba + stacking_proba == 0:
        return current_placement

    logger.info(
        f"Move one object to the air with restrictions "
        f"(goal_distance_ratio={goal_distance_ratio} "
        f"pickup_proba={pickup_proba} stacking_proba={stacking_proba})"
    )

    p = random_state.random()

    if p > pickup_proba + stacking_proba:
        # rearrange task: do nothing.
        return current_placement

    elif p < pickup_proba:
        # pick up task
        min_h, max_h = height_range
        height = random_state.uniform(low=min_h, high=max_h)

        n_objects = current_placement.shape[0]
        target_i = random_state.randint(n_objects)

        # The height of a pick-and-place goal is set w.r.t the goal distance too.
        current_placement[target_i, -1] += height * goal_distance_ratio
        return current_placement

    else:
        # stacking task
        n_objects = current_placement.shape[0]

        if n_objects >= 2:
            # Randomly pick block indices for building a tower of size [2, n_objects].
            tower_size = random_state.randint(2, n_objects + 1)
            obj_indices = np.random.choice(
                list(range(n_objects)), size=tower_size, replace=False
            )
            assert len(obj_indices) > 1

            logger.info(
                f"Building a tower of size {tower_size} with object indices {obj_indices} ..."
            )
            target_i = obj_indices[0]
            for height, i in enumerate(obj_indices[1:]):
                current_placement[i, 0] = current_placement[target_i, 0]
                current_placement[i, 1] = current_placement[target_i, 1]
                current_placement[i, 2] += object_size * (height + 1) * 2

        return current_placement


class TrainStateGoal(ObjectStateGoal):
    """
    This is a goal state generator for the training environment. The goal is placed wrt
    a goal_distance restriction.
    """

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        placement, is_valid = place_targets_with_goal_distance_ratio(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            self.mujoco_simulation.get_object_pos()[
                : self.mujoco_simulation.num_objects
            ],
            goal_distance_ratio=self.mujoco_simulation.goal_distance_ratio,
            goal_distance_min=self.mujoco_simulation.goal_distance_min,
            max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
            max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
            random_state=random_state,
        )

        placement = move_one_object_to_the_air_with_restrictions(
            placement,
            height_range=self.args.height_range,
            object_size=self.mujoco_simulation.simulation_params.object_size,
            random_state=random_state,
            pickup_proba=self.args.pickup_proba,
            stacking_proba=self.args.stacking_proba,
            goal_distance_ratio=self.mujoco_simulation.goal_distance_ratio,
        )
        return placement, is_valid
