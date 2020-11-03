from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_objects_with_no_constraint
from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface


class ObjectReachGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs] = GoalArgs(),
    ):
        super().__init__(mujoco_simulation, args=args)

        assert (
            self.mujoco_simulation.num_objects == 1
        ), "reach only supports one objects"
        self.mujoco_simulation: RearrangeSimulationInterface = mujoco_simulation

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        goal_positions, goal_valid = place_objects_with_no_constraint(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
            max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
            random_state=random_state,
        )

        self.mujoco_simulation.set_object_pos(goal_positions)

        # set reach target to specified height above the table.
        goal_positions[0][2] += self.mujoco_simulation.simulation_params.target_height
        return goal_positions, goal_valid

    def current_state(self) -> dict:
        """
        Here we use gripper position as realized goal state, so the goal distance is distance
        betweeen gripper and goal position.
        """
        gripper_pos = self.mujoco_simulation.mj_sim.data.get_site_xpos(
            "robot0:grip"
        ).copy()
        obj_pos = np.array([gripper_pos])
        return dict(
            obj_pos=obj_pos, obj_rot=np.zeros_like(obj_pos)
        )  # We don't care about rotation.


class DeterministicReachGoal(ObjectReachGoal):
    """
    Reach goal generator with a deterministic sequence of goal positions.
    """

    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs] = GoalArgs(),
    ):
        super().__init__(mujoco_simulation, args=args)

        p0 = np.array([1.50253879, 0.36960144, 0.5170952])
        p1 = np.array([1.32253879, 0.53960144, 0.5170952])
        self.idx = 0
        self.all_positions = [p0, p1]

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        self.idx = (self.idx + 1) % len(self.all_positions)
        goal_pos = self.all_positions[self.idx]
        goal_positions = np.array([goal_pos])

        self.mujoco_simulation.set_object_pos(goal_positions)
        goal_positions[0][2] += self.mujoco_simulation.simulation_params.target_height
        return goal_positions, True
