import os
from typing import List

import attr
import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.holdouts import STATE_DIR
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface


@attr.s(auto_attribs=True)
class HoldoutGoalArgs(GoalArgs):

    randomize_goal_rot: bool = True

    # Path for each goal state. If multiple paths are
    # provided, goal generation will round robin through
    # all these states.
    goal_state_paths: List[str] = []


class HoldoutObjectStateGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: HoldoutGoalArgs = HoldoutGoalArgs(),
    ):
        super().__init__(mujoco_simulation, args=args)
        self.goal_pos = []
        self.goal_quat = []

        for goal_state_path in args.goal_state_paths:
            goal_state = np.load(os.path.join(STATE_DIR, goal_state_path))
            self.goal_pos.append(
                goal_state["obj_pos"][: self.mujoco_simulation.num_objects]
            )
            self.goal_quat.append(
                goal_state["obj_quat"][: self.mujoco_simulation.num_objects]
            )

        self.goal_idx = 0

    def next_goal(self, random_state: RandomState, current_state: dict):
        goal = super().next_goal(random_state, current_state)

        if self.goal_idx >= len(self.goal_pos):
            self.goal_idx = 0

        return goal

    def _sample_next_goal_positions(self, random_state: RandomState):
        if len(self.goal_pos) > 0:
            return self.goal_pos[self.goal_idx], True
        else:
            return super()._sample_next_goal_positions(random_state)

    def _sample_next_goal_orientations(self, random_state: RandomState):
        if len(self.goal_quat) > 0:
            return self.goal_quat[self.goal_idx]
        else:
            return super()._sample_next_goal_orientations(random_state)

    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        dist = super().goal_distance(goal_state, current_state)
        # calculate relative positions,
        # this logic does NOT handle duplicate objects.
        if self.mujoco_simulation.num_groups == self.mujoco_simulation.num_objects:
            goal_pos, cur_pos = goal_state["obj_pos"], current_state["obj_pos"]
            anchor_idx = np.argmin(goal_pos[:, 2])
            target_rel_pos = goal_pos - goal_pos[anchor_idx] + cur_pos[anchor_idx]
            dist["rel_dist"] = np.linalg.norm(cur_pos - target_rel_pos, axis=1)
        return dist
