from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_objects_with_no_constraint
from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.utils import rotation


class ObjectStackGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs] = GoalArgs(),
        fixed_order: bool = True,
    ):
        """
        :param fixed_order: whether goal requires objects stacked in a random order.
        """
        super().__init__(mujoco_simulation, args)
        self.fixed_order = fixed_order

    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        # place only bottom object
        bottom_positions, goal_valid = place_objects_with_no_constraint(
            self.mujoco_simulation.get_object_bounding_boxes()[:1],
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
            max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
            random_state=random_state,
        )
        goal_positions = np.repeat(
            bottom_positions, self.mujoco_simulation.num_objects, axis=0
        )

        object_size = self.mujoco_simulation.simulation_params.object_size
        block_orders = list(range(self.mujoco_simulation.num_objects))
        if not self.fixed_order:
            random_state.shuffle(block_orders)

        bottom_block_idx = block_orders[0]
        goal_positions[bottom_block_idx] = bottom_positions[0]

        for i in range(1, self.mujoco_simulation.num_objects):
            new_pos = bottom_positions[0].copy()
            new_pos[2] += i * object_size * 2
            goal_positions[block_orders[i]] = new_pos

        return goal_positions, goal_valid

    def is_object_grasped(self):
        grasped = self.mujoco_simulation.get_object_gripper_contact()
        # in teleop, somehow we could grasp object with mujoco detecting only one gripper touched the object.
        return np.array([x[0] + x[1] for x in grasped])

    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        gripper_pos = current_state["obj_pos"] - current_state["gripper_pos"]
        obj_pos = goal_state["obj_pos"] - current_state["obj_pos"]

        return {
            "obj_pos": obj_pos,
            "gripper_pos": gripper_pos,
            "obj_rot": self.rot_dist_func(goal_state, current_state),
        }

    def current_state(self) -> dict:
        gripper_pos = self.mujoco_simulation.mj_sim.data.get_site_xpos(
            "robot0:grip"
        ).copy()
        gripper_pos = np.array([gripper_pos])

        current_state = super().current_state()
        current_state.update(
            {"gripper_pos": gripper_pos, "grasped": self.is_object_grasped()}
        )
        return current_state

    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        relative_goal = self.relative_goal(goal_state, current_state)
        pos_distances = np.linalg.norm(relative_goal["obj_pos"], axis=-1)
        gripper_distances = np.linalg.norm(relative_goal["gripper_pos"], axis=-1)
        rot_distances = rotation.quat_magnitude(
            rotation.quat_normalize(rotation.euler2quat(relative_goal["obj_rot"]))
        )

        return {
            "relative_goal": relative_goal.copy(),
            "gripper_pos": gripper_distances,
            "obj_pos": pos_distances,
            "obj_rot": rot_distances,
            "grasped": current_state["grasped"],
        }
