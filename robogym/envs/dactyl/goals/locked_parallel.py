from typing import Set

import numpy as np
from numpy.random import RandomState

from robogym.envs.dactyl.common import cube_utils
from robogym.envs.dactyl.common.cube_env import CubeSimulationInterface
from robogym.goal.goal_generator import GoalGenerator
from robogym.utils import rotation


class LockedParallelGoal(GoalGenerator):
    """
    Generates random orientation goals for the locked cube. Specifically, the goal
    orientation always has the sides aligned with, or "parallel" with, the x-y-z axes.
    Hence the name "Parallel" for this goal generator.

    Note: historically, we've also called this goal generator "XYZ," so you might see
    that name mentioned in docs etc.
    """

    def __init__(self, mujoco_simulation: CubeSimulationInterface):
        """
        Create new FaceCubeSolverGoalGenerator object

        :param success_threshold: Dictionary of threshold levels for cube orientation and face
            rotation, for which we consider the cube "aligned" with the goal
        """
        self.mujoco_simulation = mujoco_simulation
        super().__init__()

    def next_goal(self, random_state: RandomState, current_state: dict) -> dict:
        """ Generate a new goal from current cube goal state """
        # we just sample a random orientation, so current goal_state isn't used
        z_quat = cube_utils.uniform_z_aligned_quat(random_state)
        quat_choice = random_state.randint(len(cube_utils.PARALLEL_QUATS))
        parallel_quat = cube_utils.PARALLEL_QUATS[quat_choice]
        goal_quat = rotation.quat_mul(z_quat, parallel_quat)

        # Create qpos for goal state (with just cube quat set) for rendering purposes.
        qpos_goal = np.zeros_like(self.mujoco_simulation.qpos)
        qpos_inds = self.mujoco_simulation.qpos_idxs["cube_rotation"]
        qpos_goal[qpos_inds] = goal_quat
        qpos_pos_inds = self.mujoco_simulation.qpos_idxs["cube_position"]
        qpos_goal[qpos_pos_inds] = np.array([0.0, 0.0, -0.025])
        return {"cube_quat": goal_quat, "qpos_goal": qpos_goal, "goal_type": "flip"}

    def current_state(self) -> dict:
        """ Extract current cube goal state """
        return {
            "cube_quat": self.mujoco_simulation.get_qpos("cube_rotation"),
        }

    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        """
        Calculate a difference in the 'goal space' between current state and the target goal
        """
        return {
            # We don't care about pos in goal. But we have to include it here because
            # we need cube pos to be present in relative_goal observation.
            "cube_pos": np.zeros(3),
            # Quaternion difference of a rotation
            "cube_quat": rotation.quat_difference(
                goal_state["cube_quat"], current_state["cube_quat"]
            ),
        }

    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        """ Distance from the current goal to the target state. """
        relative_goal = self.relative_goal(goal_state, current_state)

        goal_distance = {
            "cube_quat": rotation.quat_magnitude(relative_goal["cube_quat"]),
        }

        return goal_distance

    def goal_types(self) -> Set[str]:
        return {"flip"}
