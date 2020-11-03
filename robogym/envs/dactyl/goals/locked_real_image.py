import numpy as np
from numpy.random import RandomState

from robogym.envs.dactyl.common.cube_env import CubeSimulationInterface
from robogym.envs.dactyl.common.cube_utils import DEFAULT_CAMERA_NAMES
from robogym.envs.dactyl.goals.locked_parallel import LockedParallelGoal


class LockedRealImageGoal(LockedParallelGoal):
    """
    Goal generation which uses iterate through a sequence of goal images loaded
    from disk.
    """

    def __init__(self, mujoco_simulation: CubeSimulationInterface, goal_data_path: str):
        super().__init__(mujoco_simulation)
        self.goals = np.load(goal_data_path)
        self.goal_idx = 0

    def next_goal(self, random_state: RandomState, goal_state: dict) -> dict:
        """
        Load next goal image.
        """
        num_goals = len(self.goals["quats"])
        goal_image = np.concatenate(
            [
                self.goals[cam][self.goal_idx % num_goals]
                for cam in DEFAULT_CAMERA_NAMES
            ],
            axis=0,
        )

        goal = {
            "cube_quat": self.goals["quats"][self.goal_idx % num_goals],
            "qpos_goal": np.zeros_like(self.mujoco_simulation.qpos),
            "vision_goal": goal_image,
            "goal_type": "flip",
        }

        self.goal_idx += 1
        return goal
