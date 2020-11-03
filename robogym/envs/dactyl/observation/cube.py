import numpy as np

from robogym.observation.goal import GoalObservation
from robogym.observation.mujoco import MujocoObservation
from robogym.utils.rotation import quat_normalize


class MujocoCubePosObservation(MujocoObservation):
    """
    Implement mujoco base cube position observation.
    """

    def get(self) -> np.ndarray:
        """
        Get cube position.
        """
        return self.provider.mujoco_simulation.get_qpos("cube_position")


class MujocoCubeRotObservation(MujocoObservation):
    """
    Implement mujoco base cube rotation observation.
    """

    def get(self) -> np.ndarray:
        """
        Get cube position.
        """
        return quat_normalize(self.provider.mujoco_simulation.get_qpos("cube_rotation"))


class GoalCubeRotObservation(GoalObservation):
    """
    Implement goal cube rotation observation.
    """

    def get(self) -> np.ndarray:
        """
        Get cube position.
        """
        assert self.provider.goal
        return quat_normalize(self.provider.goal["cube_quat"])


class GoalQposObservation(GoalObservation):
    """
    Retrieves the qpos corresponding to the goal state.
    """

    def get(self) -> np.ndarray:
        """
        Get goal qpos.
        """
        assert self.provider.goal
        return self.provider.goal["qpos_goal"]


class GoalIsAchievedObservation(GoalObservation):
    """
    Implement observation indicating if we've achieved the current goal.
    """

    def get(self) -> np.ndarray:
        """
        Get the flag indicating if we've achieved the current goal.
        """
        return self.provider.is_successful
