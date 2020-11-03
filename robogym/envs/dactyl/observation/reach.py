import numpy as np

from robogym.observation.goal import GoalObservation


class GoalFingertipPosObservation(GoalObservation):
    """
    Implement goal fingertip pos observation.
    """

    def get(self) -> np.ndarray:
        assert self.provider.goal
        return self.provider.goal["fingertip_pos"]


class GoalIsAchievedObservation(GoalObservation):
    """
    Implement observation indicating if we've achieved the current goal.
    """

    def get(self) -> np.ndarray:
        """
        Get the flag indicating if we've achieved the current goal.
        """
        return self.provider.is_successful
