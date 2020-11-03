import numpy as np

from robogym.observation.goal import GoalObservation


class GoalCubePosObservation(GoalObservation):
    """
    Implement goal cube position observation.
    """

    def get(self) -> np.ndarray:
        """
        Locked cube doesn't take position as part of goal.
        """
        return np.zeros(3)
