from abc import ABC, abstractmethod
from typing import Any, Dict, Set

from numpy.random import RandomState


class GoalGenerator(ABC):
    """
    Interface for goal generation

    Goals are represented as states in which the environment must get. Usually, only a subset
    of the full state vector (or "qpos" in Mujoco-terms) determines the goal. For example, for
    cube-flipping tasks only the orientation (quaternion) of the cube matters, the position of
    the ShadowHand joints do not.

    The GoalGenerator interface requires two goal state dicts {goal_name: state}, where the state
    is usually a scalar or numpy.ndarray representing some subset of the full state (like the pose
    of the robot or a position of an object in the environment). The two dicts are:

    - current_state: current state of the environment
    - goal_state: the state in which the goal has been achieved

    The environment will use the goal_distance() method and compare it to a threshold to determine
    if the goal_state has been achieved.
    """

    def __init__(self):
        # reached_terminal_state is used for longer sequences of goals to signal that the final
        # goal state has been reached.
        self.reached_terminal_state = False

    @abstractmethod
    def next_goal(self, random_state: RandomState, current_state: dict) -> dict:
        """ Generate a new goal from current cube goal state """
        raise NotImplementedError

    @abstractmethod
    def current_state(self) -> dict:
        """ Extract current cube goal state """
        raise NotImplementedError

    @abstractmethod
    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        """
        Calculate a difference in the 'goal space' between current state and the target goal.

        For example, for a (x, y, z) position this might be the vector from the current_state
        to the goal_state (goal_state - current_state). The magnitude of the relative_goal is
        often used as the goal_distance.
        """
        raise NotImplementedError

    @abstractmethod
    def goal_distance(self, goal_state: dict, current_state: dict) -> Dict[str, Any]:
        """ Distance from the current goal to the target state. """
        raise NotImplementedError

    def goal_types(self) -> Set[str]:
        """Returns the set of goal types supported by this goal generator"""
        return {"generic"}

    def goal_reachable(self, goal_state: dict, current_state: dict) -> bool:
        """ Return if target goal state is reachable from current state. """
        return True

    def reset(self, random_state: RandomState) -> None:
        """ Reset state of the goal generator. Will be called when env is reset. """
        self.reached_terminal_state = False
