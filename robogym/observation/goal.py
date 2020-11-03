import abc
from copy import deepcopy
from typing import Callable, Dict, Optional, TypeVar

import numpy as np

from robogym.observation.common import Observation, ObservationProvider, SyncType
from robogym.observation.image import ImageObservationProvider


class GoalObservationProvider(ObservationProvider):
    """
    Provider for observations coming from target goal state.
    """

    def __init__(self, get_goal: Callable[[], dict]):
        self.get_goal_info = get_goal

    def sync(self):
        """
        Goal is property of environment so getting it doesn't come with
        any extra cost. We can just always return latest goal.
       """
        pass

    @property
    def goal(self):
        goal_info = deepcopy(self.get_goal_info()[2])
        goal_dict = goal_info.pop("goal")
        goal_dict.update(goal_info)
        return goal_dict

    @property
    def is_successful(self):
        return np.array([self.get_goal_info()[1]], dtype=np.int32)


class GoalImageObservationProvider(
    GoalObservationProvider, ImageObservationProvider, abc.ABC
):
    def __init__(self, get_goal: Callable[[], dict]):
        GoalObservationProvider.__init__(self, get_goal)
        ImageObservationProvider.__init__(self)


class GoalRenderedImageObservationProvider(GoalImageObservationProvider, abc.ABC):
    """
    Goal observation provider with images rendered for goal state.
    """

    # We only need to re-render goal image at reset_goal.
    SYNC_TYPE = SyncType.RESET_GOAL

    def __init__(
        self, get_goal: Callable[[], dict], goal_qpos_key: str,
    ):
        """
        :param get_goal: The function to get latest goal information.
        :param goal_qpos_key: The key to fetch goal qpos from goal dict.
        """
        super().__init__(get_goal)

        self.goal_qpos_key = goal_qpos_key

        # Numpy array of goal image of shape (n cameras, image_size, image_size, 3)
        self._goal_images: Optional[np.ndarray] = None

    def sync(self):
        super().sync()
        self._goal_images = self._render_goal_images(self.goal[self.goal_qpos_key])

    @property
    def images(self) -> np.ndarray:
        """
        Return numpy array of the goal image.
        """
        assert self._goal_images is not None, "Goal image not rendered yet."
        return self._goal_images

    @abc.abstractmethod
    def _render_goal_images(self, goal_qpos) -> Dict[str, np.ndarray]:
        pass


class GoalRealImageObservationProvider(GoalImageObservationProvider):
    """
    Goal observation provider with images directly from goal dict.
    """

    SYNC_TYPE = SyncType.RESET_GOAL

    def __init__(self, get_goal: Callable[[], dict], goal_image_key: str):
        """
        :param get_goal: The function to get latest goal information.
        :param goal_image_key: The key to fetch goal image from goal dict.
        """
        super().__init__(get_goal)
        self.goal_image_key = goal_image_key

    @property
    def images(self) -> np.ndarray:
        return self.goal[self.goal_image_key]


PType = TypeVar("PType", bound=GoalObservationProvider)


class GoalObservation(Observation[PType], abc.ABC):
    """
    Observation which derives from another observation. There is no strong reason
    to have this but there are cases where we keep redundant observations in order
    to be backward compatible with old policies. This class provides a simple way to
    create new observation based on existing observation.
    """

    pass
