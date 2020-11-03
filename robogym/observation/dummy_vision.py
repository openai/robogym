import abc
from typing import Callable, Generic, List

import numpy as np

from robogym.observation.goal import GoalRenderedImageObservationProvider
from robogym.observation.image import ImageObservationProvider
from robogym.robot_env import SType


class DummyVisionObservationProvider(ImageObservationProvider, Generic[SType], abc.ABC):
    """
    Image observation provider which produces a placeholder vision observation so that the gym env using
    it has the proper observatin space. This can be replacaed by a Mujoco provider to use rendered Mujoco images.
    """

    def __init__(
        self, camera_names: List[str], image_size: int,
    ):
        super().__init__()
        self._images = np.zeros((len(camera_names), image_size, image_size, 3))

    @property
    def images(self):
        return self._images

    def sync(self):
        pass


class DummyVisionGoalObservationProvider(
    GoalRenderedImageObservationProvider, Generic[SType], abc.ABC
):
    """
    Goal image observation provider which produces a dummy vision goal so that the gym env
    using it has the proper observation space.
    """

    def __init__(
        self,
        get_goal: Callable[[], dict],
        goal_qpos_key: str,
        camera_names: List[str],
        image_size: int,
    ):
        super().__init__(get_goal, goal_qpos_key)
        self._goal_images = np.zeros((len(camera_names), image_size, image_size, 3))

    @property
    def goal_images(self):
        return self._goal_images

    def _render_goal_images(self, goal_qpos):
        return self._goal_images
