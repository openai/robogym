from typing import Callable, List, Optional, TypeVar

import numpy as np

from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.observation.goal import GoalRenderedImageObservationProvider
from robogym.observation.image import ImageObservationProvider

SType = TypeVar("SType", bound=RearrangeSimulationInterface)


class MujocoImageObservationProvider(ImageObservationProvider):
    """
    Observation provider which provider mujoco rendered image.
    """

    def __init__(
        self, mujoco_simulation: SType, camera_names: List[str], image_size: int
    ):
        """
        :param mujoco_simulation: this mujoco simulation instance.
        :param camera_names: name of the cameras to render images.
        :param image_size: size of rendered image.
        """
        self.mujoco_simulation = mujoco_simulation
        self.camera_names = camera_names
        self.image_size = image_size
        self._images: Optional[np.ndarray] = None

    def sync(self):
        with self.mujoco_simulation.hide_target():
            images = np.array(
                [
                    self.mujoco_simulation.render(
                        width=self.image_size, height=self.image_size, camera_name=cam
                    )
                    for cam in self.camera_names
                ],
                dtype=np.uint8,
            )

            assert images.dtype == np.uint8

            self._images = images

    @property
    def images(self):
        assert self._images is not None
        return self._images


class MujocoGoalImageObservationProvider(GoalRenderedImageObservationProvider):
    """
    Goal observation which renders mujoco image for goal qpos.
    """

    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        camera_names: List[str],
        image_size: int,
        get_goal: Callable[[], dict],
        goal_qpos_key: str,
        hide_robot: bool = True,
    ):
        """
        :param mujoco_simulation: this mujoco simulation instance.
        :param camera_names: name of the cameras to render images.
        :param image_size: size of rendered image.
        :param get_goal: The function to get goal info dict.
        :param goal_qpos_key: The key to get goal qpos from goal info dict.
        :param hide_robot: If true, hide robot from rendered image.
        """
        super().__init__(get_goal, goal_qpos_key)
        self.mujoco_simulation = mujoco_simulation
        self.camera_names = camera_names
        self.image_size = image_size
        self.hide_robot = hide_robot

    def _render_goal_images(self, goal_qpos):
        with self.mujoco_simulation.hide_target(hide_robot=self.hide_robot):
            old_qpos = self.mujoco_simulation.qpos.copy()
            self.mujoco_simulation.mj_sim.data.qpos[:] = goal_qpos.copy()
            self.mujoco_simulation.forward()

            goal_images = np.array(
                [
                    self.mujoco_simulation.render(
                        width=self.image_size, height=self.image_size, camera_name=cam
                    )
                    for cam in self.camera_names
                ]
            )

            self.mujoco_simulation.mj_sim.data.qpos[:] = old_qpos
            self.mujoco_simulation.forward()

            return goal_images
