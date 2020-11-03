import abc
from typing import Generic, TypeVar

import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.observation.common import Observation, ObservationProvider

SType = TypeVar("SType", bound=SimulationInterface)


class MujocoObservationProvider(ObservationProvider, Generic[SType]):
    """
    Provider for observation coming from Mujoco.
    """

    def __init__(
        self, mujoco_simulation: SType,
    ):
        self.mujoco_simulation = mujoco_simulation

    def sync(self):
        """
        To sync observation from Mujoco simulation, we just need to forward the
        simulation.
        """
        self.mujoco_simulation.forward()


class MujocoObservation(Observation[MujocoObservationProvider[SType]], abc.ABC):
    """
    Interface for all observation coming from Mujoco.
    """


class MujocoQposObservation(MujocoObservation):
    """
    Implement mujoco base qpos observation.
    """

    def get(self) -> np.ndarray:
        """
        Return mujoco qpos, excluding target joints.
        """
        qpos_obs = self.provider.mujoco_simulation.qpos
        qpos_obs[self.provider.mujoco_simulation.qpos_idxs["target_all_joints"]] = 0.0
        return qpos_obs


class MujocoQvelObservation(MujocoObservation):
    """
    Implement mujoco base qvel observation.
    """

    def get(self) -> np.ndarray:
        """
        Return mujoco qvel, excluding target joints.
        """
        qvel_obs = self.provider.mujoco_simulation.qvel
        qvel_obs[self.provider.mujoco_simulation.qvel_idxs["target_all_joints"]] = 0.0
        return qvel_obs
