import abc
from enum import Enum
from typing import Dict, Optional

import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface


class PrincipalAxis(Enum):
    ROLL = 0
    PITCH = 2
    YAW = 1


class Solver(abc.ABC):
    """
    TCP Solver Base.
    """

    JOINT_MAPPING: Dict[PrincipalAxis, int] = {}

    def __init__(
        self,
        simulation: SimulationInterface,
        body_name: str,
        robot_prefix: str,
        quat_dof_dims: np.ndarray,
        alignment_axis: Optional[PrincipalAxis],
    ):
        """

        :param simulation: Mujoco SimulationInterface
        :param quat_dof_dims: Quaternion dimensions that are beng controlled
        :param body_name: Body name that is controlled by TCP
        :param alignment_axis: Alignment axis to force align quaternion to.
        """
        self.simulation = simulation
        self.body_name = body_name
        self.robot_prefix = robot_prefix
        self.dof_dims = quat_dof_dims
        self.dof_dims_axes = [axis.value for axis in quat_dof_dims]
        self.alignment_axis = alignment_axis

    @abc.abstractmethod
    def get_tcp_quat(self, quat_ctrl: np.ndarray) -> np.ndarray:
        """
        Returns a relative TCP quaternion to be applied
        given quat control
        :param quat_ctrl: Relative quaternion control array
        :return:
        """
        pass

    @abc.abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """
        Sets 6-DOF TCP action via an underlying controller
        :param action:
        :return:
        """
        pass

    def get_joint_mapping(self) -> np.ndarray:
        """
        Map DOF dimensions to joints if we should restrict this DOF based on a joint range.
        :return: list of joint ids or None that correspond to dof_dims_axes.
        """
        return np.array(
            [self.JOINT_MAPPING.get(dof_dimension) for dof_dimension in self.dof_dims]
        )

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset TCP solver state.
        """
        pass

    @property
    def mj_sim(self):
        """ MuJoCo MjSim simulation object """
        return self.simulation.mj_sim
