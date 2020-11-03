from typing import Callable, Optional

import numpy as np

from robogym.robot.robot_interface import ControlMode, RobotControlParameters
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import (
    FreeDOFTcpArm,
    FreeRollYawTcpArm,
    FreeWristTcpArm,
)
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface


def build_tcp_controller(
    robot_control_params: RobotControlParameters,
    simulation: ArmSimulationInterface,
    initial_qpos: Optional[np.ndarray],
    autostep: bool = False,
) -> FreeDOFTcpArm:
    """

    :param robot_control_params:
    :param simulation: Solver simulation.
    :param initial_qpos: Initial state for the solver robot.
    :param autostep:
    :return:
    """
    assert robot_control_params.is_tcp_controlled()
    robot_cls: Optional[Callable[..., FreeDOFTcpArm]] = None
    if robot_control_params.control_mode is ControlMode.TCP_WRIST:
        robot_cls = FreeWristTcpArm
    elif robot_control_params.control_mode is ControlMode.TCP_ROLL_YAW:
        robot_cls = FreeRollYawTcpArm
    else:
        raise ValueError("Unknown control mode for TCP.")

    robot = robot_cls(
        simulation=simulation,
        initial_qpos=initial_qpos,
        robot_control_params=robot_control_params,
        autostep=autostep,
    )
    assert isinstance(robot, FreeDOFTcpArm)

    return robot
