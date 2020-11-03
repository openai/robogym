from copy import deepcopy
from typing import Callable, Dict, Optional

import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.composite.composite_robot import CompositeRobot
from robogym.robot.gripper.mujoco.mujoco_robotiq_gripper import MujocoRobotiqGripper
from robogym.robot.robot_interface import (
    Robot,
    RobotControlParameters,
    RobotObservation,
    TcpSolverMode,
)
from robogym.robot.ur16e.mujoco.ideal_joint_controlled_tcp_arm import (
    IdealJointControlledTcpArm,
)
from robogym.robot.ur16e.mujoco.joint_controlled_arm import JointControlledArm
from robogym.robot.ur16e.mujoco.joint_controlled_tcp_arm import JointControlledTcpArm
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface


class MujocoURGripperCompositeObservation(RobotObservation):
    def __init__(self, robot_obs: Dict[str, RobotObservation]):
        self.robot_obs = robot_obs

    def gripper_aperture(self) -> np.ndarray:
        """:return: Gripper's linear position of the fingers, like its actuator position."""
        return self.robot_obs["gripper"].joint_positions()

    def joint_positions(self) -> np.ndarray:
        return self.robot_obs["arm"].joint_positions()

    def joint_velocities(self) -> np.ndarray:
        return self.robot_obs["arm"].joint_velocities()

    def tcp_xyz(self) -> np.ndarray:
        """
        :return: Tooltip position in Cartesian coordinate space
        """
        return self.robot_obs["arm"].tcp_xyz()  # type: ignore

    def tcp_vel(self) -> np.ndarray:
        """
        :return: Tooltip velocity in Cartesian coordinate space
        """
        return self.robot_obs["arm"].tcp_vel()  # type: ignore

    def tcp_rot(self) -> np.ndarray:
        """
        :return: Tooltip velocity in Cartesian coordinate space
        """
        return self.robot_obs["arm"].tcp_rot()  # type: ignore

    def tcp_force(self) -> np.ndarray:
        """
        :return: TCP force in world coordinates
        """
        return self.robot_obs["arm"].tcp_force()  # type: ignore

    def tcp_torque(self) -> np.ndarray:
        """
        :return: TCP torque in world coordinates
        """
        return self.robot_obs["arm"].tcp_torque()  # type: ignore

    def is_in_safety_stop(self) -> bool:
        """
        :return: True if the arm is in a safety stop, False otherwise.
        """
        return self.robot_obs["arm"].is_in_safety_stop()  # type: ignore

    def gripper_qpos(self) -> np.ndarray:
        """
        :return: Gripper joint positions
        """
        return self.robot_obs["gripper"].joint_positions()

    def gripper_vel(self) -> np.ndarray:
        """
        :return: Gripper joint velocities
        """
        return self.robot_obs["gripper"].joint_velocities()

    def gripper_controls(self) -> np.ndarray:
        """:return: Gripper's linear target position."""
        return self.robot_obs["gripper"].joint_controls()  # type: ignore

    def timestamp(self) -> float:
        return self.robot_obs["arm"].timestamp()


class URGripperCompositeRobot(CompositeRobot):
    OBS_LABEL = ["arm", "gripper"]
    OBS_CLS = MujocoURGripperCompositeObservation


class MujocoURTcpJointGripperCompositeRobot(URGripperCompositeRobot):
    ROBOT_CLS = [JointControlledTcpArm, MujocoRobotiqGripper]


class MujocoURJointGripperCompositeRobot(URGripperCompositeRobot):
    ROBOT_CLS = [JointControlledArm, MujocoRobotiqGripper]


class MujocoIdealURGripperCompositeRobot(URGripperCompositeRobot):
    ROBOT_CLS = [IdealJointControlledTcpArm, MujocoRobotiqGripper]


def build_composite_robot(
    robot_control_params: RobotControlParameters, simulation: SimulationInterface
):
    robot_cls: Optional[Callable[..., Robot]] = None

    solver_simulation: Optional[SimulationInterface] = None

    if robot_control_params.requires_solver_sim():
        solver_simulation = build_solver_sim(
            robot_control_params=robot_control_params,
            n_substeps=simulation.n_substeps,
            mujoco_timestep=simulation.mj_sim.model.opt.timestep,
        )
        robot_cls = (
            MujocoURTcpJointGripperCompositeRobot  # JointActuated, TCPControlled
        )
    elif robot_control_params.is_tcp_controlled():
        solver_simulation = simulation
        robot_cls = MujocoIdealURGripperCompositeRobot  # MocapActuated, TCPControlled
    elif robot_control_params.is_joint_actuated():
        robot_cls = MujocoURJointGripperCompositeRobot  # JointActuated, JointControlled
    else:
        raise ValueError(
            f"Unknown combination of robot_control_params to "
            f"build a composite robot: {robot_control_params}"
        )

    assert robot_cls
    robot = robot_cls(
        simulation=simulation,
        robot_control_params=robot_control_params,
        solver_simulation=solver_simulation,
    )
    assert isinstance(robot, Robot)

    return robot


def build_solver_sim(
    robot_control_params: RobotControlParameters,
    n_substeps: int,
    mujoco_timestep: float,
) -> ArmSimulationInterface:
    solver_sim_params = deepcopy(robot_control_params)
    if solver_sim_params.tcp_solver_mode is TcpSolverMode.MOCAP_IK:
        solver_sim_params.tcp_solver_mode = TcpSolverMode.MOCAP
    return ArmSimulationInterface.build(
        robot_control_params=solver_sim_params,
        n_substeps=n_substeps,
        mujoco_timestep=mujoco_timestep,
    )
