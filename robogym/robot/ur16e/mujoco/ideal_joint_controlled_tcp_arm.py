from typing import Optional

import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.factories import build_tcp_controller
from robogym.robot.robot_interface import RobotControlParameters
from robogym.robot.ur16e.arm_interface import TABLETOP_EXPERIMENT_INITIAL_POS, Arm
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import FreeDOFTcpArm
from robogym.robot.ur16e.mujoco.joint_controlled_arm import MujocoObservation
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface


class IdealJointControlledTcpArm(Arm):
    """
    An "ideal" joint controlled arm that can immediately achieve
    the positions that are achieved by its controller_arm.
    """

    def __init__(
        self,
        simulation: SimulationInterface,
        robot_control_params: RobotControlParameters,
        solver_simulation: ArmSimulationInterface,
        robot_prefix="robot0:",
        autostep=False,
    ):
        """
        :param simulation: simulation interface for the mujoco UR arm
        :param robot_control_params: Robot control parameters
        :param robot_prefix: prefix to add to the joint names while constructing the mujoco simulation
        :param autostep: when true, calls step() on the simulation whenever a control is set. this
        should only be used only when the Robot is being controlled without a
        simulationrunner in the loop.
        """

        assert robot_control_params.is_tcp_controlled()

        self.simulation = simulation
        self.robot_prefix = robot_prefix
        self.autostep = autostep
        self.joint_group = robot_prefix + "arm_joint_angles"
        self.simulation.register_joint_group(
            self.joint_group, prefix=robot_prefix + "J"
        )

        self._max_position_change = robot_control_params.max_position_change
        self.simulation.set_qpos(self.joint_group, TABLETOP_EXPERIMENT_INITIAL_POS)

        # Currently, we use the same simulation instance between this class and its controller
        # arm, therefore, the controller_arm is never allowed to autostep to avoid double-stepping
        # sim.
        self.controller_arm: FreeDOFTcpArm = build_tcp_controller(
            robot_control_params=robot_control_params,
            initial_qpos=self.simulation.get_qpos(self.joint_group),
            simulation=solver_simulation,
            autostep=False,
        )

        self._is_in_joint_control_mode = False

    def set_simulation_start_position(self, pos: np.ndarray) -> None:
        """Sets simulation qpos (there are no actuators).

        :param pos: Joint positions, also set in the controls.
        """
        self.simulation.set_qpos(self.joint_group, pos)
        self.mj_sim.forward()

    def set_position_control(self, control: np.ndarray) -> None:
        if self.is_in_joint_control_mode:
            self.simulation.set_qpos(self.joint_group, control)
        else:
            self.controller_arm.set_position_control(control)

        if self.autostep:
            self.mj_sim.step()
            if self.is_in_joint_control_mode:
                self.controller_arm.reset()

    def get_name(self) -> str:
        return "mujoco-ideal-arm"

    def observe(self) -> MujocoObservation:
        return MujocoObservation(self.simulation, self.robot_prefix, self.joint_group)

    def zero_control(self) -> np.ndarray:
        return self.controller_arm.zero_control()

    @property
    def is_in_joint_control_mode(self):
        return self._is_in_joint_control_mode

    def switch_to_joint_control(self) -> None:
        """Set flag so that commands are now interpreted as joint space commands. See reach_helper for details on
        why this is necessary."""
        self._is_in_joint_control_mode = True

    def switch_to_tcp_control(self) -> None:
        """Set flag so that commands are now interpreted as TCP space commands. See reach_helper for details on
        why this is necessary."""
        self._is_in_joint_control_mode = False

    def denormalize_position_control(
        self, position_control: np.ndarray, relative_action: bool = False,
    ) -> np.ndarray:
        return self.controller_arm.denormalize_position_control(
            position_control, relative_action
        )

    def get_control_time_delta(self) -> float:
        """Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.

        :return: Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.
        """
        dt = self.mj_sim.nsubsteps * self.mj_sim.model.opt.timestep
        return dt

    def get_robot_transform(self) -> Optional[np.ndarray]:
        return self.controller_arm.get_robot_transform()

    @property
    def mj_sim(self):
        """ MuJoCo MjSim simulation object """
        return self.simulation.mj_sim

    @property
    def max_position_change(self):
        return self._max_position_change

    def reset(self):
        self.simulation.set_qpos(self.joint_group, TABLETOP_EXPERIMENT_INITIAL_POS)
        self.controller_arm.reset()

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        # We use the joint range in xml since there are no actuators for this arm.
        return self.controller_arm.actuator_ctrl_range_upper_bound()

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        # We use the joint range in xml since there are no actuators for this arm.
        return self.controller_arm.actuator_ctrl_range_lower_bound()
