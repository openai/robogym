from typing import Any, Dict, Optional

import numpy as np

from robogym.robot.factories import build_tcp_controller
from robogym.robot.robot_interface import RobotControlParameters
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import FreeDOFTcpArm
from robogym.robot.ur16e.mujoco.joint_controlled_arm import JointControlledArm
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface


class JointControlledTcpArm(JointControlledArm):
    """
    Mujoco implementation of a joint-actuated UR 16e arm.
    """

    def __init__(
        self,
        simulation: ArmSimulationInterface,
        robot_control_params: RobotControlParameters,
        solver_simulation: ArmSimulationInterface,
        robot_prefix="robot0:",
        autostep=False,
        controller_autostep=True,
    ):
        """
        :param simulation: simulation interface for the mujoco UR arm
        :param robot_control_params: Robot control parameters
        :param robot_prefix: prefix to add to the joint names while constructing the mujoco simulation
        :param autostep: when true, calls step() on the simulation whenever a control is set. this
        should only be used only when the Robot is being controlled without a
        simulationrunner in the loop.
        :param controller_autostep: whether the controller arm should autostep.
        For joint controlled TCP arm, this is set to true by default as we always want
        the inner sim to be autostepping when a command is applied as it has a separate
        solver_simulation that will not be stepped by the main env loop.
        """

        assert robot_control_params.is_tcp_controlled()

        super().__init__(
            simulation=simulation,
            robot_control_params=robot_control_params,
            robot_prefix=robot_prefix,
            autostep=autostep,
        )

        inner_tcp_solver_mode = robot_control_params.get_controller_arm_solver_mode()
        inner_params = RobotControlParameters(
            max_position_change=robot_control_params.max_position_change,
            control_mode=robot_control_params.control_mode,
            tcp_solver_mode=inner_tcp_solver_mode,
        )
        self.controller_arm: FreeDOFTcpArm = build_tcp_controller(
            robot_control_params=inner_params,
            initial_qpos=self.simulation.get_qpos(self.joint_group),
            simulation=solver_simulation,
            autostep=controller_autostep,
        )

        self.reset_controller_error = robot_control_params.arm_reset_controller_error

    def get_name(self) -> str:
        return "mujoco-joint-controlled-tcp-arm"

    @property
    def is_in_joint_control_mode(self):
        return self.controller_arm.is_in_joint_control_mode

    def zero_control(self) -> np.ndarray:
        return self.controller_arm.zero_control()

    def get_robot_transform(self) -> Optional[np.ndarray]:
        return self.controller_arm.get_robot_transform()

    def denormalize_position_control(
        self, position_control: np.ndarray, relative_action: bool = False,
    ) -> np.ndarray:
        return self.controller_arm.denormalize_position_control(
            position_control, relative_action
        )

    def get_helper_robot(self):
        """Retrieve the internal controller robot. External users should only query things, not cause modifications.

        :return: The internal controller robot.
        """
        return self.controller_arm

    def set_position_control(self, control: np.ndarray) -> None:
        if self.reset_controller_error:
            self.controller_arm.sync_to(
                joint_positions=self.observe().joint_positions(), joint_controls=None
            )

        self.controller_arm.set_position_control(control)
        control = self.controller_arm.get_joint_state()
        super().set_position_control(control)

    def reset(self) -> None:
        super().reset()
        self.controller_arm.reset()

    def switch_to_joint_control(self) -> None:
        """Set flag so that commands are now interpreted as joint space commands. See reach_helper for details on
        why this is necessary."""
        self.controller_arm.switch_to_joint_control()

    def switch_to_tcp_control(self) -> None:
        """Set flag so that commands are now interpreted as TCP space commands. See reach_helper for details on
        why this is necessary."""
        self.controller_arm.switch_to_tcp_control()

    def on_observations_updated(self, new_observations: Dict[str, Any]) -> None:
        """Event to notify the robot that new observations have been collected. See parents for more detailed
        documentation.

        Overridden here so that we update sub-simulations for TCP controlled arms that have them.

        :param new_observations: New observations collected.
        """

        # update the gripper in the controller arm's simulation if needed with the new observations for this step
        if self.controller_arm:
            pos = new_observations["gripper_qpos"]
            ctrl = new_observations["gripper_controls"]
            self.controller_arm.simulation.gripper.sync_to(
                joint_positions=pos, joint_controls=ctrl
            )
