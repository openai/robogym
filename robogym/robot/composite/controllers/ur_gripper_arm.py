import logging
from enum import Enum
from typing import Optional

import numpy as np

from robogym.robot.composite.ur_gripper_arm import URGripperCompositeRobot
from robogym.robot.gripper.mujoco.mujoco_robotiq_gripper import MujocoRobotiqGripper
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import FreeRollYawTcpArm
from robogym.robot_env import RobotEnv


class Direction(Enum):
    POS: int = 1
    NEG: int = -1


class URGripperArmController:
    """
    Controller to convert human interpretable action into relative action control.

    This controller assumes relative action control vector in [-1, 1] of

    [mocap_x, mocap_y, mocap_z, wrist_joint_angle, gripper_joint_angle]
    """

    # The max speed.
    MAX_SPEED = 1.0

    # The minimum speed.
    MIN_SPEED = 0.0

    SPEED_CHANGE_PERCENT = 0.2

    def __init__(self, env: RobotEnv):
        self._speeds = np.array([0.3, 0.5, 0.3])
        assert isinstance(env.robot, URGripperCompositeRobot)
        self.env = env

    @property
    def robot(self):
        return self.env.robot

    @property
    def arm_speed(self):
        """
        The speed that arm moves.
        """
        return self._speeds[0]

    @property
    def wrist_speed(self):
        """
        The speed that wrist rotates.
        """
        return self._speeds[1]

    @property
    def gripper_speed(self):
        """
        The speed that gripper opens/closes.
        """
        return self._speeds[2]

    def zero_control(self):
        """
        Returns zero control, meaning gripper shouldn't move by applying this action.
        """
        return self.robot.zero_control()

    def speed_up(self):
        """
        Increase gripper moving speed.
        """
        self._speeds = np.minimum(
            self._speeds * (1 + self.SPEED_CHANGE_PERCENT), self.MAX_SPEED
        )

    def speed_down(self):
        """
        Decrease gripper moving speed.
        """
        self._speeds = np.maximum(
            self._speeds * (1 - self.SPEED_CHANGE_PERCENT), self.MIN_SPEED
        )

    def move_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._move(0, direction)

    def move_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._move(1, direction)

    def move_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._move(2, direction)

    def _move(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        ctrl = self.zero_control()
        ctrl[axis] = self.arm_speed * direction.value
        return ctrl

    def move_gripper(self, direction: Direction):
        """
        Open/close gripper.
        """
        ctrl = self.zero_control()
        ctrl[-1] = self.gripper_speed * direction.value
        return ctrl

    def tilt_gripper(self, direction: Direction) -> np.ndarray:
        """
        Tilt the gripper
        """
        ctrl = self.zero_control()

        if not isinstance(self.robot.robots[0].controller_arm, FreeRollYawTcpArm):
            logging.warning(
                "This robot doesn't support tilting gripper, skip this action."
            )

            return ctrl

        ctrl[-3] = self.arm_speed * direction.value
        return ctrl

    def rotate_wrist(self, direction: Direction) -> np.ndarray:
        """
        Rotate the wrist joint.
        """
        ctrl = self.zero_control()
        ctrl[-2] = self.wrist_speed * direction.value
        return ctrl

    def get_gripper_actuator_force(self) -> Optional[float]:
        """
        Get actuator force for the gripper.
        """
        gripper = self.robot.robots[1]
        if isinstance(gripper, MujocoRobotiqGripper):
            return gripper.simulation.mj_sim.data.actuator_force[gripper.actuator_id]
        else:
            return None

    def get_gripper_regrasp_status(self) -> Optional[bool]:
        """Returns True if re-grasp feature is ON for the gripper, False if it's OFF, and None if the gripper does
        not report the status of the re-grasp feature, or it does not have that feature.

        :return: True if re-grasp feature is ON for the gripper, False if it's OFF, and None if the gripper does
        not report the status of the re-grasp feature, or it does not have that feature.
        """
        status: Optional[bool]

        gripper = self.robot.robots[1]

        if not hasattr(gripper, "regrasp_enabled"):
            return None
        elif gripper.regrasp_enabled:
            is_currently_regrasping = gripper.regrasp_helper.regrasp_command is not None
            status = is_currently_regrasping
        else:
            status = None
        return status
