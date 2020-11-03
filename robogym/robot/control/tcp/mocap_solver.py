from typing import Optional

import numpy as np
from gym.envs.robotics import utils

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.control.tcp.solver import PrincipalAxis, Solver
from robogym.utils import rotation


class MocapSolver(Solver):
    """
    A TCP solver class that uses Mujoco's mocap weld
    to track and apply TCP control.
    """

    JOINT_MAPPING = {
        PrincipalAxis.PITCH: 5,
    }

    def __init__(
        self,
        simulation: SimulationInterface,
        body_name: str,
        robot_prefix: str,
        quat_dof_dims: np.ndarray,
        alignment_axis: Optional[PrincipalAxis],
    ):
        super().__init__(
            simulation, body_name, robot_prefix, quat_dof_dims, alignment_axis
        )

    def get_tcp_quat(self, ctrl: np.ndarray) -> np.ndarray:
        assert len(ctrl) == len(
            self.dof_dims
        ), f"Unexpected control dim {len(ctrl)}, should be {len(self.dof_dims)}"

        euler = np.zeros(3)
        euler[self.dof_dims_axes] = ctrl
        quat = rotation.euler2quat(euler)
        gripper_quat = self.mj_sim.data.get_body_xquat(self.body_name)

        if self.alignment_axis is not None:
            return (
                self.align_axis(
                    rotation.quat_mul(gripper_quat, quat), self.alignment_axis.value
                )
                - gripper_quat
            )
        return rotation.quat_mul(gripper_quat, quat) - gripper_quat

    def set_action(self, action: np.ndarray) -> None:
        utils.mocap_set_action(self.mj_sim, action)

    def reset(self):
        utils.reset_mocap_welds(self.mj_sim)
        utils.reset_mocap2body_xpos(self.mj_sim)

    @staticmethod
    def align_axis(cmd_quat, axis):
        """ Align quaternion into given axes """
        alignment = np.zeros(3)
        alignment[axis] = 1
        mtx = rotation.quat2mat(cmd_quat)
        # Axis that is the closest (by dotproduct) to alignment
        axis_nr = np.abs((alignment.T @ mtx)).argmax()

        # Axis of the cmd_quat
        axis = mtx[:, axis_nr]
        axis = axis * np.sign(axis @ alignment)

        difference_quat = rotation.vectors2quat(axis, alignment)

        return rotation.quat_mul(difference_quat, cmd_quat)
