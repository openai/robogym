from typing import Dict, List, Optional

import numpy as np

from robogym.robot.control.tcp.mocap_solver import MocapSolver
from robogym.robot.control.tcp.solver import PrincipalAxis
from robogym.robot.robot_interface import RobotControlParameters, TcpSolverMode
from robogym.robot.ur16e.arm_interface import Arm
from robogym.robot.ur16e.mujoco.joint_controlled_arm import MujocoObservation
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface

# speed factors that are scaled by the max_position_change.
DOF_DIM_SPEED_SCALE: Dict[PrincipalAxis, float] = {
    PrincipalAxis.ROLL: np.deg2rad(200),
    PrincipalAxis.PITCH: np.deg2rad(600),
    PrincipalAxis.YAW: np.deg2rad(300),
}


class FreeDOFTcpArm(Arm):
    """
     Mujoco implementation of a tool center point (TCP) actuated UR 16e arm
     with a user-defined set of quaternion  DOFs.
    """

    JOINT_DRIFT_THRESHOLD = np.deg2rad(
        1
    )  # an additional buffer to prevent us from hitting joint range

    DOF_DIMS: List[PrincipalAxis] = []
    ALIGN_AXIS: Optional[PrincipalAxis] = None

    def __init__(
        self,
        simulation: ArmSimulationInterface,
        robot_control_params: RobotControlParameters,
        initial_qpos: Optional[List],
        robot_prefix="robot0:",
        autostep=False,
    ):
        """
        :param simulation: simulation interface for the mujoco robot.
        :param robot_control_params: Robot control parameters
        :param initial_qpos: The initial valueto be applied to the simulation.
        :param robot_prefix: prefix to add to the joint names while constructing the mujoco simulation.
        :param autostep: when true, calls step() on the simulation whenever a control is set. this
        should only be used only when the Robot is being controlled without a
        simulation runner in the loop.
        """
        super().__init__()
        self.simulation = simulation
        self.robot_prefix = robot_prefix
        self.autostep = autostep
        self.joint_group = robot_prefix + "arm_joint_angles"
        self.simulation.register_joint_group(
            self.joint_group, prefix=robot_prefix + "J"
        )

        if initial_qpos is not None:
            self.simulation.set_qpos(self.joint_group, initial_qpos)
        self.simulation.forward()

        assert (
            robot_control_params.max_position_change
            and robot_control_params.max_position_change > 0.0
        ), "Position multiplier must be a positive number"
        self._max_position_change = robot_control_params.max_position_change

        self.speed_per_dof_dim = [
            DOF_DIM_SPEED_SCALE[axis] * self._max_position_change
            for axis in self.DOF_DIMS
        ]
        self.is_in_joint_control_mode = False

        tcp_solver_mode = robot_control_params.tcp_solver_mode
        assert (
            tcp_solver_mode is TcpSolverMode.MOCAP
        ), f"Invalid solver mode: {tcp_solver_mode}"

        self.solver = MocapSolver(
            simulation=self.simulation,
            body_name="robot0:gripper_tcp",
            robot_prefix=self.robot_prefix,
            quat_dof_dims=self.DOF_DIMS,
            alignment_axis=self.ALIGN_AXIS,
        )

    def get_name(self) -> str:
        return "unnamed-mujoco-ur16e-tcp-arm"

    def switch_to_joint_control(self) -> None:
        """Set flag so that commands are now interpreted as joint space commands. See reach_helper for details on
        why this is necessary."""
        self.is_in_joint_control_mode = True

    def switch_to_tcp_control(self) -> None:
        """Set flag so that commands are now interpreted as TCP space commands. See reach_helper for details on
        why this is necessary."""
        self.is_in_joint_control_mode = False

    @property
    def mj_sim(self):
        """ MuJoCo MjSim simulation object """
        return self.simulation.mj_sim

    def get_control_time_delta(self) -> float:
        """Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.

        :return: Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.
        """
        dt = self.mj_sim.nsubsteps * self.mj_sim.model.opt.timestep
        return dt

    def zero_control(self):
        return np.zeros(3 + len(self.DOF_DIMS))

    def get_robot_transform(self) -> Optional[np.ndarray]:
        """Return the robot transformation wrt its mujoco world. Coordinates are (xyz + rot_quat).
        :return: Return the robot transformation wrt its mujoco world.
        """
        robot_xyz = self.mj_sim.data.get_body_xpos(
            f"{self.robot_prefix}base_link"
        ).copy()
        robot_quat = self.mj_sim.data.get_body_xquat(
            f"{self.robot_prefix}base_link"
        ).copy()
        return np.concatenate((robot_xyz, robot_quat))

    def constrain_quat_ctrl(self, ctrl: np.ndarray):
        """
        Constrain quat control if the solver defines a mapping between control dimensions and
        joints
        :param ctrl:
        :return:
        """
        joint_ids = self.solver.get_joint_mapping()
        if not any(joint_ids):
            return ctrl

        joint_ids_mask = [i for i, v in enumerate(joint_ids) if v is not None]
        joint_mask = np.array(joint_ids[joint_ids_mask], dtype=np.int8)

        joint_pos = self.observe().joint_positions()[joint_mask]
        ctrl[joint_ids_mask] = np.clip(
            ctrl[joint_ids_mask],
            self.actuator_ctrl_range_lower_bound()[joint_mask]
            + self.JOINT_DRIFT_THRESHOLD
            - joint_pos,
            self.actuator_ctrl_range_upper_bound()[joint_mask]
            - self.JOINT_DRIFT_THRESHOLD
            - joint_pos,
        )
        return ctrl

    @property
    def max_position_change(self):
        return self._max_position_change

    def denormalize_position_control(
        self, position_control: np.ndarray, relative_action: bool = False,
    ) -> np.ndarray:

        # if in joint control, delegate on parent
        if self.is_in_joint_control_mode:
            raise NotImplementedError(
                "Denormalization is not implemented with joint controls."
            )

        if relative_action and self.max_position_change is not None:

            return np.concatenate(
                (
                    position_control[:3] * self.max_position_change,
                    np.multiply(position_control[3:], self.speed_per_dof_dim),
                )
            )
        else:
            return position_control

    def set_position_control(self, control: np.ndarray) -> None:

        # if in joint control, directly apply to the position
        if self.is_in_joint_control_mode:
            self.mj_sim.data.qpos[:6] = control
            if self.autostep:
                self.solver.reset()
                self.mj_sim.step()
            return

        # Arm action space is TCP position [x,y,z] + arm wrist angle.
        assert control.shape == (
            len(self.zero_control()),
        ), f"{control} vs {self.zero_control()}"
        pos, angle = np.split(control, (3,))

        angle = self.constrain_quat_ctrl(angle)
        quat_ctrl = self.solver.get_tcp_quat(angle)

        agg_control = np.concatenate([pos, quat_ctrl])
        agg_control = np.array(agg_control, dtype=np.double)
        self.solver.set_action(agg_control)

        if self.autostep:
            self.mj_sim.step()

    def observe(self) -> MujocoObservation:
        return MujocoObservation(self.simulation, self.robot_prefix, self.joint_group)

    def get_joint_state(self) -> np.ndarray:
        return self.observe().joint_positions()

    def sync_to(
        self, joint_positions: np.ndarray, joint_controls: Optional[np.ndarray]
    ):
        """Update this arm to the given position and control. Update from values rather than observations so
        that we can sync from any observation providers.

        :param joint_positions: Arm position.
        :param joint_controls: Arm control target. Currently unused since this arm will be controlled during
        the tick.
        """
        self.simulation.set_qpos(self.joint_group, joint_positions)
        self.mj_sim.forward()

    def reset(self) -> None:
        self.solver.reset()

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        # We use the joint range in xml since there are no actuators for this arm.
        return self.mj_sim.model.jnt_range[:6, 1]

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        # We use the joint range in xml since there are no actuators for this arm.
        return self.mj_sim.model.jnt_range[:6, 0]


class FreeWristTcpArm(FreeDOFTcpArm):
    DOF_DIMS = [PrincipalAxis.PITCH]

    # WARNING: We share here the notation RPY with PrincipalAxis, but current code may use this wrt world axes, which
    # don't necessarily map those of the arm/tcp/gripper.
    # TODO Fix this alignment disparity
    ALIGN_AXIS = PrincipalAxis.PITCH


class FreeRollYawTcpArm(FreeDOFTcpArm):
    """
    TCP with DOF for roll and yaw. This mode currently
    is only supported using the MocapJoint solver.
    """

    DOF_DIMS = [PrincipalAxis.ROLL, PrincipalAxis.PITCH]
