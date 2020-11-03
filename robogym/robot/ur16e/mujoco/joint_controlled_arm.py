from typing import Optional

import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.robot_interface import RobotControlParameters, RobotObservation
from robogym.robot.ur16e.arm_interface import (
    SAFETY_STOP_FORCE_THRESHOLD,
    TABLETOP_EXPERIMENT_INITIAL_POS,
    Arm,
)
from robogym.robot.ur16e.mujoco.simulation.base import ArmSimulationInterface
from robogym.utils import rotation

FINGERTIP_SITE_NAME = "ur_centerplate"
TCP_BODY_NAME = "gripper_tcp"


class MujocoObservation(RobotObservation):
    """ UR Arm observation coming from the MuJoCo simulation """

    def __init__(
        self, simulation: SimulationInterface, hand_prefix: str, joint_group: str
    ):
        sim = simulation.mj_sim
        self._joint_positions = simulation.get_qpos(joint_group).copy()
        self._joint_controls = sim.data.ctrl[:6]
        self._time = sim.data.time
        self._joint_velocities = simulation.get_qvel(joint_group).copy()

        self._tcp_xyz = sim.data.get_body_xpos(f"{hand_prefix}{TCP_BODY_NAME}").copy()
        self._tcp_vel = sim.data.get_body_xvelp(f"{hand_prefix}{TCP_BODY_NAME}").copy()
        quat = sim.data.get_body_xquat(f"{hand_prefix}{TCP_BODY_NAME}").copy()
        self._tcp_rot = rotation.quat2euler(quat)
        sensordata = sim.data.sensordata.copy()
        force_sensor_id = simulation.mj_sim.model.sensor_name2id("toolhead_force")
        sensor_adr = sim.model.sensor_adr[force_sensor_id]
        sensor_dim = sim.model.sensor_dim[force_sensor_id]

        self._tcp_force = sensordata[sensor_adr: sensor_adr + sensor_dim]

        torque_sensor_id = simulation.mj_sim.model.sensor_name2id("toolhead_torque")
        sensor_adr = sim.model.sensor_adr[torque_sensor_id]
        sensor_dim = sim.model.sensor_dim[torque_sensor_id]
        self._tcp_torque = sensordata[sensor_adr: sensor_adr + sensor_dim]

    def timestamp(self):
        return self._time

    def joint_positions(self) -> np.ndarray:
        return self._joint_positions

    def joint_controls(self) -> np.ndarray:
        return self._joint_controls

    def joint_velocities(self) -> np.ndarray:
        return self._joint_velocities

    def tcp_xyz(self) -> np.ndarray:
        """ :return: Tooltip position in the Cartesian coordinate space."""
        return self._tcp_xyz

    def tcp_vel(self) -> np.ndarray:
        """ :return: Tooltip velocity in the Cartesian coordinate space."""
        return self._tcp_vel

    def tcp_rot(self) -> np.ndarray:
        """Rotation in euler angles."""
        return self._tcp_rot

    def tcp_force(self) -> np.ndarray:
        """:return: TCP force in world coordinates."""
        return self._tcp_force

    def tcp_torque(self) -> np.ndarray:
        """:return: TCP torque in world coordinates."""
        return self._tcp_torque

    def is_in_safety_stop(self) -> bool:
        """Returns always false since mujoco arms don't currently have safety stops.

        :return: True if the arm is in a safety stop, False otherwise.
        """
        tcp_force = self.tcp_force()
        return np.linalg.norm(tcp_force) > SAFETY_STOP_FORCE_THRESHOLD


class JointControlledArm(Arm):
    """
    Mujoco implementation of a joint-actuated UR 16e arm.
    """

    def __init__(
        self,
        simulation: ArmSimulationInterface,
        robot_control_params: RobotControlParameters,
        robot_prefix="robot0:",
        autostep=False,
        **kwargs,
    ):
        """
        :param simulation: simulation interface for the mujoco UR arm
        :param robot_control_params: Robot control parameters
        :param robot_prefix: prefix to add to the joint names while constructing the mujoco simulation
        :param autostep: when true, calls step() on the simulation whenever a control is set. this
        should only be used only when the Robot is being controlled without a
        simulationrunner in the loop.
        """
        self.simulation = simulation
        self.robot_prefix = robot_prefix
        self.autostep = autostep
        self.joint_group = robot_prefix + "arm_joint_angles"
        self.simulation.register_joint_group(
            self.joint_group, prefix=robot_prefix + "J"
        )
        self.set_simulation_start_position(TABLETOP_EXPERIMENT_INITIAL_POS)

        self._max_position_change = robot_control_params.max_position_change

    # Note: This could be automatically set for all simulated robots using keyframes from the xml, but it would
    # still require the Simulation to know which keyframe to apply (eg: default, experiment, phys_check, ..)
    # See http://www.mujoco.org/book/XMLreference.html#keyframe
    def set_simulation_start_position(self, pos: np.ndarray) -> None:
        """Sets simulation qpos and ctrl. Intended to be used as a way to set the initial position for the
        arm in the mujoco simulation.

        :param pos: Joint positions, also set in the controls.
        """
        self.simulation.set_qpos(self.joint_group, pos)
        self.mj_sim.data.ctrl[
            :6
        ] = pos  # otherwise the robot will try to go to the last control set if not commanded

    def get_control_time_delta(self) -> float:
        """Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.

        :return: Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.
        """
        dt = self.mj_sim.nsubsteps * self.mj_sim.model.opt.timestep
        return dt

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

    def get_name(self) -> str:
        """Returns a name, expected to uniquely identify this robot within our environments. Examples would be: ALPHA,
        EPSILON, etc. for hands, and UR16e-ONE for UR16 arms."""
        return "mujoco-ur16e"  # note that all mujoco arms will have the same name

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        # We use control range in xml instead of constants to take into account
        # ADR randomization for joint limit.
        return self.mj_sim.model.actuator_ctrlrange[:6, 1]

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        # We use control range in xml instead of constants to take into account
        # ADR randomization for joint limit.
        return self.mj_sim.model.actuator_ctrlrange[:6, 0]

    @property
    def mj_sim(self):
        """ MuJoCo MjSim simulation object """
        return self.simulation.mj_sim

    @property
    def max_position_change(self):
        return self._max_position_change

    def set_position_control(self, control: np.ndarray) -> None:
        self.mj_sim.data.ctrl[:6] = control

        if self.autostep:
            self.mj_sim.step()

    def observe(self) -> MujocoObservation:
        return MujocoObservation(self.simulation, self.robot_prefix, self.joint_group)

    def reset(self) -> None:
        self.set_simulation_start_position(TABLETOP_EXPERIMENT_INITIAL_POS)
