import numpy as np

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.gripper.mujoco.regrasp_helper import RegraspHelper
from robogym.robot.robot_interface import (
    Robot,
    RobotControlParameters,
    RobotObservation,
)


class MujocoObservation(RobotObservation):
    """ UR Arm observation coming from the MuJoCo simulation """

    def __init__(
        self, simulation: SimulationInterface, joint_group: str, actuator_id: int,
    ):
        self._time = simulation.mj_sim.data.time

        sim = simulation.mj_sim
        self._joint_positions = simulation.get_qpos(joint_group).copy()
        self._joint_controls = sim.data.ctrl[actuator_id: actuator_id + 1].copy()
        self._joint_vel = simulation.get_qvel(joint_group).copy()

    def joint_positions(self) -> np.ndarray:
        return self._joint_positions

    def joint_controls(self) -> np.ndarray:
        return self._joint_controls

    def joint_velocities(self) -> np.ndarray:
        return self._joint_vel

    def timestamp(self) -> float:
        return self._time


class MujocoRobotiqGripper(Robot):
    """
    Mujoco implementation of a 1-DOF RobotIQ 2f-85 gripper.
    """

    ACTUATORS = ["A_J1"]
    JOINTS = ["r_gripper_RJ0_outer"]

    def __init__(
        self,
        simulation: SimulationInterface,
        robot_control_params: RobotControlParameters,
        robot_prefix="robot0:",
        autostep=False,
        **kwargs
    ):
        """
        :param simulation: simulation interface for the mujoco shadow hand xml
        :param hand_prefix: prefix to add to the joint names while constructing the mujoco simulation
        :param autostep: when true, calls step() on the simulation whenever a control is set. this
        should only be used only when the Robot is being controlled without a
        simulationrunner in the loop.
        """

        self.simulation = simulation
        self.robot_prefix = robot_prefix
        self.autostep = autostep
        self.joint_group = robot_prefix + "gripper_joint_angles"
        self.simulation.register_joint_group(
            self.joint_group, prefix=[robot_prefix + "r_gripper_RJ0_outer"]
        )

        # assert max_position_change > 0.0, f"Position multiplier must be a positive number"
        # For now, we do not constrain gripper motion.
        self._max_position_change = None
        self.actuator_id = self.mj_sim.model.actuator_name2id(
            robot_prefix + "r_gripper_finger_joint"
        )

        # set a consistent initial control to prevent drifting at the start of the simulation
        # otherwise the robot will try to go to the last control set, which may not be the same as the position)
        # Since qpos is not set directly for gripper, simply set the ctrl to whatever position we have now
        self.mj_sim.data.ctrl[self.actuator_id] = simulation.get_qpos(
            self.joint_group
        ).copy()

        self.regrasp_helper = None
        if robot_control_params.enable_gripper_regrasp:
            self.regrasp_helper = RegraspHelper(
                initial_position=self.observe().joint_positions()
            )

    def get_control_time_delta(self) -> float:
        """Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.

        :return: Returns the delta that the robot wants to be controlled under, by matching it with its mujoco simulation
        step delta.
        """
        dt = self.mj_sim.nsubsteps * self.mj_sim.model.opt.timestep
        return dt

    def get_name(self) -> str:
        return "robotiq-2f-85"

    @property
    def max_position_change(self):
        return self._max_position_change

    @classmethod
    def actuators(cls) -> np.ndarray:
        return np.asarray(cls.ACTUATORS)

    @classmethod
    def joints(cls) -> np.ndarray:
        return np.asarray(cls.JOINTS)

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        # We use control range in xml instead of constants to take into account
        # ADR randomization for joint limit.
        return np.asarray([self.mj_sim.model.actuator_ctrlrange[self.actuator_id, 1]])

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        # We use control range in xml instead of constants to take into account
        # ADR randomization for joint limit.
        return np.asarray([self.mj_sim.model.actuator_ctrlrange[self.actuator_id, 0]])

    @classmethod
    def joint_positions_to_control(cls, joint_pos: np.ndarray) -> np.ndarray:
        # Only one joint is actuated among the six
        return joint_pos

    @property
    def mj_sim(self):
        """ MuJoCo MjSim simulation object """
        return self.simulation.mj_sim

    @property
    def regrasp_enabled(self):
        return self.regrasp_helper is not None

    def get_current_position(self):
        return self.mj_sim.data.ctrl[self.actuator_id]

    def denormalize_position_control(
        self, position_control: np.ndarray, relative_action: bool = False,
    ) -> np.ndarray:
        """
        Transform position control from the [-1,+1] range into supported
        joint position range in radians, where 0 represents the midpoint between
        two extreme positions.

        NOTE: Overriden in MujocoRobotiqGripper to add re-grasping capabilities. Re-grasping means that when
        the gripper detects backdrive due to neutral commands, it will try itself to re-issue a previous command
        that is deemed as more desirable than the backdrive result.

        :param position_control: The normalized position control from [-1,+1] range.
        :param relative_action: If true, map current joint position to center of the
               normalized position control which means hand will not move if position_control
               is zero. Otherwise map center of actuator control range to center of the
               normalized position control.
        """
        default_control = super().denormalize_position_control(
            position_control=position_control, relative_action=relative_action
        )
        # absolute control does not need re-grasp
        if not relative_action or not self.regrasp_enabled:
            return default_control

        assert self.regrasp_helper
        return self.regrasp_helper.compute_regrasp_control(
            position_control=position_control,
            default_control=default_control,
            current_position=self.observe().joint_positions(),
        )

    def sync_to(self, joint_positions: np.ndarray, joint_controls: np.ndarray) -> None:
        """Update this gripper to the given position and control. Update from values rather than observations so
        that we can sync from any observation providers.

        :param joint_positions: Gripper position (currently an array of 1 float linear displacement.)
        :param joint_controls: Gripper control target (currently an array of 1 float linear displacement). Useful
        if this gripper is not controlled during the step.
        """
        self.simulation.set_qpos(self.joint_group, joint_positions)
        self.set_position_control(joint_controls)

    def set_position_control(self, control: np.ndarray) -> None:
        self.mj_sim.data.ctrl[self.actuator_id] = control[0]

        if self.autostep:
            self.mj_sim.step()

    def observe(self) -> MujocoObservation:
        return MujocoObservation(self.simulation, self.joint_group, self.actuator_id)
