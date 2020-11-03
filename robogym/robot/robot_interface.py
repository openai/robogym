import abc
from enum import Enum
from typing import Any, Dict, Optional

import attr
import numpy as np


class ControlMode(Enum):
    """
    Control mode. Defines the action space of a particular robot.
    """

    TCP_WRIST = "tcp+wrist"  # TCP control with wrist rotation
    TCP_ROLL_YAW = "tcp+roll+yaw"  # TCP control with roll and yaw rotation enabled.
    JOINT = "joint"  # Joint control

    def is_tcp_controlled(self):
        return self in [ControlMode.TCP_WRIST, ControlMode.TCP_ROLL_YAW]


class TcpSolverMode(Enum):
    """
    Mode of actuation used. This does not have to match control_mode as long as there is
    a known transformation from one control mode to another actuation mode.
    """

    MOCAP = "mocap"
    MOCAP_IK = "mocap_ik"


@attr.s(auto_attribs=True)
class RobotControlParameters:
    """ Robot control parameters – defined as parameters since max_position_change can be
    subject to ADR randomization """

    MOCAP_DEFAULT_MAX_POSITION_CHANGE = 0.05
    MOCAP_RESET_DEFAULT_MAX_POSITION_CHANGE = 0.1
    JOINT_CONTROL_DEFAULT_MAX_POSITION_CHANGE = 2.4
    # robot control mode. Supported modes are
    # joint: Joint position control
    # tcp: Tool center point control
    control_mode: ControlMode = attr.ib(
        default=ControlMode.TCP_ROLL_YAW,
        validator=attr.validators.in_(ControlMode),
        converter=ControlMode,
    )
    # hard constraints on maximum position change per step on a given action dimension.
    # for joint control, it limits position change per joint
    # for TCP control, it is interpreted as a multiplier
    # on the commanded absolute tool position and rotation
    max_position_change: Optional[float] = None

    tcp_solver_mode: TcpSolverMode = attr.ib(
        default=TcpSolverMode.MOCAP_IK,
        validator=attr.validators.in_(TcpSolverMode),
        converter=TcpSolverMode,
    )

    # subdirectory of the joint calibration path to load while making the arm xml
    arm_joint_calibration_path: str = attr.ib(
        default="cascaded_pi", validator=attr.validators.in_(["cascaded_pi", "pid"])
    )

    # when set, resets controller error on every action between main and controller simulations
    arm_reset_controller_error: bool = True

    # whether or not to use the force limiter
    use_force_limiter: bool = True

    # when set, enables regrasp logic for gripper
    enable_gripper_regrasp: bool = False

    def is_joint_actuated(self):
        return (
            self.control_mode is ControlMode.JOINT
            or self.tcp_solver_mode is TcpSolverMode.MOCAP_IK
        )

    def is_tcp_controlled(self):
        return self.control_mode.is_tcp_controlled()

    def requires_solver_sim(self):
        """
        If a robot is joint actuated but TCP controlled, we need to build a solver
        simulation that will perform the conversion from the TCP action space to the joint
        actuation space.
        :return:
        """
        return self.is_joint_actuated() and self.is_tcp_controlled()

    def get_controller_arm_solver_mode(self):
        """
        For mocap_ik mode, we need to specify that the controller arm will be using
        mocap. For now, we dynamically make the correction here.
        :return:
        """
        if self.tcp_solver_mode is TcpSolverMode.MOCAP_IK:
            return TcpSolverMode.MOCAP
        return self.tcp_solver_mode

    @staticmethod
    def default_max_pos_change_for_solver(
        *,
        control_mode: ControlMode,
        tcp_solver_mode: TcpSolverMode,
        arm_reset_controller_error: bool = True
    ) -> float:
        """
        Returns a default max position change scaling that is appropriate for the selected control
        parameters.
        :return: A recommended value for max_pos_change
        """
        if control_mode is ControlMode.JOINT:
            return (
                RobotControlParameters.JOINT_CONTROL_DEFAULT_MAX_POSITION_CHANGE
            )  # per joint
        elif tcp_solver_mode is TcpSolverMode.MOCAP:
            return RobotControlParameters.MOCAP_DEFAULT_MAX_POSITION_CHANGE
        elif tcp_solver_mode is TcpSolverMode.MOCAP_IK:
            if arm_reset_controller_error:
                return RobotControlParameters.MOCAP_RESET_DEFAULT_MAX_POSITION_CHANGE
            else:
                return RobotControlParameters.MOCAP_DEFAULT_MAX_POSITION_CHANGE
        else:
            raise ValueError(
                "No default is defined for the given parameter combination."
            )


class RobotObservation(abc.ABC):
    """
    Interface for the single observation of the shadow hand. Observation object owns and manages
    a set of data that constitutes an "observation", and exposes a number of methods to extract
    commonly used fields (like joint positions or actuator effort).


    All fields in a single observation object come as much as possible from a single moment in time
    (as much as the hardware allows, as different sensors may have different sampling rates).
    """

    @abc.abstractmethod
    def joint_positions(self) -> np.ndarray:
        """
        Return observed joint angles (in rad)

        :returns array of joint angles (one for each joint)
        """
        pass

    @abc.abstractmethod
    def joint_velocities(self) -> np.ndarray:
        """
        Return observed joint velocities (in rad/s)

        :returns array of joint velocities (one for each joint)
        """
        pass

    @abc.abstractmethod
    def timestamp(self) -> float:
        """
        Time in seconds since some unspecified event in the past.
        Useful for comparing time passed between observations.
        """
        pass


class Robot(abc.ABC):
    """
    High level API for controlling a general robot.
    """

    @classmethod
    @abc.abstractmethod
    def actuators(cls) -> np.ndarray:
        """
        Return an array containing the names the actuators of the robot.
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        """Returns a name, expected to uniquely identify this robot within our environments. Examples would be: ALPHA,
        EPSILON, etc. for hands, and UR16e-ONE for UR16 arms."""
        pass

    @abc.abstractmethod
    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        """
        Returns the upper bound of actuator control range in radians.
        """
        pass

    @abc.abstractmethod
    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        """
        Returns the lower bound of actuator control range in radians.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def joint_positions_to_control(cls, joint_pos: np.ndarray) -> np.ndarray:
        """
        Transform a position into a control input
        :param joint_pos:
        :return: control array
        """
        pass

    @property
    def max_position_change(self):
        """
        Maximum allowable position change per step
        :return:
        """
        pass

    def actuation_range(self, relative_action: bool) -> np.ndarray:
        """
        Returns the an array of actuation ranges per joint. By default, it returns the
        range defined by the upper and lower bounds of the control ranges.
        """
        base_range = (
            self.actuator_ctrl_range_upper_bound()
            - self.actuator_ctrl_range_lower_bound()
        ) / 2.0
        if relative_action and self.max_position_change:
            return np.minimum(base_range, self.max_position_change)
        return base_range

    def is_position_control_valid(self, control: np.ndarray) -> bool:
        """
        Check if position control vector is within (inclusive) supported ranges and
        of correct shape.
        """
        return (
            control.shape == self.actuator_ctrl_range_upper_bound().shape
            and np.all(control >= self.actuator_ctrl_range_lower_bound() - 1e-6)
            and np.all(control <= self.actuator_ctrl_range_upper_bound() + 1e-6)
        )

    def get_current_position(self) -> np.ndarray:
        return self.observe().joint_positions()

    def denormalize_position_control(
        self, position_control: np.ndarray, relative_action: bool = False,
    ) -> np.ndarray:
        """
        Transform position control from the [-1,+1] range into supported
        joint position range in radians, where 0 represents the midpoint between
        two extreme positions.

        :param position_control: The normalized position control from [-1,+1] range.
        :param relative_action: If true, map current joint position to center of the
               normalized position control which means hand will not move if position_control
               is zero. Otherwise map center of actuator control range to center of the
               normalized position control.
        """

        if relative_action:
            joint_positions = self.get_current_position()
            actuation_center = self.joint_positions_to_control(joint_positions)
        else:
            actuation_center = (
                self.actuator_ctrl_range_upper_bound()
                + self.actuator_ctrl_range_lower_bound()
            ) / 2.0

        ctrl = actuation_center + position_control * self.actuation_range(
            relative_action
        )
        return np.clip(
            ctrl,
            self.actuator_ctrl_range_lower_bound(),
            self.actuator_ctrl_range_upper_bound(),
        )

    def zero_control(self) -> np.ndarray:
        """
        Return an array representing a zero actuator control vector (in rad).
        Zero positions represent a flat, straightened out hand.
        """
        return np.zeros(len(self.actuators()), dtype=np.float)

    def get_control_time_delta(self) -> float:
        """Returns the time slice the robot desires to be controlled under. This is indicative of the frequency that
        a robot wants to be controlled under.

        :return: The time slice the robot desires to be controlled under.
        """
        return 0.0

    ##############################################################################################
    # ROBOT INTERFACE ABSTRACT METHODS
    @abc.abstractmethod
    def set_position_control(self, control: np.ndarray) -> None:
        """
        Set actuator position control vector. Each coordinate of this vector is the desired
        joint angle (or sum of joint angles for coupled joints). Internal robot controller
        then chooses the right force to achieve that position.

        Both control modes are mutually exclusive and turning on one turns off the other.

        :param control: 20-element array of actuator control (target joint angles in radians)
        """
        pass

    @abc.abstractmethod
    def observe(self) -> RobotObservation:
        """
        Return the "observation" object which contains all the most recent,
        contemporaneous observations of the state of the robotic hand.
        """
        pass

    def on_observations_updated(self, new_observations: Dict[str, Any]) -> None:
        """Event to notify the robot that new observations have been collected. This should be expected to happen
        once at the start of the tick, so that all robots can synchronize their simulations accordingly, specially
        with respect to real observations. If required, robots could also cache these observations if they
        needed them during the tick, this preventing them from calling observe(), which might return (for real
        robots) a new observation mid-tick.

        :param new_observations: New observations collected.
        """
        pass

    @classmethod
    def joints(cls) -> np.ndarray:
        """
        Return the joint space of the robot. Defaults to the actuator space by default.
        """
        return cls.actuators()

    def reset(self) -> None:
        """
        Reset robot state.
        :return:
        """
        pass
