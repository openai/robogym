"""Provides utilities to command reaching a specified position in the robot, and waiting for its completion."""
from __future__ import annotations

import enum
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from robogym.robot.robot_interface import Robot, RobotObservation
from robogym.robot.utils.measurement_units import MeasurementUnit

logger = logging.getLogger(__name__)


@dataclass
class ReachResult:
    """Final result of a command to reach position."""

    reached: bool  # Whether the destination was reached when the action ended (it can time out)
    robot_stopped: bool  # Whether the robot was considered as stopped (not moving) when the action ended
    desired_joint_positions: np.ndarray  # desired destination
    last_joint_positions: np.ndarray  # last observed positions for all actuators/joints
    last_joint_velocities: np.ndarray  # last observed velocities for all actuators/joints
    position_threshold: float  # threshold to consider positions as reached (against the goal)
    velocity_threshold: float  # threshold to consider velocities as stopped (against 0)

    @staticmethod
    def _error_message_line(name, error, threshold, units: MeasurementUnit):
        """Provides a single line message for an actuator that has failed with the given params."""
        return f"Actuator {name} with error abs({error:.4f}) {units.name} > threshold {threshold:.4f} {units.name}"

    def _failed_actuators(self, robot: Robot):
        """Returns a dictionary of failed actuators, with their name as the key, and the error as value."""
        actuators = robot.actuators()
        assert len(actuators) == len(self.last_joint_positions)

        error_per_failed_actuators = {}
        errors_per_actuator = self.last_joint_positions - self.desired_joint_positions
        for act_idx, error in enumerate(errors_per_actuator):
            if np.abs(error) > self.position_threshold:
                # this actuator failed
                actuator_name = actuators[act_idx]
                error_per_failed_actuators[actuator_name] = error

        return error_per_failed_actuators

    def error_message(
        self, robot: Robot, units: MeasurementUnit = MeasurementUnit.RADIANS
    ):
        """Format a nice error message if position was not reached successfully."""
        failed_acts_dict = self._failed_actuators(robot)

        line_per_actuator = [
            self.__class__._error_message_line(
                name, failed_acts_dict[name], self.position_threshold, units
            )
            for name in robot.actuators()
            if name in failed_acts_dict
        ]

        return (
            "Positions not reached successfully:\n - "
            + "\n - ".join(line_per_actuator)
            + "\n"
            if len(line_per_actuator) > 0
            else ""
        )


@dataclass
class ReachHelperConfigParameters:
    """Configuration parameters that should not change between runs to reach different positions, they are more
    related to the robot that will be reaching the positions, than to the individual attempt.
    See related class ReachHelperRunParameters."""

    reached_position_threshold: float  # position threshold to consider the target as reached
    stopped_velocity_threshold: float  # velocity threshold to consider the robot as stopped
    stopped_stable_time: float  # how long (in seconds) we have to wait stopped to declare speed as stable
    safety_speed_limit: float  # speed that is considered safe, and that will throw an error if exceeded
    minimum_time_to_move: float  # time that the robot can be stopped after the command without thinking it failed


@dataclass
class ReachHelperRunParameters:
    """Parameters that can depend on how far the target position is, or how quickly we want to reach the position.
    See related class ReachHelperConfigParameters."""

    timeout: float  # how long we allow the robot to move at max
    max_speed_per_sec: Union[float, np.ndarray]  # max speed shared or per control


class PlotOutput(enum.Flag):
    """Where the plot output can be sent to."""

    DISABLED = enum.auto()
    SCREEN = enum.auto()
    FILE = enum.auto()
    SCREEN_AND_FILE = SCREEN | FILE

    def is_enabled(self) -> bool:
        return self != PlotOutput.DISABLED


class ReachHelperDebugRecorder:
    """Class to record each delta that is sent to the robot, along with logs or other useful information, for
    debugging purposes."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.timestamps: List[float] = []
        self.commanded: List[np.ndarray] = []
        self.obs_pos: List[np.ndarray] = []
        self.obs_vel: List[np.ndarray] = []
        self.logs: List[str] = []
        self.obs_helper_pos: List[np.ndarray] = []

    def add_log(self, log_entry: str) -> None:
        """Adds the given string as a log entry to this recorder, so that it can be logged later if required.

        :param log_entry: String to add as a log entry to this record.
        """
        self.logs.append(f"{time.time()} [reach_helper] {log_entry}")

    def add_sample(self, timestamp, command, obs_pos, obs_vel, obs_helper_pos) -> None:
        """Add a delta to the collection.

        :param timestamp: Timestamp at which the delta happened.
        :param command: Commanded position.
        :param obs_pos: Observed position from the robot (all actuators).
        :param obs_vel: Observed velocity from the robot (all actuators).
        :param obs_helper_pos: Observed position from the helper robot (all actuators) if any.
        """
        self.timestamps.append(timestamp)
        self.commanded.append(command)
        self.obs_pos.append(obs_pos)
        self.obs_vel.append(obs_vel)
        if obs_helper_pos is not None:
            self.obs_helper_pos.append(obs_helper_pos)

    def dump_logs(self) -> None:
        """Sends all log entries in the instance to the logger."""
        for entry in self.logs:
            logger.info(entry)

    def plot_pos_and_vel_for_actuator(
        self,
        actuator_index: int,
        output: PlotOutput,
        data_unit: MeasurementUnit,
        plot_unit: MeasurementUnit,
        debug_reach_try_id: str,
    ) -> None:
        """Plots the commanded position, and observed positions and velocities for the given actuator over the deltas
        that have been provided by the reach helper to the robot.

        :param actuator_index: Index of the actuator within the collected data to plot.
        :param output: Where to send the plot.
        :param data_unit: Unit the data was collected in.
        :param plot_unit: Unit we want to visualize that data (must be related to data_unit).
        :param debug_reach_try_id: Identifier of the reach helper attempt whose plot we are showing (sets title).
        """
        assert output.is_enabled()

        n_rows = 1
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))

        plt.suptitle(
            f"Try:{debug_reach_try_id} - Actuator:{self.robot.actuators()[actuator_index]}"
        )

        qpos_axis = axes[0]
        qvel_axis = axes[1]

        commanded_plot_data = data_unit.convert_to(
            np.asarray(self.commanded), plot_unit
        )
        obs_pos_plot_data = data_unit.convert_to(np.asarray(self.obs_pos), plot_unit)
        obs_vel_plot_data = data_unit.convert_to(np.asarray(self.obs_vel), plot_unit)
        if len(self.obs_helper_pos) > 0:
            obs_helper_pos_data = data_unit.convert_to(
                np.asarray(self.obs_helper_pos), plot_unit
            )
        else:
            obs_helper_pos_data = None

        # plot cmd, qpos
        size = 3
        qpos_axis.plot(
            self.timestamps,
            commanded_plot_data[:, actuator_index],
            marker=4,
            color="purple",
            ms=size,
            linestyle="--",
            label="commanded",
        )
        qpos_axis.plot(
            self.timestamps,
            obs_pos_plot_data[:, actuator_index],
            marker="+",
            color="blue",
            ms=size,
            label="measured pos",
        )
        if obs_helper_pos_data is not None:
            qpos_axis.plot(
                self.timestamps,
                obs_helper_pos_data[:, actuator_index],
                marker="+",
                color="orange",
                ms=size,
                label="helper pos",
            )

        qpos_axis.legend()
        qpos_axis.grid()

        # plot qvel
        qvel_axis.plot(
            self.timestamps,
            obs_vel_plot_data[:, actuator_index],
            marker="+",
            color="blue",
            ms=size,
            label="measured vel",
        )
        qvel_axis.legend()
        qvel_axis.grid()

        # output: record file
        if (output & PlotOutput.FILE).value != 0:
            folder = "reach_helper_images/"
            os.makedirs(folder, exist_ok=True)
            fig_name = f"reach_helper_{debug_reach_try_id}_jnt_{actuator_index}.png"
            fig_path = os.path.join(folder, fig_name)
            logger.info(f"Note: Reach helper debug saving plot '{fig_path}'")
            plt.savefig(fig_path)


class ReachHelper:
    """This class can be configured to command and keep track of a robot as it moves towards a goal destination."""

    # Useful type definitions
    ObservationList = List[RobotObservation]

    def __init__(self, robot: Robot, config: ReachHelperConfigParameters):
        """
        Constructor.

        :param robot: Robot interface that will perform the movements.
        :param config: Parameters the configure the helper. These parameters are considered to be stable even
            if you reach different positions over the lifetime of the helper.
        """
        self.robot = robot
        self.config = config

        # Current state
        self.start_time: float
        self.robot_stopped_at: float
        self.observation: RobotObservation
        self.reset()

    def reset(self) -> None:
        """Reset internal state of the util, ignoring any previous movements performed."""
        self.observation = self.robot.observe()
        self.start_time = self.observation.timestamp()
        self.robot_stopped_at = -1.0  # not stopped

    def _observe(self, buffer: Optional[ObservationList] = None) -> RobotObservation:
        """Make an observation on the robot, and update internal state, appending it to the buffer if provided.

        :param buffer: List of observations to append the newly observed one.
        :return: A newly observed robot observation.
        """
        self.observation = self.robot.observe()

        if buffer is not None:
            buffer.append(self.observation)

        return self.observation

    def _set_position_control(self, position_control: np.ndarray) -> None:
        """Set position control for the underlying robot.

        :param position_control: Array of desired position for each joint (control) in the robot.
        """
        self.robot.set_position_control(position_control)

    def _is_position_reached(self, position_control: np.ndarray) -> bool:
        """Returns whether joint (control) positions are close to a specified value, with closeness defined by the
         position threshold set in the constructor."""
        obs_control_positions = self.robot.joint_positions_to_control(
            self.observation.joint_positions()
        )
        position_errors = obs_control_positions - position_control
        return (np.abs(position_errors) <= self.config.reached_position_threshold).all()

    def _is_robot_stopped_and_stable(self) -> bool:
        """Returns whether joint velocity has stabilized to a low value, defined by the velocity threshold set in
        the constructor."""
        joint_velocity = self.observation.joint_velocities()
        is_currently_stopped = (
            np.abs(joint_velocity) <= self.config.stopped_velocity_threshold
        ).all()

        # calculate whether stable
        is_stable = False
        if is_currently_stopped:
            # we are stopped, if time of stop not set, set now
            if self.robot_stopped_at < self.start_time:
                self.robot_stopped_at = self.observation.timestamp()

            # check if we have reached stability according to the threshold
            is_stable = (
                self.observation.timestamp() - self.robot_stopped_at
            ) >= self.config.stopped_stable_time
        else:
            # not currently stopped, clear stop time
            self.robot_stopped_at = -1.0

        return is_currently_stopped and is_stable

    def reach(
        self,
        target_position_control: np.ndarray,
        *,
        run_params: ReachHelperRunParameters,
        debug_recorder: Optional[ReachHelperDebugRecorder] = None,
        iteration_callback: Optional[Callable[[], None]] = None,
    ) -> ReachResult:
        """Move the robot to the desired position (control). Block until the robot stops moving or time out occurs.

        :param target_position_control: Target positions (controls) of the robot.
        :param run_params: Parameters that configure this execution of a move. These parameters are considered to be
            potentially different per run, since the target position or the trajectory may have different requirements.
        :param debug_recorder: Optional instance where records will be added so that they can be inspected
            outside this function.
        :param iteration_callback: Optional callback to call on each iteration of the reach helper. This iteration
            timing depends on whether the robot is simulated or real, but should it match the control delta configured
            for physical robots, or the simulation step size for simulated robots.
        """
        # - - - - - - - - - - - - - - - - - - - - - - - -
        # DEBUG OPTIONS:
        # a) set debug_enable_logs to True to get logs printed
        # b) set debug_enable_logs to the desired PlotOutput value to generate plots
        #   b.1) set debug_plots_data_unit to the measurement unit the data is collected at
        #   b.2) set debug_plots_view_unit to the measurement unit you want to visualize the data
        debug_enable_logs = False
        debug_plots_output = PlotOutput.DISABLED
        debug_plots_data_unit = MeasurementUnit.RADIANS
        debug_plots_view_unit = MeasurementUnit.DEGREES
        debug_reach_try_id = str(time.time())
        # - - - - - - - - - - - - - - - - - - - - - - - -

        # Current implementations of robots (eg: ur16e) expect to be controlled regularly, rather than
        # being sent the target position. Although some hardware may support the latter (ur16e's can through move
        # functions, instead of servoJ functions), this helper will try to command with linear interpolation on
        # small deltas.
        delta_secs = self.robot.get_control_time_delta()
        if delta_secs <= 0.0:
            raise RuntimeError(
                "Failed to obtain control delta from robot. Can't reach position."
            )

        self._observe()
        local_start_time = self.observation.timestamp()

        # unfortunately, we can't use zero_control to know the shape of the controls, since UR TCP controlled arms
        # are switching to Joint control for the helper, while zero_control is a class method. Use current control
        # instead of zero_control to expand the speed to the correct shape
        current_control = self.robot.joint_positions_to_control(
            self.observation.joint_positions()
        )

        # compute max speed per control if specified as float, otherwise simply grab
        if isinstance(run_params.max_speed_per_sec, float):
            max_speed_per_sec_per_control = np.full(
                current_control.shape, run_params.max_speed_per_sec
            )
        else:
            max_speed_per_sec_per_control = run_params.max_speed_per_sec

        # check that no speed violates the safety limit
        if np.any(max_speed_per_sec_per_control > self.config.safety_speed_limit):
            raise RuntimeError(
                f"Speed is above limit, aborting reach: "
                f"{max_speed_per_sec_per_control} > {self.config.safety_speed_limit}"
            )

        # calculate the position deltas we will use to feed to the robot smaller offsets
        max_speed_per_delta_per_control = delta_secs * max_speed_per_sec_per_control
        control_distance = target_position_control - current_control
        generated_deltas_list: List[np.ndarray] = []
        max_computed_steps = 0

        controls_that_will_timeout = []

        # we allow each control to move as fast as possible up to its max speed.
        for control_idx in range(len(target_position_control)):

            # calculate how long it takes for this control to reach its destination
            distance_i = np.abs(control_distance[control_idx])
            max_speed_i = max_speed_per_delta_per_control[control_idx]

            # if there's only one step, linspace will generate as output only the current value, not the target. Make
            # sure that we at least get 2 steps so that we have the target in the output too. So although the formula
            # would be:
            #   steps_at_max_speed_i = int(distance_i / max_speed_i) + 1
            # add 2 instead of 1. This covers the case of the distance being smaller than the max speed per delta,
            # which would yield 0+1 steps needed, thus not moving. Not moving is an issue when the distance between
            # the current delta and the target is smaller than the max speed, but larger than the reach threshold.
            steps_at_max_speed_i = int(distance_i / max_speed_i) + 2

            # generate the curve for this control
            position_deltas_i = np.linspace(
                current_control[control_idx],
                target_position_control[control_idx],
                steps_at_max_speed_i,
            )
            generated_deltas_list.append(position_deltas_i)

            # record the max because that's the length that we need all arrays to be expanded to
            if steps_at_max_speed_i > max_computed_steps:
                max_computed_steps = steps_at_max_speed_i

            # if we know that a control is not fast enough (it will timeout), flag now
            required_time = (
                steps_at_max_speed_i * delta_secs
            ) + self.config.stopped_stable_time
            if required_time > run_params.timeout:
                msg = f"{control_idx} | distance:{distance_i}, speed:{max_speed_i}/sec, requiredTime:{required_time}sec"
                controls_that_will_timeout.append(msg)

        # if we know that at least one control will not reach the target, fail now before trying
        if len(controls_that_will_timeout) > 0:
            raise RuntimeError(
                f"Some controls won't reach their target before the timeout {run_params.timeout} sec: "
                f"\nControlIdx:{controls_that_will_timeout}, with "
                f"\nTarget  : {target_position_control}"
                f"\nCurrent : {current_control}"
                f"\nMaxSpeed: [setting ] {run_params.max_speed_per_sec}"
                f"\nMaxSpeed: [per_ctrl] {max_speed_per_sec_per_control}"
            )

        # pad each control to match the size of the longest control
        generated_deltas_padded = []
        for delta_i in generated_deltas_list:
            pad = max_computed_steps - len(delta_i)
            if pad > 0:
                padded_delta_i = np.concatenate(
                    (delta_i, np.full((pad,), delta_i[-1], dtype=delta_i.dtype))
                )
                generated_deltas_padded.append(padded_delta_i)
            else:
                generated_deltas_padded.append(delta_i)

        # transpose deltas to the correct control shape
        generated_deltas = np.transpose(np.asarray(generated_deltas_padded))
        next_delta = (
            1 if len(generated_deltas) > 1 else 0
        )  # can skip 0, since it's the same as current control

        if debug_recorder is None:
            debug_recorder = ReachHelperDebugRecorder(robot=self.robot)
        next_tick_sec = time.time()
        elapsed_time = 0.0
        robot_autosteps = getattr(self.robot, "autostep", False)
        robot_runs_in_real_time = not robot_autosteps

        while elapsed_time < run_params.timeout:

            def add_debug_log(reason: str) -> None:
                """Mini-helper function to add an entry that will be logged if requested or if reach fails.
                :param reason: Reason why we are logging the entry.
                """
                old_precision = np.get_printoptions()["precision"]
                np.set_printoptions(precision=3, suppress=True)

                obspos = self.robot.joint_positions_to_control(
                    self.observation.joint_positions()
                )
                obsvel = self.observation.joint_velocities()
                distance = target_position_control - obspos
                log_entry = (
                    f"{elapsed_time:.3f}) {reason} | "
                    f"Stopped:{self._is_robot_stopped_and_stable()}, "
                    f"Reached:{self._is_position_reached(target_position_control)}, "
                    f"Delta:{delta_secs} sec, "
                    f"RunRealTime:{robot_runs_in_real_time}"
                    f"\nMaxSpeed:{max_speed_per_delta_per_control}/step | "
                    f"{run_params.max_speed_per_sec}/sec, "
                    f"\nTarget    :{target_position_control}, Distance: {distance}"
                    f"\nNext cmd  :{next_position}"
                    f"\nObsPos    :{obspos}"
                    f"\nObsVel    :{obsvel}"
                    f"\n"
                )
                assert debug_recorder is not None
                debug_recorder.add_log(log_entry)

                np.set_printoptions(precision=old_precision, suppress=False)

            # wait until it's time to control the robot:
            # for physical or remote robots, we wait real time with a resolution of nanoseconds, since the real robot
            # or the remote server both run at real time rates. For mujoco/simulated robots, which don't run at real
            # time rate, we only support auto-stepping ones at the moment (we could also tick the simulation.) Note
            # that since we break down steps based on `self.robot.get_control_time_delta()`, but we tick each frame
            # for auto-stepping robots, for the deltas to match, `self.robot.get_control_time_delta()` should be
            # equal to the time it elapses with one call to `_set_position_control()`. Otherwise we will be sending
            # deltas scoped for the robot delta, at simulation delta rates.
            if robot_runs_in_real_time:
                tick_time_sec = time.time()
                remaining_delta = next_tick_sec - tick_time_sec
                if remaining_delta > 0:
                    time.sleep(remaining_delta)
                next_tick_sec = (
                    next_tick_sec + delta_secs
                )  # update when the next tick will be

            # observe
            self._observe()

            # check if we have actually reached the destination with the robot stopped
            if self._is_robot_stopped_and_stable():
                # if we have not reached the goal, check whether we have moved at all (via minimum_time_to_move)
                is_position_reached = self._is_position_reached(target_position_control)
                has_minimum_time_passed = (
                    elapsed_time >= self.config.minimum_time_to_move
                )
                if is_position_reached or has_minimum_time_passed:
                    # robot is not moving after min, either we reached the destination or robot will not move at all
                    add_debug_log("ExitCond")
                    break

            # send command to move as much as possible during a delta
            next_position = generated_deltas[next_delta]
            next_delta = (
                next_delta
                if next_delta == len(generated_deltas) - 1
                else next_delta + 1
            )
            self._set_position_control(next_position)

            elapsed_time = self.observation.timestamp() - local_start_time

            obs_helper_pos = None
            if hasattr(self.robot, "get_helper_robot"):
                obs_helper_pos = (
                    self.robot.get_helper_robot().observe().joint_positions()
                )  # type:ignore
            add_debug_log("TickEnd")
            debug_recorder.add_sample(
                elapsed_time,
                next_position,
                self.robot.joint_positions_to_control(
                    self.observation.joint_positions()
                ),
                self.observation.joint_velocities(),
                obs_helper_pos=obs_helper_pos,
            )

            # call the iteration callback now
            if iteration_callback is not None:
                iteration_callback()

        # Just format and return the result of reach operation
        observed_control_positions = self.robot.joint_positions_to_control(
            self.observation.joint_positions()
        )
        observed_joint_velocities = self.observation.joint_velocities()
        is_position_reached = self._is_position_reached(target_position_control)
        is_robot_stopped = self._is_robot_stopped_and_stable()

        # if we failed, configure logs and plots so that they get dumped
        if not is_position_reached:
            logger.info(
                "*** ReachHelper did not reach its destination. Will dump logs and plots automatically."
            )
            debug_enable_logs = True
            if not debug_plots_output.is_enabled():
                debug_plots_output = PlotOutput.FILE

        # log if required
        if debug_enable_logs:
            debug_recorder.dump_logs()

        # plot if required
        if debug_plots_output.is_enabled():
            for act_index, act in enumerate(self.robot.actuators()):
                debug_recorder.plot_pos_and_vel_for_actuator(
                    act_index,
                    debug_plots_output,
                    debug_plots_data_unit,
                    debug_plots_view_unit,
                    debug_reach_try_id,
                )

            # output: display all to screen now (if required)
            if (debug_plots_output & PlotOutput.SCREEN).value != 0:
                plt.show()

        return ReachResult(
            reached=is_position_reached,
            robot_stopped=is_robot_stopped,
            desired_joint_positions=target_position_control,
            last_joint_positions=observed_control_positions,
            last_joint_velocities=observed_joint_velocities,
            position_threshold=self.config.reached_position_threshold,
            velocity_threshold=self.config.stopped_velocity_threshold,
        )


def reach_position(
    robot: Robot,
    position_control: np.ndarray,
    *,
    timeout: float = 10.0,
    minimum_time_to_move: float = 2.0,
    speed_units_per_sec: Optional[Union[float, np.ndarray]] = None,
    position_threshold: Optional[float] = None,
    measurement_unit: MeasurementUnit = MeasurementUnit.RADIANS,
    debug_recorder: Optional[ReachHelperDebugRecorder] = None,
    iteration_callback: Optional[Callable[[], None]] = None,
) -> ReachResult:
    """Perform given movement of a hand to desired position.

    :param robot: Robot interface that will perform the movements.
    :param position_control: Target positions for actuators in the robot. Note that the target position needs to
        be in the same units specified by the 'measurement_unit' parameter.
    :param timeout: Maximum time allowed to reach the target positions and stop the robot, before we consider it
        failed. In seconds.
    :param minimum_time_to_move: If we haven't reached target positions, wait at least for this amount of time for
        the joints to start moving, before thinking the robot will never move. In seconds.
    :param speed_units_per_sec: If specified as float or np.array, the speed at which we want to generate deltas to
        move the robot (similar to the expected robot output speed). If not provided, it is set to a sensitive speed
        that depends on the 'measurement_unit' parameter. If specified as float, the speed is shared among all
        controls. If specified as np.array, the shape must match the shape of the controls, so that each speed limit
        applies to one control.
    :param position_threshold: If specified, the threshold in position so that we can claimed that we have reached
        the target position. If not provided, a default one will be set for the measurement unit provided.
    :param measurement_unit: Measurement unit the robot actuators operate in.
    :param debug_recorder: Optional instance where records will be added so that they can be inspected
        outside this function.
    :param iteration_callback: Optional callback to call on each iteration of the reach helper. This iteration
        timing depends on whether the robot is simulated or real, but should it match the control delta configured
        for physical robots, or the simulation step size for simulated robots.

    :return: Result of the reach operation. See class for more details.
    """

    # default parameters depend on measurement unit
    default_config = {
        MeasurementUnit.RADIANS: ReachHelperConfigParameters(
            reached_position_threshold=np.deg2rad(1),
            stopped_velocity_threshold=np.deg2rad(1),
            stopped_stable_time=0.5,
            safety_speed_limit=np.deg2rad(60),
            minimum_time_to_move=0.0,
        ),
        MeasurementUnit.METERS: ReachHelperConfigParameters(
            reached_position_threshold=0.005,
            stopped_velocity_threshold=0.001,
            stopped_stable_time=0.5,
            safety_speed_limit=0.050,
            minimum_time_to_move=0.0,
        ),
    }

    default_run_params = {
        MeasurementUnit.RADIANS: ReachHelperRunParameters(
            timeout=0.0, max_speed_per_sec=np.deg2rad(30)
        ),
        MeasurementUnit.METERS: ReachHelperRunParameters(
            timeout=0.0, max_speed_per_sec=0.025
        ),
    }

    # get default config and override as needed
    cur_config = default_config[measurement_unit]
    cur_config.minimum_time_to_move = minimum_time_to_move
    cur_config.reached_position_threshold = (
        cur_config.reached_position_threshold
        if position_threshold is None
        else position_threshold
    )

    # get default run params and override as needed
    run_params = default_run_params[measurement_unit]
    run_params.timeout = timeout

    # speed is complicated, allow first setting from argument passed to this method. If not set, ask the robot if
    # it has one (since each robot could have a different one), otherwise use the default one that was configured here
    # for the measurement unit
    if speed_units_per_sec is not None:
        run_params.max_speed_per_sec = speed_units_per_sec
    elif hasattr(robot, "get_default_reach_helper_speed"):
        run_params.max_speed_per_sec = robot.get_default_reach_helper_speed()  # type: ignore

    # reach_helper does not support mixed control units at this moment. For example, for TCP controlled robots it
    # would need to move TCP xyz (meters/millimeters) and TCP rot (radians/degrees). For now, identify the issue and
    # hack those robots so that instead the use Joint commands. This will change in the future, but allows
    # requesting a Joint destination for TCP controlled arms
    assert hasattr(robot, "switch_to_joint_control") == hasattr(
        robot, "switch_to_tcp_control"
    )
    if hasattr(robot, "switch_to_joint_control"):
        robot.switch_to_joint_control()  # type: ignore

    try:
        reach_helper = ReachHelper(robot, config=cur_config)
        ret_val = reach_helper.reach(
            target_position_control=position_control,
            run_params=run_params,
            debug_recorder=debug_recorder,
            iteration_callback=iteration_callback,
        )
    finally:
        # restore tcp control mode
        if hasattr(robot, "switch_to_tcp_control"):
            robot.switch_to_tcp_control()  # type: ignore

    return ret_val


def reach_wrist_angle(
    robot: Robot,
    wrist_angle: float,
    Kp: int = 1,
    max_steps=100,
    speed_limit: float = np.deg2rad(30),
    atol=1e-2,
) -> bool:
    """ Helper for a FreeWristTcpArm robot to achieve a desired wrist position.
    Since wrist control is relative and no position controllers exist, this is
    done using a simple proportional control.

    :param robot: a FreeWristTcpArm robot
    :param robot: wrist displacement limit in radians per step
    :param wrist_angle: desired absolute wrist angle
    :param Kp: Proportional gain coefficient
    :param max_steps: Maximum steps for reach
    :param atol: Absolute tolerance for rech result
    :return: Reach state
    """
    step = 0
    ctrl = robot.zero_control()
    error = wrist_angle - robot.observe().joint_positions()[-1]
    reached = np.abs(error) < atol
    while step < max_steps and not reached:
        angle_ctrl = Kp * np.clip(error, -speed_limit, speed_limit)
        ctrl[-1] = angle_ctrl
        robot.set_position_control(ctrl)
        error = wrist_angle - robot.observe().joint_positions()[-1]
        reached = np.abs(error) < atol
        step += 1
    return reached
