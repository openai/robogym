"""Unit tests for Reach Helper."""
import logging
import time
from typing import Union

import numpy as np

from robogym.robot.utils import reach_helper
from robogym.robot.utils.measurement_units import MeasurementUnit
from robogym.robot.utils.reach_helper import ReachHelperDebugRecorder

logger = logging.getLogger(__name__)


def assert_speed_is_ok(
    _debug_recorder: ReachHelperDebugRecorder,
    _expected_speed: Union[float, np.ndarray],
    _speed_limit_threshold: Union[float, np.ndarray],
) -> None:
    """This function inspects the speed samples from the given recorder (for all controls), and asserts whether
    all are within the desired speed limit.

    :param _debug_recorder: Recorder to check velocity samples.
    :param _expected_speed: Speed limit that we set in the reach helper for the command generation.
    :param _speed_limit_threshold: Small threshold that the robot would potentially pass above the commanded
    expected speed, since reaction time will have a catch-up effect on the robot, which may cause speed to
    increase over the commanded speed briefly.
    """
    # prepare speed limit
    actuator_count = len(_debug_recorder.robot.actuators())
    if np.isscalar(_expected_speed):
        _expected_speed = np.full(actuator_count, _expected_speed)
    if np.isscalar(_speed_limit_threshold):
        _speed_limit_threshold = np.full(actuator_count, _speed_limit_threshold)
    speed_limit = _expected_speed + _speed_limit_threshold

    # compare observed vs limit
    max_obs_speed_per_control = np.max(np.abs(_debug_recorder.obs_vel), axis=0)
    limit_ok_per_control = max_obs_speed_per_control < speed_limit
    was_speed_ok = np.alltrue(limit_ok_per_control)

    # assert/print relevant info
    random_id = str(time.time())
    if not was_speed_ok:
        logger.info(
            "Speed limit violation, will dump plots of the samples for debugging:"
        )
        for act_idx in range(len(_debug_recorder.obs_pos[0])):
            _debug_recorder.plot_pos_and_vel_for_actuator(
                act_idx,
                reach_helper.PlotOutput.FILE,
                MeasurementUnit.RADIANS,
                MeasurementUnit.DEGREES,
                f"test_reach_helper_{random_id}",
            )

    assert (
        was_speed_ok
    ), f"Speed limit violation: \n{max_obs_speed_per_control} \nvs \n{speed_limit}"


def _build_reach_helper_test_robot(max_position_change=0.020):
    from gym.envs.robotics import utils

    from robogym.envs.rearrange.simulation.blocks import (
        BlockRearrangeSim,
        BlockRearrangeSimParameters,
    )
    from robogym.robot.robot_interface import ControlMode, RobotControlParameters

    sim = BlockRearrangeSim.build(
        n_substeps=20,
        robot_control_params=RobotControlParameters(
            control_mode=ControlMode.TCP_WRIST.value,
            max_position_change=max_position_change,
        ),
        simulation_params=BlockRearrangeSimParameters(),
    )

    # reset mocap welds if any. This is actually needed for TCP arms to move
    utils.reset_mocap_welds(sim.mj_sim)

    # extract arm since CompositeRobots are not fully supported by reach_helper
    composite_robot = sim.robot
    arm = composite_robot.robots[0]
    arm.autostep = True
    return arm


def test_curve_generation_two_steps() -> None:
    """This test is used to verify a bugfix. The bug was that if a target's distance is too close to the current
     position (closer than the max speed), the curve would only generate one step for the actuator, and the step
     would be for the current position, not for the target position. Bugfix: reach helper should generate at least
     two steps.
    """
    robot = _build_reach_helper_test_robot()
    cur_pos = robot.observe().joint_positions()

    # calculate the small step that was bugged
    control_delta = robot.get_control_time_delta()
    max_speed = np.deg2rad(60)
    max_change_per_step = max_speed * control_delta
    offset_that_was_bugged = max_change_per_step - np.deg2rad(
        0.01
    )  # offset needs to be below max_change_per_step
    position_threshold = offset_that_was_bugged - np.deg2rad(
        0.01
    )  # threshold needs to be below the offset
    assert position_threshold < offset_that_was_bugged

    target_pos = cur_pos.copy()
    target_pos[0] += offset_that_was_bugged

    ret_i = reach_helper.reach_position(
        robot,
        target_pos,
        speed_units_per_sec=max_speed,
        position_threshold=position_threshold,
    )
    assert ret_i.reached
