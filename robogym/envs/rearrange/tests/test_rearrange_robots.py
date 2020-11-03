from typing import Optional

import numpy as np
import pytest
from gym.envs.robotics import utils

from robogym.envs.rearrange.simulation.blocks import (
    BlockRearrangeSim,
    BlockRearrangeSimParameters,
)
from robogym.robot.control.tcp.solver import PrincipalAxis
from robogym.robot.robot_interface import (
    ControlMode,
    RobotControlParameters,
    TcpSolverMode,
)
from robogym.robot.ur16e.arm_interface import TABLETOP_EXPERIMENT_INITIAL_POS, Arm
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import DOF_DIM_SPEED_SCALE
from robogym.robot.utils import reach_helper
from robogym.robot.utils.reach_helper import ReachHelperDebugRecorder, reach_wrist_angle
from robogym.robot.utils.tests.test_reach_helper import assert_speed_is_ok


def _build_robot(control_mode, max_position_change):
    sim = BlockRearrangeSim.build(
        n_substeps=1,
        robot_control_params=RobotControlParameters(
            control_mode=control_mode, max_position_change=max_position_change,
        ),
        simulation_params=BlockRearrangeSimParameters(),
    )

    # reset mocap welds if any. This is actually needed for TCP arms to move, so other tests that did not use
    # env may benefit from it.
    utils.reset_mocap_welds(sim.mj_sim)

    return sim.robot


ANGLE_SCALER = np.array(
    [DOF_DIM_SPEED_SCALE[PrincipalAxis.ROLL], DOF_DIM_SPEED_SCALE[PrincipalAxis.PITCH]]
)


@pytest.mark.parametrize(
    "max_position_change,normalized_control,denormalized_pos,denormalized_angle,denormalized_gripper_ctrl",
    [
        (1.0, np.ones(7), [1.0, 1.0, 1.0], ANGLE_SCALER, [0]),
        (0.05, np.ones(7), [0.05, 0.05, 0.05], ANGLE_SCALER * 0.05, [0]),
        (1.0, -np.ones(7), [-1.0, -1.0, -1.0], ANGLE_SCALER * -1, [-0.022365]),
        (0.05, -np.ones(7), [-0.05, -0.05, -0.05], ANGLE_SCALER * -0.05, [-0.022365]),
    ],
)
def test_tcp_arm_denormalization_ur16(
    max_position_change,
    normalized_control,
    denormalized_pos,
    denormalized_angle,
    denormalized_gripper_ctrl,
):
    """
    Test for the composite robot where the first 5 dimensions of the action space come from
    a TCP controlled arm, (3 entries for position, 2 for angle) and the last dimension comes from
    a position-controlled gripper.
    :return:
    """
    robot = _build_robot(ControlMode.TCP_ROLL_YAW.value, max_position_change)
    # important to make sure gripper starts at a neutral state so actions don't get clipped
    assert np.array_equal(robot.observe().gripper_qpos(), np.zeros(1))

    expected_full_ctrl = np.concatenate(
        (denormalized_pos, denormalized_angle, denormalized_gripper_ctrl)
    )

    assert np.allclose(
        robot.denormalize_position_control(
            position_control=normalized_control, relative_action=True
        ),
        expected_full_ctrl,
    )


@pytest.mark.parametrize(
    "control_mode,expected_dim", [(ControlMode.TCP_ROLL_YAW, 6), (ControlMode.JOINT, 6)]
)
def test_joint_position_observations(control_mode, expected_dim):
    robot = _build_robot(control_mode, 0.05)
    assert len(robot.observe().joint_positions()) == expected_dim


def test_gripper_observations():
    robot = _build_robot(ControlMode.TCP_ROLL_YAW.value, 0.05)
    obs = robot.observe()
    assert len(obs.gripper_qpos()) == 1
    assert len(obs.gripper_vel()) == 1
    assert len(obs.tcp_xyz()) == 3
    assert len(obs.tcp_vel()) == 3


def test_action_space_dims():
    robot = _build_robot(ControlMode.TCP_ROLL_YAW.value, 0.05)
    assert len(robot.zero_control()) == 6


def test_joint_positions_to_control():
    robot = _build_robot(ControlMode.JOINT.value, 0.05)
    assert np.array_equal(robot.joint_positions_to_control(np.zeros(12)), np.zeros(7))


@pytest.mark.parametrize(
    "control_mode,tcp_solver_mode,expected_joint_count",
    [
        (ControlMode.JOINT, TcpSolverMode.MOCAP, 7),
        (ControlMode.TCP_WRIST, TcpSolverMode.MOCAP, 1),
        (ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP, 1),
        (ControlMode.TCP_WRIST, TcpSolverMode.MOCAP_IK, 7),
        (ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP_IK, 7),
    ],
)
def test_arm_actuators(control_mode, tcp_solver_mode, expected_joint_count):
    from robogym.envs.rearrange.blocks_reach import make_env

    env = make_env(
        parameters=dict(
            robot_control_params=dict(
                control_mode=control_mode, tcp_solver_mode=tcp_solver_mode,
            )
        )
    )
    assert len(env.sim.model.actuator_names) == expected_joint_count


@pytest.mark.parametrize(
    "control_mode,tcp_solver_mode",
    [
        (ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP),
        (ControlMode.TCP_WRIST, TcpSolverMode.MOCAP),
        (ControlMode.TCP_ROLL_YAW, TcpSolverMode.MOCAP_IK),
        (ControlMode.TCP_WRIST, TcpSolverMode.MOCAP_IK),
    ],
)
def test_free_wrist_reach(control_mode, tcp_solver_mode):
    from robogym.envs.rearrange.blocks import make_env

    max_position_change = RobotControlParameters.default_max_pos_change_for_solver(
        control_mode=control_mode,
        tcp_solver_mode=tcp_solver_mode,
        arm_reset_controller_error=True,
    )
    env = make_env(
        starting_seed=1,
        parameters=dict(
            robot_control_params=dict(
                control_mode=control_mode,
                tcp_solver_mode=tcp_solver_mode,
                max_position_change=max_position_change,
                arm_reset_controller_error=True,
            ),
            n_random_initial_steps=10,
        ),
    )

    arm = env.robot.robots[0]
    j6_lower_bound = arm.actuator_ctrl_range_lower_bound()[5]
    j6_upper_bound = arm.actuator_ctrl_range_upper_bound()[5]
    reach_buffer = np.deg2rad(5)

    def _reset_and_get_arm(env):
        env.reset()
        arm = env.robot.robots[0]
        arm.autostep = True
        return arm

    for target_angle in np.linspace(
        j6_lower_bound + reach_buffer, j6_upper_bound - reach_buffer, 12
    ):
        arm = _reset_and_get_arm(env)
        assert reach_wrist_angle(
            robot=arm, wrist_angle=target_angle, max_steps=200, Kp=2,
        ), (
            f"Failed reach. desired pos: {np.rad2deg(target_angle)} "
            f"achieved pos: {np.rad2deg(arm.observe().joint_positions()[-1]):.2f}"
        )

    for unreachable_angle in [
        j6_lower_bound - reach_buffer,
        j6_upper_bound + reach_buffer,
    ]:
        arm = _reset_and_get_arm(env)
        assert not reach_wrist_angle(
            robot=arm, wrist_angle=unreachable_angle
        ), f"Reached unreachable angle: {np.rad2deg(unreachable_angle)}"


@pytest.mark.parametrize(
    "control_mode,tcp_solver_mode,max_position_change, expected_controls",
    [
        (
            ControlMode.TCP_WRIST,
            TcpSolverMode.MOCAP,
            0.05,
            [0.05, 0.05, 0.05, np.deg2rad(30)],
        ),
        (
            ControlMode.TCP_WRIST,
            TcpSolverMode.MOCAP_IK,
            0.27,
            [0.27, 0.27, 0.27, np.deg2rad(162)],
        ),
        (
            ControlMode.TCP_ROLL_YAW,
            TcpSolverMode.MOCAP,
            0.05,
            [0.05, 0.05, 0.05, np.deg2rad(10), np.deg2rad(30)],
        ),
        (
            ControlMode.TCP_ROLL_YAW,
            TcpSolverMode.MOCAP_IK,
            0.27,
            [0.27, 0.27, 0.27, np.deg2rad(54), np.deg2rad(162)],
        ),
    ],
)
def test_tcp_action_scaling(
    control_mode, tcp_solver_mode, max_position_change, expected_controls
):
    from robogym.envs.rearrange.blocks import make_env

    env = make_env(
        starting_seed=0,
        parameters=dict(
            robot_control_params=dict(
                control_mode=control_mode,
                tcp_solver_mode=tcp_solver_mode,
                max_position_change=max_position_change,
            ),
        ),
    )
    env.reset()
    arm = env.robot.robots[0]
    control = np.ones_like(arm.zero_control())
    denormalized_ctrl = arm.denormalize_position_control(control, relative_action=True)
    assert np.allclose(denormalized_ctrl, expected_controls)


@pytest.mark.parametrize(
    "tcp_solver_mode,arm_joint_calibration_path,off_axis_threshold",
    [
        (TcpSolverMode.MOCAP_IK, "cascaded_pi", 3),
        (TcpSolverMode.MOCAP, "cascaded_pi", 3),
    ],
)
def test_free_wrist_quat_constraint(
    tcp_solver_mode: TcpSolverMode,
    arm_joint_calibration_path: str,
    off_axis_threshold: float,
):
    """
    FreeWristTCP controlled env is only allowed to vary TCP quat about the y axis
    this test checks for max deviation in the other axes when the arm is driven to
    an extreme of the arm's reach so that mocap will deviate from the actual gripper
    pos and may cause angle deviations if we have an alignment bug.
    :return:
    """
    from robogym.envs.rearrange.blocks import make_env
    from robogym.utils import rotation

    max_position_change = RobotControlParameters.default_max_pos_change_for_solver(
        control_mode=ControlMode.TCP_WRIST, tcp_solver_mode=tcp_solver_mode,
    )
    env = make_env(
        starting_seed=3,
        parameters=dict(
            robot_control_params=dict(
                control_mode=ControlMode.TCP_WRIST,
                tcp_solver_mode=tcp_solver_mode,
                max_position_change=max_position_change,
                arm_joint_calibration_path=arm_joint_calibration_path,
            ),
        ),
    )
    env.reset()
    max_off_axis_deviation_deg = 0
    for _ in range(50):
        env.step([6, 10, 5, 5, 5])
        angle = np.rad2deg(
            rotation.quat2euler(
                env.mujoco_simulation.mj_sim.data.get_body_xquat("robot0:gripper_base")
            )
        )
        x_rot = min(angle[0], 90 - angle[0])
        z_rot = min(angle[2], 90 - angle[2])
        max_off_axis_deviation_deg = np.max([max_off_axis_deviation_deg, x_rot, z_rot])
    assert max_off_axis_deviation_deg < off_axis_threshold


@pytest.mark.parametrize(
    "tcp_solver_mode,control_mode,off_joint_threshold_deg",
    [
        (TcpSolverMode.MOCAP_IK, ControlMode.TCP_WRIST, 0.7),
        (TcpSolverMode.MOCAP, ControlMode.TCP_WRIST, 0.7),
        (TcpSolverMode.MOCAP_IK, ControlMode.TCP_ROLL_YAW, 0.7),
        (TcpSolverMode.MOCAP, ControlMode.TCP_ROLL_YAW, 0.7),
    ],
)
def test_wrist_isolation(
    tcp_solver_mode: TcpSolverMode,
    control_mode: ControlMode,
    off_joint_threshold_deg: float,
):
    """
    FreeWristTCP controlled env is only allowed to vary TCP quat about the y axis
    this test checks for max deviation in the other joints than J6 when only wrist
    rotation action is applied.
    :return:
    """
    from robogym.envs.rearrange.blocks import make_env

    max_position_change = RobotControlParameters.default_max_pos_change_for_solver(
        control_mode=control_mode, tcp_solver_mode=tcp_solver_mode,
    )
    env = make_env(
        starting_seed=3,
        parameters=dict(
            n_random_initial_steps=0,
            robot_control_params=dict(
                control_mode=control_mode,
                tcp_solver_mode=tcp_solver_mode,
                max_position_change=max_position_change,
            ),
        ),
    )
    env.reset()
    initial_joint_angles = env.observe()["robot_joint_pos"]
    wrist_rot_action = np.ones(env.action_space.shape[0], dtype=np.int) * 5

    wrist_joint_index = -2

    # move wrist CW
    wrist_rot_action[wrist_joint_index] = 7

    for _ in range(100):
        env.step(wrist_rot_action)

    final_joint_angles = env.observe()["robot_joint_pos"]

    assert np.allclose(
        initial_joint_angles[:5],
        final_joint_angles[:5],
        atol=np.deg2rad(off_joint_threshold_deg),
    )

    # move wrist CCW
    env.reset()
    wrist_rot_action[wrist_joint_index] = 3

    for _ in range(100):
        env.step(wrist_rot_action)

    final_joint_angles = env.observe()["robot_joint_pos"]

    assert np.allclose(
        initial_joint_angles[:5],
        final_joint_angles[:5],
        atol=np.deg2rad(off_joint_threshold_deg),
    )


@pytest.mark.parametrize(
    "control_mode, max_position_change, speed_limit_threshold_mult",
    [
        (
            ControlMode.JOINT.value,
            None,
            10 / 30,
        ),  # mult=10/30 means '10 degrees when speed is 30 deg/sc'
        (ControlMode.TCP_ROLL_YAW.value, 0.025, None),
    ],
)
def test_reach_helper(
    control_mode,
    max_position_change: float,
    speed_limit_threshold_mult: Optional[float],
) -> None:
    """Test that a given robot can reach certain positions, and fail to reach some under certain speed
    requirements.

    :param control_mode: Robot control mode.
    :param max_position_change: Max position change as expected by UR arms.
    :param speed_limit_threshold_mult: If not None, the test will check that the actual speeds achieved during the
    reach efforts do not exceed the expected speed plus a little margin, as defined by this multiplier. If None, the
    speed check is not performed (TCP arm does not currently properly report speeds, since it doesn't have the proper
    actuators to perform dynamics, so it's a case where we specify None).
    """

    # these positions are 4 corners recorded empirically. All of them should be safely reachable in the order
    # specified, but might not be if the order changes
    positions4 = [
        TABLETOP_EXPERIMENT_INITIAL_POS,
        np.deg2rad(np.array([165, -55, 90, -110, -225, 100])),
        np.deg2rad(np.array([120, -20, 100, -135, -245, 115])),
        np.deg2rad(np.array([130, -15, 75, -110, -240, 125])),
    ]

    # this position is equal to the second position in the previous array, with a wrist rotation far away from
    # the initial position
    far_wrist_pos = np.deg2rad(np.array([165, -55, 90, -110, -225, 300]))

    composite_robot = _build_robot(
        control_mode=control_mode, max_position_change=max_position_change
    )
    arm = composite_robot.robots[0]
    arm.autostep = True
    assert isinstance(arm, Arm)

    # - - - - - - - - - - - - - - - -
    # try to reach several reachable positions
    # - - - - - - - - - - - - - - - -
    shared_speed = np.deg2rad(30)
    for idx, pos in enumerate(positions4):
        debug_recorder = ReachHelperDebugRecorder(robot=arm)
        reach_ret = reach_helper.reach_position(
            arm,
            pos,
            speed_units_per_sec=shared_speed,
            timeout=10,
            minimum_time_to_move=6,
            debug_recorder=debug_recorder,
        )
        assert reach_ret.reached
        if speed_limit_threshold_mult is not None:
            assert_speed_is_ok(
                debug_recorder, shared_speed, shared_speed * speed_limit_threshold_mult
            )

    # - - - - - - - - - - - - - - - -
    # reach initial position again
    # - - - - - - - - - - - - - - - -
    debug_recorder = ReachHelperDebugRecorder(robot=arm)
    reach_ret = reach_helper.reach_position(
        arm,
        TABLETOP_EXPERIMENT_INITIAL_POS,
        speed_units_per_sec=shared_speed,
        timeout=10,
        minimum_time_to_move=6,
        debug_recorder=debug_recorder,
    )
    assert reach_ret.reached
    if speed_limit_threshold_mult is not None:
        assert_speed_is_ok(
            debug_recorder, shared_speed, shared_speed * speed_limit_threshold_mult
        )

    # - - - - - - - - - - - - - - - -
    # fail due to timeout
    # - - - - - - - - - - - - - - - -
    try:
        debug_recorder = ReachHelperDebugRecorder(robot=arm)
        reach_ret = reach_helper.reach_position(
            arm,
            far_wrist_pos,
            speed_units_per_sec=shared_speed,
            timeout=10,
            minimum_time_to_move=6,
            debug_recorder=debug_recorder,
        )
        assert reach_ret.reached
    except RuntimeError as re:
        assert str(re).startswith(
            "Some controls won't reach their target before the timeout"
        )

    # - - - - - - - - - - - - - - - -
    # attempt the same motion, with a faster speed for the wrist
    # - - - - - - - - - - - - - - - -
    debug_recorder = ReachHelperDebugRecorder(robot=arm)
    slow = np.deg2rad(30)
    fast = np.deg2rad(60)
    fast_wrist_speed = np.asarray([*np.repeat(slow, 5), fast])
    reach_ret = reach_helper.reach_position(
        arm,
        far_wrist_pos,
        speed_units_per_sec=fast_wrist_speed,
        timeout=10,
        minimum_time_to_move=6,
        debug_recorder=debug_recorder,
    )
    assert reach_ret.reached

    if speed_limit_threshold_mult is not None:
        speed_limit_threshold_with_wrist = fast_wrist_speed * speed_limit_threshold_mult
        assert_speed_is_ok(
            debug_recorder, fast_wrist_speed, speed_limit_threshold_with_wrist
        )
