import mock
import numpy as np
import pytest

from robogym.envs.rearrange.blocks import BlockRearrangeSim, BlockRearrangeSimParameters
from robogym.robot.robot_interface import (
    ControlMode,
    RobotControlParameters,
    TcpSolverMode,
)


def _build_arm(control_mode, max_position_change, tcp_solver_mode=TcpSolverMode.MOCAP):
    sim = BlockRearrangeSim.build(
        n_substeps=1,
        simulation_params=BlockRearrangeSimParameters(),
        robot_control_params=RobotControlParameters(
            control_mode=control_mode,
            max_position_change=max_position_change,
            tcp_solver_mode=tcp_solver_mode,
        ),
    )
    return sim.robot.robots[0]


@pytest.mark.parametrize(
    "max_position_change,expected_command",
    [
        (None, [6.1959, 6.1959, 2.8, 6.1959, 6.1959, 2.312561]),
        (2.5, [2.5, 2.5, 2.5, 2.5, 2.5, 2.312561]),
        ([1.0, 1.0, 1.0, 0.5, 0.5, 9.0], [1.0, 1.0, 1.0, 0.5, 0.5, 2.312561]),
    ],
)
def test_actuation_range_with_rel_action(max_position_change, expected_command):
    arm = _build_arm(ControlMode.JOINT.value, max_position_change)
    assert np.allclose(arm.actuation_range(relative_action=True), expected_command)


@pytest.mark.parametrize(
    "max_position_change", [None, 3.0, [1.0, 1.0, 1.0, 0.5, 0.5, 9.0]]
)
def test_actuation_range_with_abs_action(max_position_change):
    """ Absolute actions should ignore max_position_change command."""
    arm = _build_arm(ControlMode.JOINT.value, max_position_change)
    assert np.allclose(
        arm.actuation_range(relative_action=False),
        [6.1959, 6.1959, 2.8, 6.1959, 6.1959, 2.312561],
    )


def test_composite_autostep():
    from robogym.robot.composite.ur_gripper_arm import (
        MujocoURJointGripperCompositeRobot,
    )

    sim = mock.MagicMock()
    robot = MujocoURJointGripperCompositeRobot(
        simulation=sim,
        solver_simulation=sim,
        robot_control_params=RobotControlParameters(max_position_change=0.05),
        autostep=True,
    )

    assert sim.mj_sim.step.call_count == 0

    robot.set_position_control(np.zeros(7))

    assert sim.mj_sim.step.call_count == 1


def test_free_wrist_composite():
    arm = _build_arm(
        control_mode=ControlMode.TCP_WRIST,
        max_position_change=0.05,
        tcp_solver_mode=TcpSolverMode.MOCAP_IK,
    )
    for applied_ctrl in [-1, 0, 1]:
        expected_change_deg = 30 * applied_ctrl
        new_pos = arm.denormalize_position_control(
            np.concatenate((np.zeros(3), [applied_ctrl])), relative_action=True
        )
        assert np.isclose(new_pos[-1], np.deg2rad(expected_change_deg), atol=1e-7)
