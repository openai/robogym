import pytest

from robogym.envs.rearrange.blocks import make_env
from robogym.robot.composite.ur_gripper_arm import (
    MujocoIdealURGripperCompositeRobot as IdealDynamicsCls,
)
from robogym.robot.composite.ur_gripper_arm import (
    MujocoURTcpJointGripperCompositeRobot as JointDynamicsCls,
)
from robogym.robot.robot_interface import (
    ControlMode,
    RobotControlParameters,
    TcpSolverMode,
)
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import (
    FreeRollYawTcpArm,
    FreeWristTcpArm,
)


def test_rearrange_defaults():
    from robogym.robot.composite.ur_gripper_arm import (
        MujocoURTcpJointGripperCompositeRobot,
    )

    env = make_env()
    assert isinstance(env.robot, MujocoURTcpJointGripperCompositeRobot)
    assert (
        env.parameters.robot_control_params.max_position_change
        == RobotControlParameters.default_max_pos_change_for_solver(
            control_mode=ControlMode.TCP_ROLL_YAW,
            tcp_solver_mode=TcpSolverMode.MOCAP_IK,
        )
    )
    assert env.parameters.robot_control_params.arm_reset_controller_error
    assert env.parameters.robot_control_params.control_mode is ControlMode.TCP_ROLL_YAW
    assert env.parameters.robot_control_params.tcp_solver_mode is TcpSolverMode.MOCAP_IK
    assert env.action_space.shape == (6,)


@pytest.mark.parametrize(
    "control_mode, expected_action_dims, tcp_solver_mode, expected_main_robot, expected_helper_arm",
    [
        (
            ControlMode.TCP_WRIST,
            5,
            TcpSolverMode.MOCAP,
            IdealDynamicsCls,
            FreeWristTcpArm,
        ),
        (
            ControlMode.TCP_WRIST,
            5,
            TcpSolverMode.MOCAP_IK,
            JointDynamicsCls,
            FreeWristTcpArm,
        ),
        (
            ControlMode.TCP_ROLL_YAW,
            6,
            TcpSolverMode.MOCAP,
            IdealDynamicsCls,
            FreeRollYawTcpArm,
        ),
        (
            ControlMode.TCP_ROLL_YAW,
            6,
            TcpSolverMode.MOCAP_IK,
            JointDynamicsCls,
            FreeRollYawTcpArm,
        ),
    ],
)
def test_rearrange_with_ur_tcp(
    control_mode,
    expected_action_dims,
    tcp_solver_mode,
    expected_main_robot,
    expected_helper_arm,
):
    env = make_env(
        parameters=dict(
            robot_control_params=dict(
                control_mode=control_mode,
                tcp_solver_mode=tcp_solver_mode,
                max_position_change=0.1,
            )
        )
    )
    assert isinstance(env.robot, expected_main_robot)
    assert isinstance(env.robot.robots[0].controller_arm, expected_helper_arm)
    assert env.robot.robots[0].max_position_change == 0.1
    assert env.robot.robots[1].max_position_change is None
    assert env.action_space.shape == (expected_action_dims,)
    assert env.robot.autostep is False, "Robot should not be in autostep mode"


def test_rearrange_sim_defaults():
    env = make_env(
        parameters=dict(
            robot_control_params=dict(
                control_mode=ControlMode.TCP_WRIST, tcp_solver_mode=TcpSolverMode.MOCAP,
            ),
        )
    )
    assert env.robot.autostep is False
    arm_robot = env.robot.robots[0]
    assert (
        arm_robot.simulation == arm_robot.controller_arm.simulation
    ), "Simulation should be shared"
    assert (
        arm_robot.controller_arm.autostep is False
    ), "Controller arm is not allowed to autostep"


def test_rearrange_with_ur_joint():
    from robogym.robot.composite.ur_gripper_arm import (
        MujocoURJointGripperCompositeRobot,
    )

    env = make_env(
        parameters=dict(
            robot_control_params=dict(
                control_mode=ControlMode.JOINT, max_position_change=2.4,
            )
        )
    )
    assert isinstance(env.robot, MujocoURJointGripperCompositeRobot)
    assert env.robot.robots[0].max_position_change == 2.4
    assert env.robot.robots[1].max_position_change is None
    assert env.parameters.robot_control_params.control_mode == ControlMode.JOINT
    assert env.action_space.shape == (7,)
