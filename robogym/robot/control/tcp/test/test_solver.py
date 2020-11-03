import numpy as np
import pytest

from robogym.envs.rearrange.blocks import make_env
from robogym.robot.robot_interface import ControlMode, TcpSolverMode
from robogym.robot.ur16e.mujoco.free_dof_tcp_arm import FreeDOFTcpArm
from robogym.utils import rotation


def _build_env(control_mode, tcp_solver_mode):
    env = make_env(
        starting_seed=0,
        parameters=dict(
            robot_control_params=dict(
                control_mode=ControlMode(control_mode), tcp_solver_mode=tcp_solver_mode,
            ),
            n_random_initial_steps=0,
        ),
    )
    env.reset()
    return env


def _build_arm(control_mode, tcp_solver_mode):
    env = _build_env(control_mode=control_mode, tcp_solver_mode=tcp_solver_mode)
    return env.mujoco_simulation.robot.robots[0].controller_arm, env


@pytest.mark.parametrize(
    "control_mode, tcp_solver_mode, expected_dims",
    [("tcp+wrist", "mocap", [5]), ("tcp+roll+yaw", "mocap", [None, 5])],
)
def test_get_joint_mapping(
    control_mode: str, tcp_solver_mode: str, expected_dims: np.ndarray
):
    arm, _ = _build_arm(control_mode=control_mode, tcp_solver_mode=tcp_solver_mode)
    solver = arm.solver
    mapping = solver.get_joint_mapping()
    assert np.array_equal(mapping, expected_dims)


def calculate_quat_adjustment_due_to_axis_alignment(
    from_quat: np.ndarray, aligned_axis_index: int
):
    """Compute what adjustment quat we would have to provide from the given quat, so that it is aligned with the
    axis defined by the alignment axis index.

    :param from_quat: Quaternion to align.
    :param aligned_axis_index: Index of the euler axis to align to.
    :return: Quaternion we would have to apply to the given quaternion so that it aligns with the given euler axis.
    """
    # test vector is the alignment axis
    test_vector = np.zeros(3)
    test_vector[aligned_axis_index] = 1.0

    # calculate the deviation with respect to the alignment axis
    current_vector = rotation.quat_rot_vec(from_quat, test_vector)
    dot_product = np.dot(current_vector, test_vector)

    # now, we would expect the code to rotate from current_vector to +-test_vector, with +-test_vector being
    # the closest one to current_vector.
    test_vector *= np.sign(dot_product)

    # expected rotation would be the angle_diff along the axis perpendicular to both vectors
    angle_diff = np.arccos(np.abs(dot_product))
    axis_of_rotation = np.cross(current_vector, test_vector)
    adjustment = rotation.quat_from_angle_and_axis(
        angle=angle_diff, axis=axis_of_rotation
    )
    return adjustment


def do_test_solver_quat_response_vs_test_response(
    arm: FreeDOFTcpArm, desired_control: np.ndarray, tcp_solver_mode: TcpSolverMode
):
    """Helper function that tests (via assertions) that when we feed a certain control to a solver, we
    get the expected delta quaternion as response.

    :param arm: Arm.
    :param desired_control: Control to send to the solver.
    :param tcp_solver_mode: Tcp solver mode used to create the arm (so that we can understand expectations)
    :return: None. Asserts the conditions internally.
    """
    # compute the control quat from the desired control
    euler_control = np.zeros(3)
    for control_idx, dimension in enumerate(arm.DOF_DIMS):
        euler_control[dimension.value] = desired_control[control_idx]
    quat_ctrl = rotation.euler2quat(euler_control)

    # - - - - - - - - - - - - - - - - - - - -
    # Ask the solver to compute the delta
    # - - - - - - - - - - - - - - - - - - - -
    solver_delta_quat = arm.solver.get_tcp_quat(desired_control)

    # - - - - - - - - - - - - - - - - - - - -
    # Compute expectations and compare
    # - - - - - - - - - - - - - - - - - - - -
    assert tcp_solver_mode is TcpSolverMode.MOCAP

    # compute the quaternion we would get with the control
    current_rot_quat = rotation.euler2quat(arm.observe().tcp_rot())
    target_quaternion = rotation.quat_mul(current_rot_quat, quat_ctrl)

    # if we have an alignment axis, compute its adjustment due to alignment
    if arm.solver.alignment_axis is not None:
        adjustment = calculate_quat_adjustment_due_to_axis_alignment(
            from_quat=target_quaternion,
            aligned_axis_index=arm.solver.alignment_axis.value,
        )

        # apply the adjustment to the target quaternion
        target_quaternion = rotation.quat_mul(adjustment, target_quaternion)

    # mocap reports a relative quaternion with respect to the current quaternion via subtraction.
    # Replicate that here. Note that we apply the adjustment to the target_quaternion, but then we return
    # relative with respect to the current quaternion (relative via subtraction)
    test_delta = target_quaternion - current_rot_quat

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Assert expectation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # The expectation here is that applying the delta to the current rotation yields the same rotation in both
    # cases, unit-test vs tested code. Note however that we can't compare deltas directly because if the
    # original quaternions are equivalent, but not identical, the subtraction makes them non-comparable!
    # For example:
    #   q1 - q != q2 - q, if q1 and q2 are equivalent but not identical
    # This happens for example for:
    #   q1 = euler2quat( [-180, 0, 170.396] ) = [ 0.417 -0.832  0.327  0.164]
    #   q2 = euler2quat( [ 180, 0, 170.396] ) = [ 0.417  0.832 -0.327  0.164]
    # which is exactly what we have in this unit-test for zero_control.
    # So we can't do just do 'assert test_delta == solver_delta_quat', we have to compare the resulting target quat

    # compute what the resulting rotation would be after we re-add the rotation to the delta
    expected_quat = current_rot_quat + test_delta
    obtained_quat = current_rot_quat + solver_delta_quat

    # calculate the rotational difference, which we expect to be 0
    difference_quat = rotation.quat_difference(expected_quat, obtained_quat)
    difference_euler = rotation.quat2euler(difference_quat)
    assert np.allclose(np.zeros(3), difference_euler, atol=1e-5)


@pytest.mark.parametrize(
    "control_mode, tcp_solver_mode",
    [("tcp+wrist", TcpSolverMode.MOCAP), ("tcp+roll+yaw", TcpSolverMode.MOCAP)],
)
def test_zero_control(control_mode: str, tcp_solver_mode: TcpSolverMode):
    """
    Test that when we apply zero control to an arm, we obtain the expected delta quaternion from the arm's solver.
    """
    tcp_solver_mode_str = tcp_solver_mode.value
    arm, _ = _build_arm(control_mode=control_mode, tcp_solver_mode=tcp_solver_mode_str)
    zero_quat_ctrl = np.zeros_like(arm.DOF_DIMS, dtype=np.float32)

    do_test_solver_quat_response_vs_test_response(
        arm=arm, desired_control=zero_quat_ctrl, tcp_solver_mode=tcp_solver_mode
    )


@pytest.mark.parametrize(
    "control_mode, tcp_solver_mode",
    [("tcp+wrist", TcpSolverMode.MOCAP), ("tcp+roll+yaw", TcpSolverMode.MOCAP)],
)
def test_wrist_rotations(control_mode: str, tcp_solver_mode: TcpSolverMode):
    """
    Test that when we apply controls with rotations along the wrist dimension to an arm, we obtain the expected delta
    quaternion from the arm's solver.
    """
    tcp_solver_mode_str = tcp_solver_mode.value
    arm, _ = _build_arm(control_mode=control_mode, tcp_solver_mode=tcp_solver_mode_str)

    # test several rotations (since the arm doesn't move we can actually test several transformations at once)
    test_rotations = [
        np.deg2rad(30),
        np.deg2rad(60),
        np.deg2rad(90),
        np.deg2rad(160),
        np.deg2rad(180),
    ]
    for rot in test_rotations:
        # POSITIVE ROTATION
        desired_ctrl = np.zeros_like(arm.DOF_DIMS, dtype=np.float32)
        desired_ctrl[-1] = rot  # wrist is the last dimension

        # check that the rotation meets expectations
        do_test_solver_quat_response_vs_test_response(
            arm=arm, desired_control=desired_ctrl, tcp_solver_mode=tcp_solver_mode
        )

        # NEGATIVE ROTATION
        desired_ctrl_neg = np.zeros_like(arm.DOF_DIMS, dtype=np.float32)
        desired_ctrl_neg[-1] = -rot  # wrist is the last dimension (opposite rotation)

        # check that the rotation meets expectations
        do_test_solver_quat_response_vs_test_response(
            arm=arm, desired_control=desired_ctrl_neg, tcp_solver_mode=tcp_solver_mode
        )


@pytest.mark.parametrize(
    "control_mode, tcp_solver_mode", [("tcp+roll+yaw", TcpSolverMode.MOCAP)]
)
def test_quat_second_dof_rotation(control_mode: str, tcp_solver_mode: TcpSolverMode):
    """
    Test that when we apply controls with rotations along the first dimension to an arm, we obtain the expected delta
    quaternion from the arm's solver.
    """
    tcp_solver_mode_str = tcp_solver_mode.value
    arm, _ = _build_arm(control_mode=control_mode, tcp_solver_mode=tcp_solver_mode_str)

    # test several rotations (since the arm doesn't move we can actually test several transformations at once)
    test_rotations = [
        np.deg2rad(30),
        np.deg2rad(60),
        np.deg2rad(90),
        np.deg2rad(160),
        np.deg2rad(180),
    ]
    for rot in test_rotations:
        # POSITIVE ROTATION
        desired_ctrl = np.zeros_like(arm.DOF_DIMS, dtype=np.float32)
        desired_ctrl[0] = rot  # first dimension

        # check that the rotation meets expectations
        do_test_solver_quat_response_vs_test_response(
            arm=arm, desired_control=desired_ctrl, tcp_solver_mode=tcp_solver_mode
        )

        # NEGATIVE ROTATION
        desired_ctrl_neg = np.zeros_like(arm.DOF_DIMS, dtype=np.float32)
        desired_ctrl_neg[0] = -rot  # first dimension

        # check that the rotation meets expectations
        do_test_solver_quat_response_vs_test_response(
            arm=arm, desired_control=desired_ctrl_neg, tcp_solver_mode=tcp_solver_mode
        )
