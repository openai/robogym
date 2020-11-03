import numpy as np
import pytest

from robogym.robot.control.tcp.force_based_tcp_control_limiter import (
    MAXIMUM_TCP_FORCE_TORQUE,
    MINIMUM_SCALING_FACTOR,
    OVER_MAX_REVERSE_SCALE,
    TRIGGER_FORCE_TORQUE_THRESHOLD,
    get_element_wise_tcp_control_limits,
)


def compute_expected_trigger_state():
    """
    Use the values printed out by this function to adjust the values in test_get_joint_mapping().
    """
    forces = np.array([float(i) for i in range(100)])
    scales = []
    triggers = []
    for force in forces:
        scale, trigger = get_element_wise_tcp_control_limits(np.ones(6) * force)
        scales.append(scale[0])
        triggers.append(trigger)

    print(
        *[
            f"\tForce: {forces[i]}\t Scale: {scales[i]}\t Triggered: {triggers[i]}\n"
            for i, _ in enumerate(forces)
        ]
    )


@pytest.mark.parametrize(
    "force_torque, expected_scales, expected_trigger_state",
    [
        (np.ones(6) * TRIGGER_FORCE_TORQUE_THRESHOLD - 1.0, np.ones(6) * 1.0, False),
        (np.ones(6) * TRIGGER_FORCE_TORQUE_THRESHOLD, np.ones(6) * 1.0, False),
        (
            np.ones(6) * TRIGGER_FORCE_TORQUE_THRESHOLD + 1.0,
            np.ones(6) * 0.9925695,
            True,
        ),
        (np.ones(6) * MAXIMUM_TCP_FORCE_TORQUE - 1.0, np.ones(6) * 0.00743045, True),
        (
            np.ones(6) * MAXIMUM_TCP_FORCE_TORQUE,
            np.ones(6) * MINIMUM_SCALING_FACTOR,
            True,
        ),
        (
            np.ones(6) * MAXIMUM_TCP_FORCE_TORQUE + 1.0,
            np.ones(6) * OVER_MAX_REVERSE_SCALE,
            True,
        ),
        (
            np.ones(6) * MAXIMUM_TCP_FORCE_TORQUE * 2.0,
            np.ones(6) * OVER_MAX_REVERSE_SCALE,
            True,
        ),
        (
            np.array([0.0, 0.0, 0.0, MAXIMUM_TCP_FORCE_TORQUE, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, MINIMUM_SCALING_FACTOR, 1.0, 1.0]),
            True,
        ),
    ],
)
def test_get_joint_mapping(
    force_torque: np.ndarray, expected_scales: np.ndarray, expected_trigger_state: bool
):
    scales, triggered = get_element_wise_tcp_control_limits(force_torque)
    assert np.allclose(scales, expected_scales)
    assert triggered == expected_trigger_state
