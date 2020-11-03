from typing import Tuple

import numpy as np

from robogym.robot.utils.logistic_functions import clipped_logistic_sigmoid

LOGISTIC_ALPHA_PARAMETER = (
    0.81  # the 0.81 value for the sigmoid is a relatively slow transition
)
MAXIMUM_TCP_FORCE_TORQUE = 40  # Newtons
TRIGGER_FORCE_TORQUE_THRESHOLD = (
    MAXIMUM_TCP_FORCE_TORQUE * 0.50
)  # % of max force to start scaling
MINIMUM_SCALING_FACTOR = (
    0.0  # The minimum value -- between zero and one -- to scale the force by
)
# Typical values for this are between 5% to 0.1% (0.05 to 0.001)

OVER_MAX_REVERSE_SCALE = (
    -0.1
)  # If the force is over MAXIMUM_TCP_FORCE_TORQUE, how much to reverse scale the force
# Keep in mind this is a scale, so it will be multiplied by the current control,
# not the current force. Something interesting to try could be to reverse scale by
# the amount that the force is over the maximum (up to max_position_change) but this
# is a bit more complicated to implement so we chose to keep things simple for now

assert np.sign(OVER_MAX_REVERSE_SCALE) == -1


def get_element_wise_tcp_control_limits(
    tcp_force_and_torque_measured: np.ndarray, reverse_over_max: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Takes the TCP force and uses a max_force threshold to scale the input values
    to limit them based on a logistic sigmoid. The limiting starts at TRIGGER_FORCE_TORQUE_THRESHOLD
    and goes to MINIMUM_SCALING_FACTOR at MAXIMUM_TCP_FORCE_TORQUE. We use MINIMUM_SCALING_FACTOR in
    order to prevent the output from getting the controls in a place where the robot cannot move out

    :param tcp_force_and_torque_measured: force measured on the arm's TCP as a
    1x6 array (x, y, z, roll, pitch, yaw)

    :return: Tuple of
            1. scales (1x6 nparray) - a scaling factor (0 to 1) for the control 1x6 np.ndarray (x, y, z, roll, pitch, yaw)
            2. triggered (bool) - a bool which is true if any of the forces were above the trigger force
    """

    force_and_torque_scales = np.ones_like(tcp_force_and_torque_measured)
    force_and_torque_over_threshold_mask = (
        tcp_force_and_torque_measured > TRIGGER_FORCE_TORQUE_THRESHOLD
    )

    # return a value ranging from 0 to 1 from the clipped logistic sigmoid
    # the 0.81 value for the sigmoid is a relatively slow transition
    exponential_scaler = clipped_logistic_sigmoid(
        (
            np.maximum(
                (
                    MAXIMUM_TCP_FORCE_TORQUE
                    - tcp_force_and_torque_measured[
                        force_and_torque_over_threshold_mask
                    ]
                ),
                0,
            )
            / (MAXIMUM_TCP_FORCE_TORQUE - TRIGGER_FORCE_TORQUE_THRESHOLD)
        ),
        LOGISTIC_ALPHA_PARAMETER,
    )
    # scale the 0 to 1 range to 0 to (1 - MINIMUM_SCALING_FACTOR) and
    # then shift up by MINIMUM_SCALING_FACTOR
    force_and_torque_scales[force_and_torque_over_threshold_mask] = (
        exponential_scaler * (1.0 - MINIMUM_SCALING_FACTOR) + MINIMUM_SCALING_FACTOR
    )

    # if enabled, reverse the controls by a scaling factor that is negative
    if reverse_over_max:
        force_and_torque_over_maximum_mask = (
            tcp_force_and_torque_measured > MAXIMUM_TCP_FORCE_TORQUE
        )
        force_and_torque_scales[
            force_and_torque_over_maximum_mask
        ] = OVER_MAX_REVERSE_SCALE

    return force_and_torque_scales, any(force_and_torque_over_threshold_mask)
