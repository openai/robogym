from typing import Optional

import numpy as np

from robogym.robot.shadow_hand.hand_interface import (
    ACTUATOR_CTRLRANGE,
    ACTUATOR_GROUPS,
    ACTUATORS,
)


def denormalize_by_limit(interpolation, limits):
    """ Scale numbers between -1 and 1 into the given limits, keeping 0 point fixed """
    final_values = limits[:, 1] * interpolation
    final_values[interpolation < 0] = (limits[:, 0] * np.abs(interpolation))[
        interpolation < 0
    ]
    return final_values


def normalize_by_limits(values, limits):
    """
    Scale values from the range of supplied limits into the [-1, 1] range,
    keeping 0 point fixed
    """
    final_values = values / limits[:, 1]
    final_values[values < 0] = (np.abs(values) / limits[:, 0])[values < 0]
    return final_values


def separate_group_control(
    control: np.ndarray, group_name: Optional[str]
) -> np.ndarray:
    """
    Transforms specified control vector by separating given actuator group on a way
    to minimize probability of hand elements colliding when moving actuators within
    that group
    """
    assert control is not None
    assert (group_name in ACTUATOR_GROUPS) or (group_name is None)

    # We don't have to do anything to separate a wrist
    if group_name is None or group_name == "WR":
        return control

    limit_id = 0

    for finger in ["LF", "RF", "MF", "FF"]:
        actuator_name = f"A_{finger}J3"
        actuator_id = ACTUATORS.index(actuator_name)
        if finger == group_name:
            limit_id = 1
        else:
            control[actuator_id] = ACTUATOR_CTRLRANGE[actuator_name][limit_id]

    return control
