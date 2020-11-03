import abc
from typing import Optional

import numpy as np

from robogym.robot.robot_interface import Robot

ACTUATOR_CTRLRANGE = {
    "A_J1": [-6.1959, 6.1959],  # DEGREES (-355, 355)
    "A_J2": [-6.1959, 6.1959],  # DEGREES (-355, 355)
    "A_J3": [-2.7925, 2.7925],  # DEGREES (-160, 160)
    "A_J4": [-6.1959, 6.1959],  # DEGREES (-355, 355)
    "A_J5": [-6.1959, 6.1959],  # DEGREES (-355, 355)
    "A_J6": [-6.1959, 6.1959],  # DEGREES (-355, 355)
}
ACTUATORS = [
    "A_J1",
    "A_J2",
    "A_J3",
    "A_J4",
    "A_J5",
    "A_J6",
]

# A good initial setting for Arm joint angles for a tabletop experiment performed with
# the UR arm
TABLETOP_EXPERIMENT_INITIAL_POS = np.deg2rad(np.array([135, -90, 135, -100, -240, 135]))

ACTUATOR_CTRLRANGE_LOWER_BOUND = np.array(
    [ACTUATOR_CTRLRANGE[key][0] for key in ACTUATORS]
)
"""
Actuator position controls vector for the lower bound of actuator control range.
"""

ACTUATOR_CTRLRANGE_UPPER_BOUND = np.array(
    [ACTUATOR_CTRLRANGE[key][1] for key in ACTUATORS]
)
"""
Actuator position controls vector for the upper bound of actuator control range.
"""

"""
Force threshold in Newtons to apply to robot for safety stops.
"""
SAFETY_STOP_FORCE_THRESHOLD = 150


class Arm(Robot, abc.ABC):
    # speed limit for joints/actuators that we want to move slowly
    SLOW_ACTUATOR_SPEED = np.deg2rad(30)
    # speed limit for joints/actuators that we want to move faster
    FAST_ACTUATOR_SPEED = np.deg2rad(60)
    # reach_helper format for speed limits, one per actuator/joint
    REACH_HELPER_DEFAULT_SPEED = np.asarray(
        [*np.repeat(SLOW_ACTUATOR_SPEED, 5), FAST_ACTUATOR_SPEED]
    )

    @classmethod
    def actuators(cls) -> np.ndarray:
        return np.asarray(ACTUATORS)

    @classmethod
    def joint_positions_to_control(cls, joint_pos: np.ndarray) -> np.ndarray:
        return joint_pos

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        return ACTUATOR_CTRLRANGE_UPPER_BOUND

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        return ACTUATOR_CTRLRANGE_LOWER_BOUND

    def get_default_reach_helper_speed(self) -> np.ndarray:
        """Returns the speed per actuator that the reach helper should use unless configured to use something
        different. Note that it's specified as joint/actuator speed since all UR arms are used with joint control
        under reach helper control.

        :return: The speed per joint that the reach helper should use unless configured to use something different.
        """
        return self.REACH_HELPER_DEFAULT_SPEED

    def get_robot_transform(self) -> Optional[np.ndarray]:
        """Return robot current configured world transformation (if any). For mujoco robot this can be the world
        coordinates of the base of the robot.

        Physical robots return their observations in local coordinates. However, we generally want the coordinates
        to be in mujoco space. This is more a convention for our convenience, since debugging coordinates in world
        space can be more comfortable. This transformation allows us to return the observations in world space
        from the robot itself, without having to replicate transformation in all systems that consume world
        coordinates. Note however that there is no such thing as world coordinates for physical robots unless
        an origin is defined for 'world'. That is what we do here, and it generally matches that of an environment.

        :return: Return robot current configured world transformation (if any). For mujoco robot this can be
        the world coordinates of the base of the robot.
        """
        return None
