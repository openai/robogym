import abc
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from robogym.robot.lib.parameter_configurer import ParameterConfigurer
from robogym.robot.robot_interface import Robot, RobotObservation

ACTUATORS = [
    "A_WRJ1",  # actuator_id 00, horizontal wrist movement
    "A_WRJ0",  # actuator_id 01, vertical wrist movement
    "A_FFJ3",  # actuator_id 02, horizontal finger movement
    "A_FFJ2",  # actuator_id 03, vertical finger movement
    "A_FFJ1",  # actuator_id 04, vertical finger movement, bending coupled joints
    "A_MFJ3",  # actuator_id 05, horizontal finger movement
    "A_MFJ2",  # actuator_id 06, vertical finger movement
    "A_MFJ1",  # actuator_id 07, vertical finger movement, bending coupled joints
    "A_RFJ3",  # actuator_id 08, horizontal finger movement
    "A_RFJ2",  # actuator_id 09, vertical finger movement
    "A_RFJ1",  # actuator_id 10, vertical finger movement, bending coupled joints
    "A_LFJ4",  # actuator_id 11, vertical finger movement toward the center of the palm
    "A_LFJ3",  # actuator_id 12, horizontal finger movement
    "A_LFJ2",  # actuator_id 13, vertical finger movement
    "A_LFJ1",  # actuator_id 14, vertical finger movement, bending coupled joints
    "A_THJ4",  # actuator_id 15, rotational movement of the thumb
    "A_THJ3",  # actuator_id 16, bending thumb
    "A_THJ2",  # actuator_id 17, bending thumb
    "A_THJ1",  # actuator_id 18, bending thumb
    "A_THJ0",  # actuator_id 19, bending thumb
]
"""
Each actuator is a motor inside of a robotic hand pulling on two tendons -
one on one side and one on the other.
"""

FINGERS: List[str] = [
    "FF",  # first finger
    "MF",  # middle finger
    "RF",  # ring finger
    "LF",  # little finger
    "TH",  # thumb
]
"""
Names of fingers of the hand.

Order of fingers on this list MUST NOT change, because it is used for indexing and identifiers.
"""

ACTUATOR_GROUPS: Dict[str, List[str]] = {
    "WR": ["A_WRJ1", "A_WRJ0"],
    "FF": ["A_FFJ3", "A_FFJ2", "A_FFJ1"],
    "MF": ["A_MFJ3", "A_MFJ2", "A_MFJ1"],
    "RF": ["A_RFJ3", "A_RFJ2", "A_RFJ1"],
    "LF": ["A_LFJ4", "A_LFJ3", "A_LFJ2", "A_LFJ1"],
    "TH": ["A_THJ4", "A_THJ3", "A_THJ2", "A_THJ1", "A_THJ0"],
}
"""
Actuators grouped by their belonging to the same physical part.
"""


def actuator2group(actuator: str) -> str:
    """
    Return name of the group actuator belongs to.

    :param actuator: name of the actuator
    :return: name of the group
    """
    assert actuator in ACTUATORS, f"{actuator} is not a valid actuator name"
    return actuator[-4:-2]


def joint2group(joint: str) -> str:
    """
    Return name of the group joint belongs to.

    :param joint: name of the joint
    :return: name of the group
    """
    assert joint in JOINTS, f"{joint} is not a valid joint name"
    return joint[-4:-2]


def filter_actuator_groups(actuators: Optional[Iterable[str]]) -> Dict[str, List[str]]:
    """
    Return new actuator groups that contains only provided actuators.

    Actuator group is filtered out if and only if there is no actuator from the provided
    collection of actuators belonging to the actuator group. Actuators of remaining actuator
    groups are filtered out as well to include only specified actuators.

    :param actuators: actuators to include; each actuator is a string exactly
           matching actuator name from shadow_hand.ACTUATORS
    :return: list of new filtered recording groups
    """
    actuators = set(actuators) if actuators else set()
    filtered_actuator_groups = {}
    for group_name, group_actuators in ACTUATOR_GROUPS.items():
        if not actuators:
            filtered_actuator_groups[group_name] = group_actuators.copy()
        elif not set(group_actuators).isdisjoint(actuators):
            filtered_actuator_groups[group_name] = [
                actuator for actuator in group_actuators if actuator in actuators
            ]
    return filtered_actuator_groups


JOINTS = [
    "WRJ1",  # joint_id 00, actuator_id 00
    "WRJ0",  # joint_id 01, actuator_id 01
    "FFJ3",  # joint_id 02, actuator_id 02
    "FFJ2",  # joint_id 03, actuator_id 03
    "FFJ1",  # joint_id 04, actuator_id 04, tendon "FFT1", coupled joint
    "FFJ0",  # joint_id 05, actuator_id 04, tendon "FFT1", coupled joint
    "MFJ3",  # joint_id 06, actuator_id 05
    "MFJ2",  # joint_id 07, actuator_id 06
    "MFJ1",  # joint_id 08, actuator_id 07, tendon "MFT1", coupled joint
    "MFJ0",  # joint_id 09, actuator_id 07, tendon "MFT1", coupled joint
    "RFJ3",  # joint_id 10, actuator_id 08
    "RFJ2",  # joint_id 11, actuator_id 09
    "RFJ1",  # joint_id 12, actuator_id 10, tendon "RFT1", coupled joint
    "RFJ0",  # joint_id 13, actuator_id 10, tendon "RFT1", coupled joint
    "LFJ4",  # joint_id 14, actuator_id 11
    "LFJ3",  # joint_id 15, actuator_id 12
    "LFJ2",  # joint_id 16, actuator_id 13
    "LFJ1",  # joint_id 17, actuator_id 14, tendon "LFT1", coupled joint
    "LFJ0",  # joint_id 18, actuator_id 14, tendon "LFT1", coupled joint
    "THJ4",  # joint_id 19, actuator_id 15
    "THJ3",  # joint_id 20, actuator_id 16
    "THJ2",  # joint_id 21, actuator_id 17
    "THJ1",  # joint_id 22, actuator_id 18
    "THJ0",  # joint_id 23, actuator_id 19
]
"""
Robotic hand joints are mechanical "hinge" joints that represent degrees of freedom of the robot.

Acronyms used:

  WR - Wrist
  FF - First Finger
  MF - Middle Finger
  RF - Ring Finger
  LF - Little Finger
  TH - Thumb

Joints are numbered from fingertip to the palm - e.g. FFJ0 is the first knuckle, FFJ1 is
the second knuckle etc.

First two joints of each of the main fingers are coupled - which means there is only
one actuator controlling them by a single tendon.
"""

ACTUATOR_CTRLRANGE = {
    "A_WRJ1": [-0.4887, 0.1396],  # DEGREES (-28, 8)
    "A_WRJ0": [-0.6981, 0.4887],  # DEGREES (-40, 28)
    "A_FFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "A_FFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "A_FFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "A_MFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "A_MFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "A_MFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "A_RFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "A_RFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "A_RFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "A_LFJ4": [0.0, 0.7854],  # DEGREES (0, 45)
    "A_LFJ3": [-0.3491, 0.3491],  # DEGREES (-20, 20)
    "A_LFJ2": [0.0, 1.5708],  # DEGREES (0, 90)
    "A_LFJ1": [0.0, 3.1416],  # DEGREES (0, 180)
    "A_THJ4": [-1.0472, 1.0472],  # DEGREES (-60, 60)
    "A_THJ3": [0.0, 1.2217],  # DEGREES (0, 70)
    "A_THJ2": [-0.2094, 0.2094],  # DEGREES (-12, 12)
    "A_THJ1": [-0.5236, 0.5236],  # DEGREES (-30, 30)
    "A_THJ0": [-1.5708, 0.0],  # DEGREES (-90, 0)
}
"""
Ranges, in radians, of what robotic joints can safely reach
"""

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

ACTUATOR_JOINT_MAPPING: Dict[str, List[str]] = {
    "A_WRJ1": ["WRJ1"],
    "A_WRJ0": ["WRJ0"],
    "A_FFJ3": ["FFJ3"],
    "A_FFJ2": ["FFJ2"],
    "A_FFJ1": ["FFJ1", "FFJ0"],  # Coupled joints
    "A_MFJ3": ["MFJ3"],
    "A_MFJ2": ["MFJ2"],
    "A_MFJ1": ["MFJ1", "MFJ0"],  # Coupled joints
    "A_RFJ3": ["RFJ3"],
    "A_RFJ2": ["RFJ2"],
    "A_RFJ1": ["RFJ1", "RFJ0"],  # Coupled joints
    "A_LFJ4": ["LFJ4"],
    "A_LFJ3": ["LFJ3"],
    "A_LFJ2": ["LFJ2"],
    "A_LFJ1": ["LFJ1", "LFJ0"],  # Coupled joints
    "A_THJ4": ["THJ4"],
    "A_THJ3": ["THJ3"],
    "A_THJ2": ["THJ2"],
    "A_THJ1": ["THJ1"],
    "A_THJ0": ["THJ0"],
}
"""
One-to-many mapping between actuators and actuated joints.

Some actuators act on a single joint while others are connected to two joints
(so called "coupled" joints).
"""

JOINT_ACTUATOR_MAPPING = {
    k: v for k, vlist in ACTUATOR_JOINT_MAPPING.items() for v in vlist
}
"""
Reverse of the ACTUATOR_JOINT_MAPPING
"""

JOINT_LIMITS: Dict[str, np.ndarray] = {}
for actuator, ctrlrange in ACTUATOR_CTRLRANGE.items():
    joints = ACTUATOR_JOINT_MAPPING[actuator]
    for joint in joints:
        JOINT_LIMITS[joint] = np.array(ctrlrange) / len(joints)
"""
Joint limits in radian are derived from ACTUATOR_CTRLRANGE.

The coupled joints share the full ctrlrange of *FJ1 and thus we divide the range by half.

Shadow official joint limits for *FJ0 and *FJ1 are [0, 90] degree.

See page 7 in https://www.shadowrobot.com/wp-content/uploads/shadow_dexterous_hand_technical_specification_E_20190221.pdf
"""


def _compute_projection_matrices():
    """ compute matrices that project joint positions to actuator control and vice versa """
    position_control_matrix = np.zeros((20, 24))
    control_to_position_matrix = np.zeros((24, 20))

    actuator_ids = dict(zip(ACTUATORS, range(len(ACTUATORS))))
    joint_ids = dict(zip(JOINTS, range(len(JOINTS))))

    for actuator_name, joint_names in ACTUATOR_JOINT_MAPPING.items():
        value = 1.0 / len(joint_names)

        actuator_id = actuator_ids[actuator_name]
        joint_id_array = np.array([joint_ids[j] for j in joint_names])

        position_control_matrix[actuator_id, joint_id_array] = 1.0
        control_to_position_matrix[joint_id_array, actuator_id] = value

    return position_control_matrix, control_to_position_matrix


# Some useful precomputed constants
POSITION_TO_CONTROL_MATRIX, CONTROL_TO_POSITION_MATRIX = _compute_projection_matrices()


def matching_actuators(actuator_pattern: Optional[str] = None) -> Set[str]:
    """
    Returns names of actuators matching given actuator_pattern.

    The actuator pattern is a comma-separated list of actuator names or prefixes,
    for example "A_FFJ0,WRJ1,THJ0", or "FF".
    """
    if actuator_pattern is None:
        return set(ACTUATORS)

    actuator_pattern = actuator_pattern.upper().strip()
    prefixes = actuator_pattern.split(",")

    matches = set()
    for actuator in ACTUATORS:
        for prefix in prefixes:
            # Forgive user is they forgot to add A_ prefix.
            if actuator.startswith(prefix) or actuator.startswith("A_" + prefix):
                matches.add(actuator)
                break
    return matches


class Observation(RobotObservation):
    """
    Interface for the single observation of the shadow hand. Observation object owns and manages
    a set of data that constitutes an "observation", and exposes a number of methods to extract
    commonly used fields (like joint positions or actuator effort).


    All fields in a single observation object come as much as possible from a single moment in time
    (as much as the hardware allows, as different sensors may have different sampling rates).
    """

    @abc.abstractmethod
    def joint_velocities(self) -> np.ndarray:
        """
        Return observed joint velocities (in rad/s)

        :returns 24-element array (one for each joint) of joint velocities
        """
        pass

    @abc.abstractmethod
    def actuator_effort(self) -> np.ndarray:
        """
        Return observed actuator effort normalized to the -1 to 1 range (1 means maximum effort).

        Effort is a unitless number proportional to the force exerted by that actuator, but the
        actual proportionality constant is not known.

        Effort is normalized based on maximum force limit of each actuator. Sign of the force
        indicates direction of the applied force - positive (negative) sign indicate force applied
        towards maximum (minimum) joint position(s).

        :return 20-element array (one for each actuator) of actuator efforts
        """
        pass

    @abc.abstractmethod
    def fingertip_positions(self) -> np.ndarray:
        """
        Return observed fingertip positions in meters measured in a coordinate system
        spanned by the three reference points on the hand mount.

        Finger ordering is: First Finger, Middle Finger, Ring Finger, Little Finger, Thumb

        :returns 5x3-element array of fingertip positions (5 fingertips in 3D)
        """
        pass


class Hand(Robot, abc.ABC):
    """
    High level API for controlling Shadow Dexterous Hand E1 Series.
    It allows manipulation of the hand in real time and making observations about the state of
    the hand. This interface allows to control physical and simulated hands depending on the
    implementation.

    An important distinction here that because of existence of coupled joints there is more joints
    than actuators and some of the actuators act on multiple joints. The control values for
    such actuators is a sum of positions of individual joints, and that's the only value
    we can control.

    With that in mind the hand can be controlled in two separate ways:
    - Via position control, where desired joint positions (or sums of joint positions) are
      specified to the internal controller. Controller then chooses the right force for each
      actuator to reach these positions.
    - Directly via desired effort - that allows us to specify an effort value (proportional
      to force) for each actuator directly.

    Both control modes are mutually exclusive and turning on one turns off the other.
    """

    ##############################################################################################
    # HAND INTERFACE UTILITY METHODS
    @classmethod
    def zero_joint_positions(cls) -> np.ndarray:
        """
        Return an array representing zero joint positions (in rad).
        Zero positions represent a flat, straightened out hand.
        """
        return np.zeros(len(JOINTS), dtype=np.float)

    @classmethod
    def zero_control(cls) -> np.ndarray:
        """
        Return an array representing a zero actuator control vector (in rad).
        """
        return np.zeros(len(cls.actuators()), dtype=np.float)

    @classmethod
    def actuators(cls) -> np.ndarray:
        return ACTUATORS

    @classmethod
    def zero_effort_control(cls) -> np.ndarray:
        """
        Return an array representing a zero actuator effort control vector.
        """
        return np.zeros(len(ACTUATORS), dtype=np.float)

    @classmethod
    def control_to_joint_positions(cls, control: np.ndarray) -> np.ndarray:
        """
        Transform 20-dimensional (position) control to 24-dimensional joint positions.
        Coupled control values are divided evenly among connected joints.
        """
        return CONTROL_TO_POSITION_MATRIX @ control

    @classmethod
    def joint_positions_to_control(cls, joint_pos: np.ndarray) -> np.ndarray:
        """
        Transform 24-dimensional joint positions to 20-dimensional (position) control vector
        Coupled control values are sums of connected joints.
        """
        return POSITION_TO_CONTROL_MATRIX @ joint_pos

    def normalize_position_control(self, position_control: np.ndarray) -> np.ndarray:
        """
        Transform position control from the supported joint position range in radians
        into [-1, +1] range, where 0 represents the midpoint between two extreme positions.
        """
        bounds = (
            self.actuator_ctrl_range_upper_bound()
            - self.actuator_ctrl_range_lower_bound()
        )
        return (
            position_control - self.actuator_ctrl_range_lower_bound()
        ) / bounds * 2 - 1

    @classmethod
    def is_effort_control_valid(cls, control: np.ndarray) -> bool:
        """
        Check if effort control vector is within (inclusive) supported ranges and
        of correct shape
        """
        return (
            control.shape == ACTUATOR_CTRLRANGE_LOWER_BOUND.shape
            and np.all(control >= -1.0)
            and np.all(control <= 1.0)
        )

    @classmethod
    def clip_position_control(cls, control: np.ndarray) -> np.ndarray:
        """ Clip (position) control vector to (inclusive) supported ranges """
        return np.clip(
            control, ACTUATOR_CTRLRANGE_LOWER_BOUND, ACTUATOR_CTRLRANGE_UPPER_BOUND
        )

    @classmethod
    def clip_effort_control(cls, control: np.ndarray) -> np.ndarray:
        """ Clip effort vector to (inclusive) supported ranges """
        return np.clip(control, -1.0, 1.0)

    @classmethod
    def joint_array_to_dict(cls, values: np.ndarray) -> dict:
        """ Convert array of joint readings into a dictionary """
        return dict(zip(JOINTS, values))

    @classmethod
    def actuator_array_to_dict(cls, values: np.ndarray) -> dict:
        """ Convert array of actuator control into a dictionary """
        return dict(zip(ACTUATORS, values))

    @classmethod
    def actuator_dict_to_array(cls, data: dict) -> np.ndarray:
        """ Convert a dictionary of actuator data into an array """
        return np.array([data[a] for a in ACTUATORS])

    @classmethod
    def joint_dict_to_array(cls, data: dict) -> np.ndarray:
        """ Convert a dictionary of joint data into an array """
        return np.array([data[a] for a in JOINTS])

    @abc.abstractmethod
    def parameter_manager(self) -> ParameterConfigurer:
        """
        Returns an instance of ParameterConfigurer that can be used to dynamically
        set parameters of the Hand
        """
        pass

    def parameter_bounds(self, actuator: str):
        """Get valid parameter bounds for the given actuator."""
        return self.parameter_manager().parameter_bounds(actuator)

    def current_parameters(self, actuator: str) -> Dict[str, float]:
        """Get current parameters for the given actuator."""
        return self.parameter_manager().current_parameters(actuator)

    def set_parameters(self, actuator: str, assignments: Dict[str, float]):
        """Set parameters for the given actuator."""
        self.parameter_manager().set_parameters(actuator, assignments)

    def export_parameters(self):
        """Export current parameters.

        For example, save parameters to XML or JSON configuration file."""
        return self.parameter_manager().export_parameters()

    @abc.abstractmethod
    def observe(self) -> Observation:
        """
        Return the "observation" object which contains all the most recent,
        contemporaneous observations of the state of the robotic hand.
        """
        pass

    @abc.abstractmethod
    def set_effort_control(self, control: np.ndarray) -> None:
        """
        Set desired actuator effort vector. Each coordinate of this vector is the desired
        normalized effort value between -1 and 1 for each motor.

        Effort is a unitless number proportional to the force exerted by that actuator, but the
        actual proportionality constant is not known.

        Effort is normalized based on maximum force limit of each actuator. Sign of the force
        indicates direction of the applied force - positive (negative) sign indicate force applied
        towards maximum (minimum) joint position(s).

        Both control modes are mutually exclusive and turning on one turns off the other.

        :param control: 20-element array of actuator force control
        """
        pass
