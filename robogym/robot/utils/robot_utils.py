"""Utilities that expand robot interface's capabilities, but that do not necessarily belong to the interface itself."""
from typing import Dict, Iterable, Optional, Type

import numpy as np

from robogym.robot.robot_interface import Robot


def denormalize_actions(
    robot: Robot,
    input_actions: Iterable[Dict[str, float]],
    defaults: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transform position control from the [-1,+1] range into supported joint position range in radians, where 0
    represents the midpoint between two extreme positions, for the supplied actuators only, using default values
    for non-supplied actuators.

    To supply the actuators to denormalize, provide a dictionary or a batch of dictionaries with the actuator
    as the key, and the [-1,+1] value as entry value.

    :param robot: Robot for which we want to denormalize actions.
    :param input_actions: Batch of dictionaries with actuator as key, and normalized value as value.
    :param defaults: Default values for actuators not specified in actions. 0 if not set.
    :return: Batch of arrays with denormalized values for actuators specified in the dictionary and
    0 for actuators not specified in the dictionary.
    """
    robot_actuators = robot.actuators()

    if defaults is None:
        defaults = robot.zero_control()
    assert defaults.shape == (len(robot_actuators),)

    ctrls = []
    # for each entry in the batch
    for action_overrides in input_actions:

        # create array with desired values for specified actuators
        normal_positions = np.zeros(len(robot_actuators))
        for actuator, value in action_overrides.items():
            assert actuator in robot_actuators
            actuator_id = robot_actuators.index(actuator)
            normal_positions[actuator_id] = value

        # denormalize array
        denormalized_positions = robot.denormalize_position_control(normal_positions)

        # generate output with denormalized positions only for controls that were specified, and default otherwise
        ctrl = defaults.copy()
        for actuator in action_overrides:
            actuator_id = robot_actuators.index(actuator)
            ctrl[actuator_id] = denormalized_positions[actuator_id]
        ctrls.append(ctrl)

    return np.asarray(ctrls)


def find_robot_by_class(
    top_robot: Robot, desired_class: Type[Robot]
) -> Optional[Robot]:
    """Examines the given robot to retrieve the first instance that matches the specified desired class. This
    includes searching in the children of a CompositeRobot.

    :param top_robot: Robot to search in. This robot can also match the desired class.
    :param desired_class: Desired specific class we are searching for.
    :return: First robot found that matches the desired class, or None if no matches are found.
    """

    desired_robot = None
    from robogym.robot.composite.composite_robot import CompositeRobot

    if isinstance(top_robot, CompositeRobot):
        for internal_robot in top_robot.robots:
            if isinstance(internal_robot, desired_class):
                desired_robot = internal_robot
                break
    elif isinstance(top_robot, desired_class):
        desired_robot = top_robot

    return desired_robot
