import numpy as np

from robogym.robot.shadow_hand.hand_interface import (
    ACTUATOR_CTRLRANGE,
    ACTUATOR_GROUPS,
    ACTUATOR_JOINT_MAPPING,
    ACTUATORS,
    CONTROL_TO_POSITION_MATRIX,
    POSITION_TO_CONTROL_MATRIX,
    actuator2group,
    filter_actuator_groups,
    joint2group,
    matching_actuators,
)
from robogym.robot.shadow_hand.mujoco.shadow_hand_simulation import ShadowHandSimulation


def test_matching_actuators():
    def verify(pattern, expected):
        assert matching_actuators(pattern) == expected

    verify(None, set(ACTUATORS))
    verify("wrj0", {"A_WRJ0"})
    verify("A_wrj0", {"A_WRJ0"})
    verify("FFJ1", {"A_FFJ1"})
    verify("WRJ0", {"A_WRJ0"})
    verify("wRj", {"A_WRJ1", "A_WRJ0"})
    verify("WRJ", {"A_WRJ1", "A_WRJ0"})
    verify(
        "WRJ,A_MFJ1,RFJ", {"A_WRJ1", "A_WRJ0", "A_MFJ1", "A_RFJ3", "A_RFJ2", "A_RFJ1"}
    )


def test_actuator2group():
    for actuator in ACTUATORS:
        group = actuator2group(actuator)
        assert actuator in ACTUATOR_GROUPS[group]


def test_joint2group():
    for actuator in ACTUATORS:
        for joint in ACTUATOR_JOINT_MAPPING[actuator]:
            group = joint2group(joint)
            assert actuator in ACTUATOR_GROUPS[group]


def test_filter_actuator_groups():
    assert filter_actuator_groups(None) == ACTUATOR_GROUPS
    assert filter_actuator_groups([]) == ACTUATOR_GROUPS
    assert filter_actuator_groups(["A_FFJ1", "A_LFJ2", "A_LFJ3"]) == {
        "FF": ["A_FFJ1"],
        "LF": ["A_LFJ3", "A_LFJ2"],
    }


def test_control_ranges():
    """ Check if control ranges match between MuJoCo sim and RoboticHand interface """
    sim = ShadowHandSimulation.build()

    mujoco_ctrlrange = sim.mj_sim.model.actuator_ctrlrange.copy()
    hand_ctrlrange = np.array([ACTUATOR_CTRLRANGE[a] for a in ACTUATORS])

    assert np.linalg.norm(mujoco_ctrlrange - hand_ctrlrange) < 1e-8


def test_joint_projection():
    """
    Make sure that transforming joint positions to control vector and back
    preserves the values
    """
    m1 = POSITION_TO_CONTROL_MATRIX @ CONTROL_TO_POSITION_MATRIX
    assert np.linalg.norm(m1 - np.eye(20)) < 1e-8


def test_normalize_position():
    """ Test if position normalization works as expected, """
    sim = ShadowHandSimulation.build()
    hand = sim.shadow_hand

    for i in range(10):  # Test ten times
        position = np.random.uniform(-1.0, 1.0, size=len(ACTUATORS))
        scaled = hand.denormalize_position_control(position)
        assert hand.is_position_control_valid(scaled)
        unscaled = hand.normalize_position_control(scaled)

        assert np.linalg.norm(position - unscaled) < 1e-8
