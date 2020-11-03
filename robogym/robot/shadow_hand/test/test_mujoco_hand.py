import mock
import numpy as np

from robogym.robot.shadow_hand.hand_forward_kinematics import (
    compute_forward_kinematics_fingertips,
)
from robogym.robot.shadow_hand.hand_interface import (
    ACTUATOR_CTRLRANGE,
    ACTUATOR_CTRLRANGE_LOWER_BOUND,
    ACTUATOR_CTRLRANGE_UPPER_BOUND,
    ACTUATOR_GROUPS,
    ACTUATORS,
)
from robogym.robot.shadow_hand.hand_utils import separate_group_control
from robogym.robot.shadow_hand.mujoco.mujoco_shadow_hand import MuJoCoShadowHand
from robogym.robot.shadow_hand.mujoco.shadow_hand_simulation import ShadowHandSimulation


def test_mujoco_forward_kinematics():
    """ Make sure forward kinematics we calculate agree with mujoco values """
    simulation = ShadowHandSimulation.build()
    hand = simulation.shadow_hand

    for i in range(5):
        position = np.random.uniform(-1.0, 1.0, size=len(ACTUATORS))
        scaled = hand.denormalize_position_control(position)

        hand.set_position_control(scaled)

        # Make sure simulation has enough time to reach position
        for _ in range(100):
            simulation.step()

        observation = hand.observe()
        computed_fingertips = compute_forward_kinematics_fingertips(
            observation.joint_positions()
        )

        assert (
            np.abs(observation.fingertip_positions() - computed_fingertips) < 1e-6
        ).all()


def test_mujoco_move_hand():
    """ Test if we can move and observe mujoco hand """
    simulation = ShadowHandSimulation.build()
    hand = simulation.shadow_hand

    for actuator_group in sorted(ACTUATOR_GROUPS.keys()):
        for actuator in ACTUATOR_GROUPS[actuator_group]:
            position_control = hand.zero_control()
            position_control = separate_group_control(position_control, actuator_group)

            random_value = np.random.uniform(0.0, 1.0)

            lower_bound = ACTUATOR_CTRLRANGE[actuator][0]
            upper_bound = ACTUATOR_CTRLRANGE[actuator][1]

            position_control[ACTUATORS.index(actuator)] = (
                random_value * (upper_bound - lower_bound) + lower_bound
            )

            hand.set_position_control(position_control)

            # Make sure simulation has enough time to reach position
            for _ in range(100):
                simulation.step()

            observation = hand.observe()
            positions_observed = hand.joint_positions_to_control(
                observation.joint_positions()
            )
            error = positions_observed - position_control

            assert (np.rad2deg(np.abs(error)) < 7.5).all()


def test_mujoco_effort_move():
    """ Test if mujoco effort control works """
    simulation = ShadowHandSimulation.build()
    hand = simulation.shadow_hand

    # Actuators that we can safely steer using effort control
    # Actuators J3 for each finger are special in a way, that until
    # we really try to separate the fingers, they're super likely to
    # collide with other fingers, preventing them from reaching desired position
    safe_actuators = [
        "A_WRJ1",  # 0
        "A_WRJ0",  # 1
        # "A_FFJ3",  # 2
        "A_FFJ2",  # 3
        "A_FFJ1",  # 4
        # "A_MFJ3",  # 5
        "A_MFJ2",  # 6
        "A_MFJ1",  # 7
        # "A_RFJ3",  # 8
        "A_RFJ2",  # 9
        "A_RFJ1",  # 10
        "A_LFJ4",  # 11
        # "A_LFJ3",  # 12
        "A_LFJ2",  # 13
        "A_LFJ1",  # 14
        "A_THJ4",  # 15
        "A_THJ3",  # 16
        "A_THJ2",  # 17
        "A_THJ1",  # 18
        "A_THJ0",  # 19
    ]

    for selected_actuator in safe_actuators:
        for selected_force in [-1.0, 1.0]:
            actuator_index = ACTUATORS.index(selected_actuator)

            effort_control = hand.zero_control()

            effort_control[actuator_index] = selected_force

            # Maximum positive effort
            hand.set_effort_control(effort_control)

            # Make sure simulation has enough time to reach position
            for _ in range(100):
                simulation.step()

            observation = hand.observe()

            # Switch back to position control.
            hand.set_position_control(hand.zero_control())

            pos_in_rad = hand.joint_positions_to_control(observation.joint_positions())
            pos_normalized = np.clip(
                hand.normalize_position_control(pos_in_rad), -1.0, 1.0
            )

            # We would expect to reach joint limit
            error = pos_normalized[actuator_index] - selected_force

            assert np.abs(error) < 0.1


def test_autostep():
    mock_sim = mock.MagicMock()
    mock_sim.mj_sim.model.nu = len(ACTUATORS)
    mock_sim.mj_sim.model.actuator_ctrlrange = np.array(
        np.transpose([ACTUATOR_CTRLRANGE_LOWER_BOUND, ACTUATOR_CTRLRANGE_UPPER_BOUND])
    )
    hand = MuJoCoShadowHand(simulation=mock_sim, autostep=True,)

    mock_step = mock.Mock()
    mock_sim.mj_sim.step = mock_step
    hand.set_position_control(hand.zero_control())
    mock_step.assert_called_once()

    mock_step.reset_mock()
    hand.set_effort_control(np.zeros_like(hand.zero_control()))
    mock_step.assert_called_once()
