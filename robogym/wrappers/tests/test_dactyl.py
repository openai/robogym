import numpy as np
from mock import patch

from robogym.envs.dactyl.locked import make_simple_env
from robogym.wrappers.dactyl import FingersOccludedPhasespaceMarkers
from robogym.wrappers.randomizations import RandomizeObservationWrapper


@patch("robogym.wrappers.dactyl.check_occlusion")
def test_fingers_occluded_phasespace_markers(mock_check_occlusion):
    # Test when a finger is marked as occluded, the phasespace fingertip_pos should stay
    # same as the last one.
    fake_is_occluded = [0, 1, 0, 0, 1]
    mock_check_occlusion.return_value = fake_is_occluded

    env = make_simple_env()
    env = RandomizeObservationWrapper(
        env=env, levels={"fingertip_pos": {"uncorrelated": 0.002, "additive": 0.001}}
    )
    env = FingersOccludedPhasespaceMarkers(env=env)

    action_shape = env.unwrapped.action_space.shape
    obs = env.reset()
    fingertip_pos = obs["noisy_fingertip_pos"].reshape(5, 3)
    for _ in range(20):
        obs, _, _, _ = env.step(np.ones(action_shape))
        new_fingertip_pos = obs["noisy_fingertip_pos"].reshape(5, 3)
        for finger_idx in range(5):
            if fake_is_occluded[finger_idx]:
                assert (
                    new_fingertip_pos[finger_idx] == fingertip_pos[finger_idx]
                ).all()
            else:
                assert (
                    new_fingertip_pos[finger_idx] != fingertip_pos[finger_idx]
                ).all()

        fingertip_pos = new_fingertip_pos.copy()
