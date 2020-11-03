import numpy as np

from robogym.envs.rearrange.blocks import make_env
from robogym.wrappers.util import DiscretizeActionWrapper


class TestDiscretizeActionWrapper:
    def test_linear_mapping(self):
        n_bins = 11
        env = make_env(apply_wrappers=False, constants=dict(n_action_bins=n_bins))
        env = DiscretizeActionWrapper(env, n_action_bins=n_bins)
        linear_bins = np.linspace(-1, 1, n_bins)
        assert np.array_equal(
            env._disc_to_cont, [linear_bins] * env.action_space.shape[0]
        )

    def test_exponential_mapping(self):
        n_bins = 11
        env = make_env(
            apply_wrappers=False,
            constants=dict(n_action_bins=n_bins, action_spacing="exponential"),
        )
        env = DiscretizeActionWrapper(
            env, n_action_bins=n_bins, bin_spacing=env.constants.action_spacing
        )
        exp_bins = np.array(
            [-1.0, -0.5, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
        )
        assert np.array_equal(
            env._disc_to_cont, [exp_bins] * env.action_space.shape[0]
        )
