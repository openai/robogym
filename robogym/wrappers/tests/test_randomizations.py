import numpy as np
import pytest
from mock import patch
from numpy.testing import assert_almost_equal

from robogym.envs.dactyl.full_perpendicular import make_simple_env
from robogym.envs.dactyl.locked import make_env as make_env_locked
from robogym.envs.dactyl.reach import make_simple_env as make_reach_env
from robogym.mujoco.helpers import joint_qpos_ids_from_prefix
from robogym.utils import rotation
from robogym.utils.dactyl_utils import actuated_joint_range
from robogym.wrappers.dactyl import (
    FingersFreezingPhasespaceMarkers,
    FingersOccludedPhasespaceMarkers,
    RandomizedPhasespaceFingersWrapper,
    RandomizedRobotDampingWrapper,
    RandomizedRobotKpWrapper,
)
from robogym.wrappers.randomizations import QUAT_NOISE_CORRECTION  # noqa
from robogym.wrappers.randomizations import (
    ActionDelayWrapper,
    ActionNoiseWrapper,
    BacklashWrapper,
    ObservationDelayWrapper,
    RandomizedActionLatency,
    RandomizedBrokenActuatorWrapper,
    RandomizedCubeFrictionWrapper,
    RandomizedGravityWrapper,
    RandomizedJointLimitWrapper,
    RandomizedRobotFrictionWrapper,
    RandomizedTendonRangeWrapper,
    RandomizedTimestepWrapper,
    RandomizeObservationWrapper,
)

VISUALIZE = False


def test_wrapper_divergence():
    """
    This test run the same action in the vanilla dactyl_locked env and the one that is wrapped in
    a given wrappers. After some steps, the wrapped env should diverge from the vanilla version.
    """
    env_kwargs = {
        "n_random_initial_steps": 0,
    }

    simple_env = make_simple_env(parameters=env_kwargs, starting_seed=0)
    dummy_env = make_simple_env(
        parameters=env_kwargs, starting_seed=0
    )  # should be exact same as `simple_env`

    # Add you wrappers here!
    wrappers_to_test = [
        (ActionNoiseWrapper, {}),
        (BacklashWrapper, {}),
        (FingersOccludedPhasespaceMarkers, {}),  # Need 'noisy_fingertip_pos'
        (FingersFreezingPhasespaceMarkers, {}),  # Need 'noisy_fingertip_pos'
        (
            RandomizedBrokenActuatorWrapper,
            {
                "proba_broken": 1.0,  # force one broken actuators
                "max_broken_actuators": 1,
            },
        ),
        (RandomizedRobotFrictionWrapper, {}),
        (RandomizedCubeFrictionWrapper, {}),
        (RandomizedGravityWrapper, {}),
        (RandomizedJointLimitWrapper, {}),
        (RandomizedTendonRangeWrapper, {}),
        (RandomizedPhasespaceFingersWrapper, {}),
        (RandomizedRobotDampingWrapper, {}),
        (RandomizedRobotKpWrapper, {}),
        (RandomizedTimestepWrapper, {}),
        (ActionDelayWrapper, {}),
        # With default args, the maximum qpos difference is too small.
        (RandomizedActionLatency, {"max_delay": 2}),  # default 1
        # (RandomizedBodyInertiaWrapper, {}),  # default mass_range=[0.5, 1.5]
    ]

    wrapped_envs = []
    for wrapper_class, kwargs in wrappers_to_test:
        env = make_simple_env(parameters=env_kwargs, starting_seed=0)

        if wrapper_class in (
            FingersOccludedPhasespaceMarkers,
            FingersFreezingPhasespaceMarkers,
        ):
            env = RandomizeObservationWrapper(
                env=env,
                levels={"fingertip_pos": {"uncorrelated": 0.002, "additive": 0.001}},
            )

        env = wrapper_class(env=env, **kwargs)
        env.reset()
        wrapped_envs.append(env)

    for i in range(200):
        action = np.ones(env.action_space.shape)
        simple_env.step(action)
        dummy_env.step(action)
        for env in wrapped_envs:
            env.step(action)

    target_qpos_idxs = joint_qpos_ids_from_prefix(
        simple_env.unwrapped.sim.model, "target:"
    )
    kept_indices = set(range(simple_env.unwrapped.sim.data.qpos.shape[0])) - set(
        target_qpos_idxs
    )
    kept_indices = sorted(kept_indices)

    def get_non_target_qpos(_env):
        return np.array(_env.unwrapped.sim.data.qpos.copy()[kept_indices])

    # Make sure the base env is deterministic
    assert np.array_equal(
        get_non_target_qpos(simple_env), get_non_target_qpos(dummy_env)
    )

    for env in wrapped_envs:
        diffs = np.absolute(get_non_target_qpos(simple_env) - get_non_target_qpos(env))
        assert np.max(diffs) > 1e-4, "failed for {}".format(env.__class__.__name__)
        assert np.min(diffs) > 0.0, "failed for {}".format(env.__class__.__name__)


def test_randomize_obs_wrapper():
    state = np.random.get_state()
    try:
        np.random.seed(1)
        quat_noise_factor = QUAT_NOISE_CORRECTION

        # test that randomization of Euler angles and quaternions has same distance
        n = 10000
        a_bias = 0.1
        additive_bias = a_bias * np.random.standard_normal(size=(n, 3))
        # multiplicative bias does not make sense for random angles

        angle = np.random.uniform(-np.pi, np.pi, size=(n, 3))
        new_angle = angle + additive_bias

        angle_dist = np.linalg.norm(rotation.subtract_euler(new_angle, angle), axis=-1)

        angle = np.random.uniform(-np.pi, np.pi, size=(n, 1))
        axis = np.random.uniform(-1.0, 1.0, size=(n, 3))
        quat = rotation.quat_from_angle_and_axis(angle, axis)

        # double the additive bias to roughly equal the angular distance
        noise_angle = a_bias * quat_noise_factor * np.random.standard_normal(size=(n,))
        noise_axis = np.random.uniform(-1.0, 1.0, size=(n, 3))
        noise_quat = rotation.quat_from_angle_and_axis(noise_angle, noise_axis)

        new_quat = rotation.quat_mul(quat, noise_quat)
        quat_diff = rotation.quat_difference(quat, new_quat)
        quat_dist = rotation.quat_magnitude(quat_diff)

        mean_angle_dist = np.mean(angle_dist)
        mean_quat_dist = np.mean(quat_dist)

        assert ((mean_angle_dist - mean_quat_dist) / mean_angle_dist) < 0.01

    finally:
        np.random.set_state(state)


def test_randomize_observation_wrapper():
    simple_env = make_simple_env()
    simple_env.reset()

    env = RandomizeObservationWrapper(
        env=simple_env, levels={"cube_pos": {"uncorrelated": 0.2, "additive": 0.1}}
    )

    with patch.object(env, "random_state") as mock_rand:
        # Remove randomness in the noise.
        mock_rand.randn.side_effect = lambda key_length: np.ones(
            key_length, dtype=np.float32
        )

        def mock_obs(o):
            simple_env.observe = lambda: o

        mock_obs({"cube_pos": np.array([0.1, 0.2, 0.3])})

        obs = env.reset()

        # Make sure noise is applied on noiseless value.
        assert_almost_equal(obs["noisy_cube_pos"], [0.4, 0.5, 0.6])

        mock_obs(
            {
                "cube_pos": np.array([0.1, 0.2, 0.3]),
                "noisy_cube_pos": np.array([0.2, 0.3, 0.4]),
            }
        )

        # Make sure noise is applied on top of noisy observation when available.
        obs = env.reset()
        assert_almost_equal(obs["noisy_cube_pos"], [0.5, 0.6, 0.7])


def test_observation_delay_wrapper():
    levels = {
        "interpolators": {
            "cube_quat": "QuatInterpolator",
            "cube_face_angle": "RadianInterpolator",
        },
        "groups": {
            "vision": {
                "obs_names": ["cube_pos", "cube_quat"],
                "mean": 1.5,
                "std": 0.0,
            },
            "giiker": {"obs_names": ["cube_face_angle"], "mean": 1.4, "std": 0.0},
            "phasespace": {"obs_names": ["fingertip_pos"], "mean": 1.2, "std": 0.0},
        },
    }

    simple_env = make_simple_env()
    simple_env.reset()

    env = ObservationDelayWrapper(simple_env, levels)

    def mock_obs(o):
        simple_env.observe = lambda: o

    initial_obs = {
        "cube_pos": np.array([0.1, 0.2, 0.3]),
        "cube_quat": rotation.euler2quat(np.array([0.0, 0.0, 0.0])),
        "cube_face_angle": np.array(
            [np.pi - 0.01, np.pi / 2 - 0.01, 0.0, 0.0, 0.0, 0.0]
        ),
        "fingertip_pos": np.array([0.5, 0.6, 0.7]),
    }

    mock_obs(initial_obs)

    env.reset()

    second_obs = {
        "cube_pos": np.array([0.2, 0.3, 0.4]),
        "cube_quat": rotation.euler2quat(np.array([0.8, 0.0, 0.0])),
        "cube_face_angle": np.array(
            [-np.pi + 0.01, np.pi / 2 + 0.01, 0.0, 0.0, 0.0, 0.0]
        ),
        "fingertip_pos": np.array([0.5, 0.6, 0.7]),
    }

    mock_obs(second_obs)

    obs = env.step(np.zeros(env.action_space.shape))[0]

    # Should take the first observation because there are only two observations and nothing
    # to interpolate.
    for key in initial_obs:
        assert_almost_equal(obs[f"noisy_{key}"], initial_obs[key])

    # Step env again so obs should be interpolation of initial and second obs.
    obs = env.step(np.zeros(env.action_space.shape))[0]

    assert_almost_equal(obs["noisy_cube_pos"], [0.15, 0.25, 0.35])
    assert_almost_equal(rotation.quat2euler(obs["noisy_cube_quat"]), [0.4, 0.0, 0.0])
    assert_almost_equal(
        obs["noisy_cube_face_angle"],
        [-np.pi + 0.002, np.pi / 2 + 0.002, 0.0, 0.0, 0.0, 0.0],
    )
    assert_almost_equal(obs["noisy_fingertip_pos"], [0.5, 0.6, 0.7])


def test_observation_wrapper_order():
    # Test to make sure observation noise wrappers are applied in correct order.
    simple_env = make_simple_env()
    simple_env.reset()
    simple_env.observe = lambda: {"cube_pos": np.array([0.1, 0.2, 0.3])}

    env = RandomizeObservationWrapper(
        env=simple_env, levels={"cube_pos": {"uncorrelated": 0.2, "additive": 0.1}}
    )

    env.reset()

    env = ObservationDelayWrapper(
        env,
        levels={
            "interpolators": {},
            "groups": {
                "vision": {"obs_names": ["cube_pos"], "mean": 1.5, "std": 0.0},
            },
        },
    )

    with pytest.raises(AssertionError):
        env.step(np.zeros(env.action_space.shape))


@pytest.mark.skip(reason="This test needs to be updated to work properly.")
def test_randomized_joint_range_wrapper_subset():
    selected = [
        "robot0:WRJ1",
        "robot0:FFJ2",
        "robot0:FFJ1",
        "robot0:FFJ0",
        "robot0:MFJ1",
        "robot0:MFJ0",
        "robot0:THJ2",
        "robot0:THJ0",
    ]

    env0 = make_reach_env()
    env0.reset()
    orig_sim_limits = actuated_joint_range(env0.unwrapped.sim)

    env1 = make_reach_env()
    env1 = RandomizedJointLimitWrapper(env=env1, joint_names=selected, relative_std=0.3)
    env1.reset()

    for _ in range(5):
        env1.reset()
        rand_sim_limits = actuated_joint_range(env1.unwrapped.sim)

        for i, jnt_name in enumerate(env1.unwrapped.sim.model.joint_names):
            low, high = orig_sim_limits[i]

            if jnt_name not in selected:
                assert low == rand_sim_limits[i][0] and high == rand_sim_limits[i][1]
            else:
                assert (low != 0.0) or rand_sim_limits[i][0] >= 0.0
                assert (high != 0.0) or rand_sim_limits[i][1] <= 0.0


def test_randomized_broken_actuator_wrapper():
    env = make_simple_env()
    env.reset()

    env = RandomizedBrokenActuatorWrapper(
        env=env, proba_broken=0.5, max_broken_actuators=4, uncorrelated=0.0
    )
    env.reset()
    assert len(env._broken_aids) <= 4

    # The broken actuators are different after reset.
    orig_broken_aids = env._broken_aids.copy()
    env.reset()
    assert sorted(env._broken_aids) != sorted(orig_broken_aids)

    # The action is modified
    action = env.action(np.ones(env.action_space.shape)).copy()
    for i in range(env.action_space.shape[0]):
        if i in env._broken_aids:
            assert action[i] == 0.0
        else:
            assert action[i] == 1.0


def test_replace_cube_obs_vision_wrapper():
    # Disabled for now until new models are trained
    vision_args = {
        "vision_model_path": "projects/vision/experiments/gan-muj-100x100/20180109_18_41/",
    }

    env = make_env_locked(constants={"randomize": False, "vision_args": vision_args})

    env.reset()
    env.step(env.action_space.nvec // 2)


def test_action_delay_wrapper_inactive():
    env = make_simple_env(starting_seed=0)
    env.reset()

    # Wrapper calls reset in its __init__ so no need to
    # call reset explicitly.
    delayed_env = ActionDelayWrapper(
        make_simple_env(starting_seed=0),
        delay=0.0,
        per_episode_std=0.0,
        per_step_std=0.0,
        random_state=np.random.RandomState(),
    )

    action = env.action_space.sample()
    for _ in range(20):
        ob_env, _, _, _ = env.step(action)
        ob_delayed_env, _, _, _ = delayed_env.step(action)

    for name in ob_env:
        assert (
            np.mean(np.abs(ob_env[name] - ob_delayed_env[name])) < 1e-6
        ), "ActionDelayWrapper should be inactive."
