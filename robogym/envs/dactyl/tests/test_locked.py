import numpy as np
from mujoco_py import ignore_mujoco_warnings
from numpy.testing import assert_allclose

from robogym.envs.dactyl.common.cube_utils import on_palm
from robogym.envs.dactyl.locked import make_env, make_simple_env
from robogym.utils import rotation


def test_locked_cube():
    env = make_env(starting_seed=0)

    is_on_palm = []
    for idx in range(20):
        env.reset()

        expected_joints = (
            "cube:cube_tx",
            "cube:cube_ty",
            "cube:cube_tz",
            "cube:cube_rot",
            "target:cube_tx",
            "target:cube_ty",
            "target:cube_tz",
            "target:cube_rot",
            "robot0:WRJ1",
            "robot0:WRJ0",
            "robot0:FFJ3",
            "robot0:FFJ2",
            "robot0:FFJ1",
            "robot0:FFJ0",
            "robot0:MFJ3",
            "robot0:MFJ2",
            "robot0:MFJ1",
            "robot0:MFJ0",
            "robot0:RFJ3",
            "robot0:RFJ2",
            "robot0:RFJ1",
            "robot0:RFJ0",
            "robot0:LFJ4",
            "robot0:LFJ3",
            "robot0:LFJ2",
            "robot0:LFJ1",
            "robot0:LFJ0",
            "robot0:THJ4",
            "robot0:THJ3",
            "robot0:THJ2",
            "robot0:THJ1",
            "robot0:THJ0",
        )

        assert env.unwrapped.sim.model.joint_names == expected_joints
        with ignore_mujoco_warnings():
            for _ in range(20):
                obs, _, _, _ = env.step(env.action_space.nvec // 2)

        is_on_palm.append(on_palm(env.unwrapped.sim))

        # Make sure the mass is right.
        cube_body_idx = env.unwrapped.sim.model.body_name2id("cube:middle")
        assert_allclose(
            env.unwrapped.sim.model.body_subtreemass[cube_body_idx], 0.078, atol=1e-3
        )

    assert (
        np.mean(is_on_palm) >= 0.8
    ), "Cube should stay in hand (most of the time) when zero action is sent."


def test_observe():
    # Test observation matches simulation state.
    env = make_simple_env()
    env.reset()
    simulation = env.mujoco_simulation

    obs = env.observe()

    qpos = simulation.qpos
    qpos[simulation.qpos_idxs["target_all_joints"]] = 0.0
    qvel = simulation.qvel
    qvel[simulation.qvel_idxs["target_all_joints"]] = 0.0

    true_obs = {
        "cube_pos": simulation.get_qpos("cube_position"),
        "cube_quat": rotation.quat_normalize(simulation.get_qpos("cube_rotation")),
        "hand_angle": simulation.get_qpos("hand_angle"),
        "fingertip_pos": simulation.shadow_hand.observe()
        .fingertip_positions()
        .flatten(),
        "qpos": qpos,
        "qvel": qvel,
    }

    for obs_key, true_val in true_obs.items():
        assert np.allclose(
            obs[obs_key], true_val
        ), f"Value for obs {obs_key} {obs[obs_key]} doesn't match true value {true_val}."


def test_informative_obs():
    WHITELIST = [
        # The position of the goal is zeroed
        "relative_goal_pos",
        "noisy_relative_goal_pos",
        "goal_pos",
        # Not all episodes end with a fall, i.e. it might be all zeros
        "fell_down",
        "is_goal_achieved",
    ]

    env = make_env(constants=dict(randomize=False, max_timesteps_per_goal=50))
    obs = env.reset()
    done = False
    all_obs = [obs]
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        all_obs.append(obs)
    all_obs.append(env.reset())  # one more reset at the end

    # Collect all obs and index by key.
    keys = set(all_obs[0].keys())
    assert len(keys) > 0
    combined_obs_by_keys = {key: [] for key in keys}
    for obs in all_obs:
        assert set(obs.keys()) == keys
        for key in keys:
            combined_obs_by_keys[key].append(obs[key])

    # Make sure that none of the keys has all-constant obs.
    for key, obs in combined_obs_by_keys.items():
        assert len(obs) == len(all_obs)
        if key in WHITELIST:
            continue

        obs0 = obs[0]
        equals = [np.array_equal(obs0, obs_i) for obs_i in obs]
        # If ob0 is equal to all other obs, all obs are equal, i.e. the observation
        # contains no information whatsoever. This is usually bad (e.g. we had an issue
        # in the past where qpos was aways set to all-zeros).
        assert not np.all(equals), "observations for {} are all equal to {}".format(
            key, obs0
        )


def test_relative_action():
    for relative_action in [True, False]:
        # NOTE - this seed choice is very important, since starting states affect the test.
        # The test could be updated to be robust in the future.
        env = make_env(
            starting_seed=586895, constants={"relative_action": relative_action}
        )
        env.reset()
        zeros = np.zeros(env.unwrapped.sim.model.nu)
        not_zeros = np.ones(env.unwrapped.sim.model.nu) * 0.5
        num_robot_joints = len(
            [x for x in env.unwrapped.sim.model.joint_names if "robot0" in x]
        )
        qpos_shape = env.unwrapped.sim.data.qpos.shape
        num_cube_joints = qpos_shape[0] - num_robot_joints

        for action in [zeros, not_zeros]:
            env.unwrapped.sim.data.qvel[:] = 0
            env.unwrapped.sim.data.qpos[:] = np.random.randn(*qpos_shape) * 0.1
            env.unwrapped.sim.data.qpos[:num_cube_joints] = -10.0
            for _ in range(10):
                env.unwrapped.step(action)
            qvel = np.sum(np.square(env.unwrapped.sim.data.qvel[num_cube_joints:]))
            if (action == zeros).all() and relative_action:
                assert qvel < 0.09
            else:
                assert qvel > 0.09


def helper_test_two_deterministic_envs(env1, env2):
    env1.reset()
    env2.reset()

    env1.unwrapped.reset_goal()
    env2.unwrapped.reset_goal()

    action = env2.action_space.sample()

    env1_obs, env1_reward = env1.step(action)[:2]
    env2_obs, env2_reward = env2.step(action)[:2]

    for key in env1_obs.keys():
        assert np.all(
            np.isclose(env1_obs[key], env2_obs[key])
        ), "Key: %s -- Diff:\n%s" % (key, env1_obs[key] - env2_obs[key])

    assert np.allclose(env1_reward, env2_reward)


def test_rand_locked_consistent():
    seed = 12345
    helper_test_two_deterministic_envs(
        make_env(starting_seed=seed), make_env(starting_seed=seed)
    )


def test_det_locked_consistent():
    seed = 12345
    helper_test_two_deterministic_envs(
        make_env(constants=dict(randomize=False), starting_seed=seed),
        make_env(constants=dict(randomize=False), starting_seed=seed),
    )
