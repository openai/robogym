import numpy as np
import pytest
from numpy.testing import assert_allclose

from robogym.envs.dactyl.full_perpendicular import make_env, make_simple_env
from robogym.utils import rotation


def test_cube_mass():
    env = make_env(constants=dict(randomize=False))
    sim = env.unwrapped.sim
    cube_id = sim.model.body_name2id("cube:middle")

    # The mass of the giiker cube is 90g
    assert_allclose(sim.model.body_subtreemass[cube_id], 0.09, atol=0.005)


@pytest.mark.parametrize(
    "goal_generation",
    [
        "face_curr",
        "face_free",
        "face_cube_solver",
        "unconstrained_cube_solver",
        "full_unconstrained",
        "release_cube_solver",
    ],
)
def test_goal_info(goal_generation):
    constants = {
        "goal_generation": goal_generation,
        "randomize_face_angles": False,
        "randomize": False,
    }

    # There is some small chance that cube can get into invalid state in simulation
    # which will cause cube solver to fail. Fixing the seed here to mitigate this
    # issue.
    env = make_env(constants=constants, starting_seed=12312)
    env.reset()
    _, _, goal_info = env.unwrapped.goal_info()
    assert "goal" in goal_info
    assert "goal_type" in goal_info["goal"]


def test_make_simple_env():
    env = make_simple_env(
        parameters={
            "simulation_params": dict(cube_appearance="vision", hide_target=True)
        }
    )

    env.reset()
    sim = env.sim  # there is no wrapper.
    sticker_geoms = [g for g in sim.model.geom_names if g.startswith("cube:sticker:")]
    assert len(sticker_geoms) == 9 * 6


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


def test_min_episode_length():
    min_steps = 1000
    env_1 = make_env(constants=dict(min_episode_length=min_steps), starting_seed=12312)
    env_1.reset()
    # fix seed to avoid stochastic tests
    num_fallen = 0
    for _ in range(min_steps):
        o, r, d, i = env_1.step(env_1.action_space.sample())
        assert not d
        if i["fell_down"]:
            num_fallen += 1
    assert num_fallen > 0

    env_2 = make_env(constants=dict(min_episode_length=-1), starting_seed=12312)
    env_2.reset()
    # fix seed to avoid stochastic tests
    for t in range(min_steps):
        o, r, d, i = env_2.step(env_2.action_space.sample())
        if d:
            break

    assert t < min_steps - 1
