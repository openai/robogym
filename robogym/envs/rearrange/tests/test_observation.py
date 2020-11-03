import numpy as np
import pytest

from robogym.envs.rearrange.blocks import make_env as make_blocks_env
from robogym.envs.rearrange.blocks_train import make_env as make_train_env
from robogym.envs.rearrange.common.utils import safe_reset_env
from robogym.envs.rearrange.composer import make_env as make_composer_env
from robogym.envs.rearrange.tests.test_rearrange_envs import (
    _list_rearrange_envs,
    is_fixed_goal_env,
    is_fixed_initial_state_env,
)
from robogym.envs.rearrange.ycb import make_env as make_ycb_env
from robogym.robot.robot_interface import ControlMode


def _read_goal_state_from_sim(sim):
    return np.stack(
        [
            sim.data.get_body_xpos(name).copy()
            for name in sim.model.body_names
            if name.startswith("target:object")
            and not name.startswith("target:object:letter")
        ],
        axis=0,
    )


def is_env_without_goal_rot_randomize(env) -> bool:
    return any(
        keyword in str(env.unwrapped.__class__).lower()
        for keyword in ("reach", "attachedblock")
    )


def test_vision_observations():
    for env in _list_rearrange_envs(constants={"vision": True}):
        env.seed(123)
        obs1 = safe_reset_env(env)

        assert "vision_obs" in obs1
        assert "vision_goal" in obs1

        for _ in range(10):
            obs2, _, _, info = env.step(np.ones_like(env.action_space.sample()))

        assert "vision_obs" in obs2
        assert "vision_goal" in obs2
        assert "successes_so_far" in info
        assert not np.allclose(obs1["vision_obs"], obs2["vision_obs"])

        if info["successes_so_far"] == 0:
            # Goal should not have changed.
            assert np.allclose(obs1["vision_goal"], obs2["vision_goal"])

            # check the goal is up-to-date in simulator.
            old_goal = obs1["goal_obj_pos"].copy()
            old_sim_goal = _read_goal_state_from_sim(env.unwrapped.sim)
            assert np.array_equal(old_goal, old_sim_goal)
        else:
            # Unlikely but might happen: We achieved the goal on accident and we might
            # thus have a new goal.
            old_goal = obs2["goal_obj_pos"].copy()
            assert not np.allclose(obs1["vision_goal"], obs2["vision_goal"])

        # Reset the goal and ensure that the goal vision observation has indeed changed.
        env.reset_goal()
        obs3, _, _, _ = env.step(np.zeros_like(env.action_space.sample()))
        assert "vision_obs" in obs3
        assert "vision_goal" in obs3

        assert is_fixed_goal_env(env) or not np.allclose(
            obs2["vision_goal"], obs3["vision_goal"]
        )

        # check the new goal should take effect in simulator.
        new_goal = obs3["goal_obj_pos"].copy()
        new_sim_goal = _read_goal_state_from_sim(env.unwrapped.sim)
        assert is_fixed_goal_env(env) or not np.array_equal(new_goal, old_goal)
        assert np.array_equal(new_goal, new_sim_goal)


@pytest.mark.parametrize("randomize_goal_rot", [True, False])
def test_randomize_goal_rotation(randomize_goal_rot):
    make_env_args = {
        "starting_seed": 0,
        "constants": {"goal_args": {"randomize_goal_rot": randomize_goal_rot}},
        "parameters": {
            "n_random_initial_steps": 0,
            "simulation_params": {"num_objects": 1},
        },
    }
    for make_env in [make_blocks_env, make_train_env]:
        env = make_env(**make_env_args)
        obs = safe_reset_env(env)

        if randomize_goal_rot:
            assert not np.allclose(obs["obj_rot"], obs["goal_obj_rot"], atol=1e-3)
        else:
            # If we do not randomize goal rotation, it is same as object rotation by default.
            assert np.allclose(obs["obj_rot"], obs["goal_obj_rot"], atol=1e-3)
            assert np.allclose(
                obs["rel_goal_obj_rot"],
                np.zeros(obs["rel_goal_obj_rot"].shape),
                atol=1e-3,
            )


@pytest.mark.parametrize("n_random_initial_steps", [0, 5])
@pytest.mark.parametrize("randomize_goal_rot", [True, False])
def test_observations_change_between_resets(n_random_initial_steps, randomize_goal_rot):
    # Ignore keys which are mostly static, and may or may not differ between resets.
    skip_keys = {
        "is_goal_achieved",
        "rel_goal_obj_rot",
        "obj_bbox_size",
        # gripper initial control (should be zero)
        "gripper_controls",
        # robogym.envs.rearrange.common.utils.stabilize_objects is not perfect to ensure zero
        # initial velocity of every object
        "obj_vel_pos",
        "obj_vel_rot",
        # gripper data
        "gripper_qpos",
        "gripper_vel",
        "gripper_pos",
        "gripper_velp",
        "obj_gripper_contact",
        # robot_joint_pos has small perturbation even when n_random_initial_steps=0
        "robot_joint_pos",
        # safety stop obs will be False most of the time, and tcp_force/torque may not change
        # when the robot is not interacting with objects
        "safety_stop",
        "tcp_force",
        "tcp_torque",
    }

    static_keys = {"action_ema"}
    static_keys = frozenset(static_keys)
    skip_keys = frozenset(skip_keys)

    make_env_args = dict(
        constants={"goal_args": {"randomize_goal_rot": randomize_goal_rot}},
        parameters={"n_random_initial_steps": n_random_initial_steps},
        starting_seed=1,  # seed 0 failed for ReachEnv
    )
    for env in _list_rearrange_envs(**make_env_args):
        # Fixed goal envs are taken care of in the test below.
        if is_fixed_goal_env(env):
            continue

        extra_skip_keys = set()
        if is_env_without_goal_rot_randomize(env):
            extra_skip_keys.add("goal_obj_rot")

        extra_static_keys = set()
        if is_fixed_initial_state_env(env):
            # Unreliable gripper make obj_rel_pos also unreliable.
            extra_skip_keys.add("obj_rel_pos")
            extra_static_keys.update(["obj_pos", "obj_rot"])

            if not randomize_goal_rot:
                extra_static_keys.add("goal_obj_rot")

            if n_random_initial_steps == 0:
                extra_static_keys.add("qpos")

        if "block" in env.unwrapped.__class__.__name__.lower():
            extra_skip_keys.add("obj_colors")

        obs1 = safe_reset_env(env)
        obs2 = safe_reset_env(env)

        assert set(obs1.keys()) == set(obs2.keys())

        for key in obs1:
            if key in skip_keys or key in extra_skip_keys:
                continue
            elif key in static_keys or key in extra_static_keys:
                assert np.allclose(obs1[key], obs2[key], atol=5e-3), (
                    f"Observations in {env.unwrapped.__class__.__name__} changed "
                    f"between resets for {key}: {obs1[key]}"
                )
            else:
                assert not np.allclose(obs1[key], obs2[key], atol=1e-3), (
                    f"Observations in {env.unwrapped.__class__.__name__} unchanged "
                    f"between resets for {key}: {obs1[key]}"
                )


def test_static_observations_for_fixed_goal_envs():
    static_keys = set(["goal_obj_pos", "goal_obj_rot", "robot_joint_pos", "qpos_goal"])

    for env in _list_rearrange_envs(
        constants={"goal_args": {"randomize_goal_rot": False}},
        parameters={"n_random_initial_steps": 0},
        starting_seed=0,
    ):

        # We are only interested in fixed-goal envs here.
        # Goals for Chessboard are currently non-deterministic, likely due to physics
        # stability issues. Remove the check below once physics issues are resolved.
        if (
            not is_fixed_goal_env(env)
            or env.unwrapped.__class__.__name__ == "ChessboardRearrangeEnv"
        ):
            continue

        obs1 = env.reset()
        obs2 = env.reset()

        for key in static_keys:
            assert np.allclose(obs1[key], obs2[key], atol=1e-2), (
                f"Observations in {env.unwrapped.__class__.__name__} changed "
                f"between resets for {key}: {obs1[key]}"
            )


def test_observation_static_within_step():
    for env in _list_rearrange_envs():
        obs = env.reset()

        for _ in range(5):
            new_obs = env.observe()
            for key in new_obs:
                assert np.allclose(obs[key], new_obs[key])


def test_observations_change_over_time(n_resets=3, n_trials=3):
    static_keys = frozenset(
        (
            "obj_pos",
            "obj_rel_pos",
            "obj_vel_pos",
            "obj_rot",
            "obj_vel_rot",
            "obj_bbox_size",
            "goal_obj_pos",
            "goal_obj_rot",
            "qpos_goal",
            "is_goal_achieved",
            "gripper_controls",
            "gripper_qpos",
            "gripper_vel",
            "rel_goal_obj_pos",
            "rel_goal_obj_rot",
            "obj_gripper_contact",
            "obj_colors",
            "safety_stop",
        )
    )
    for env in _list_rearrange_envs():
        for reset_i in range(n_resets):
            obs = safe_reset_env(env)

            for i in range(n_trials):
                last_obs = obs

                # Apply 10 random steps.
                for _ in range(10):
                    obs, _, _, _ = env.step(env.action_space.sample())

                for key in obs:
                    if key not in static_keys:
                        assert not np.allclose(obs[key], last_obs[key]), (
                            f"Observations unchanged between steps for {key} at reset {reset_i} "
                            f"and trial {i}: {obs[key]}"
                        )


def test_observation_deterministic_with_fixed_seed():
    batch1 = _list_rearrange_envs(starting_seed=0)
    batch2 = _list_rearrange_envs(starting_seed=0)

    for env1, env2 in zip(batch1, batch2):
        if "Composer" in env1.unwrapped.__class__.__name__:
            # FIXME: ComposerRearrangeSim.make_object_xml cannot be controlled by starting seed.
            continue

        obs1 = env1.reset()
        obs2 = env2.reset()

        assert set(obs1.keys()) == set(obs2.keys())

        for key in obs1:
            assert np.allclose(
                obs1[key], obs2[key], atol=1e-5
            ), f"{env1.unwrapped.__class__.__name__} failed on key {key}"


def test_info_obs_consistency():
    for env in _list_rearrange_envs():
        if "ReachEnv" in str(env.unwrapped):
            # obs and info['current_state'] consistency does not hold for ReachEnv because
            # ReachEnv uses `gripper_pos` as `obj_pos` for checking goals. Therefore be sure not
            # to use `current_state` instead of `obs` for ReachEnv.
            continue
        obs1 = env.reset()
        info1 = env.unwrapped.get_info()
        for _ in range(5):
            obs2 = env.observe()
            info2 = env.unwrapped.get_info()
            for key in ["obj_pos", "obj_rot"]:
                assert np.allclose(
                    obs1[key], info1["current_state"][key]
                ), f"key: {key}"
                assert np.allclose(
                    obs1[key], info2["current_state"][key]
                ), f"key: {key}"
                assert np.allclose(obs1[key], obs2[key]), f"key: {key}"

            obs1, _, _, info1 = env.step(np.ones_like(env.action_space.sample()))


def test_observation_with_default_sim_randomization():
    test_seed = 2
    env1 = make_blocks_env(
        starting_seed=test_seed, parameters=dict(n_random_initial_steps=0)
    )
    env1.unwrapped.randomization.simulation_randomizer.disable()
    env1.seed(test_seed)
    env1.reset()
    env1.seed(test_seed)
    env1.reset_goal()
    env1.action_space.seed(test_seed)

    # This feels like an excessive amount of fixing the seed, but it appears that enabling the
    # default simulation randomizer affects random number generation to some extent, so all of the
    # seed-fixing below is necessary.
    env2 = make_blocks_env(
        starting_seed=test_seed, parameters=dict(n_random_initial_steps=0)
    )
    env2.seed(test_seed)
    env2.reset()
    env2.seed(test_seed)
    env2.reset_goal()
    env2.action_space.seed(test_seed)

    # Step such that two envs have same initial and goal states. The
    # test is to make sure sim randomization by default shoudn't change env
    # behavior.
    for _ in range(50):
        obs1 = env1.step(env1.action_space.sample())[0]
        obs2 = env2.step(env2.action_space.sample())[0]
        assert set(obs1.keys()) == set(obs2.keys())
        for key in obs1:
            assert np.allclose(
                obs1[key], obs2[key], atol=1e-4
            ), f"{env1.unwrapped.__class__.__name__} failed on key {key}"


@pytest.mark.parametrize(
    "control_mode, action_dim, gripper_dim, robot_dim, robot_joint_pos",
    [
        (ControlMode.TCP_ROLL_YAW.value, 6, 1, 8, 6),
        (ControlMode.TCP_WRIST.value, 5, 1, 8, 6),
    ],
)
def test_observation_space(
    control_mode, action_dim, gripper_dim, robot_dim, robot_joint_pos
):
    def _get_expected_obs_space_shape(num_objects):
        return {
            "goal_obj_pos": (num_objects, 3,),
            "gripper_pos": (3,),
            "gripper_qpos": (gripper_dim,),
            "gripper_vel": (gripper_dim,),
            "gripper_velp": (3,),
            "obj_pos": (num_objects, 3,),
            "obj_rel_pos": (num_objects, 3,),
            "obj_rot": (num_objects, 3,),
            "obj_vel_pos": (num_objects, 3,),
            "obj_vel_rot": (num_objects, 3,),
            "qpos": (robot_dim + 7 * num_objects,),
            "qpos_goal": (robot_dim + 7 * num_objects,),
            "robot_joint_pos": (robot_joint_pos,),
            "safety_stop": (1,),
            "tcp_force": (3,),
            "tcp_torque": (3,),
        }

    env_args = dict(
        parameters=dict(robot_control_params=dict(control_mode=control_mode)),
    )
    for env in _list_rearrange_envs(**env_args):
        num_objects = env.unwrapped.mujoco_simulation.num_objects
        ob_shapes = _get_expected_obs_space_shape(num_objects)

        for key, ref_shape in ob_shapes.items():
            assert (
                key in env.observation_space.spaces
            ), f"{key} not in observation_space"
            curr_shape = env.observation_space.spaces[key].shape
            assert (
                curr_shape == ref_shape
            ), f"{key} has wrong shape: is {curr_shape}, expected {ref_shape}"

        obs = safe_reset_env(env)

        for key, ref_shape in ob_shapes.items():
            assert key in obs, f"{key} not in observation_space"
            curr_shape = obs[key].shape
            assert (
                curr_shape == ref_shape
            ), f"{key} has wrong shape: is {curr_shape}, expected {ref_shape}"


@pytest.mark.parametrize("vision", [True, False])
def test_goals_differ_from_obs(vision):
    make_env_args = {
        "constants": {"vision": vision, "goal_args": {"randomize_goal_rot": True}},
        "parameters": {
            "simulation_params": {
                "num_objects": 3,
                "goal_rot_weight": 1.0,
                "goal_distance_ratio": 1.0,
            }
        },
    }
    for make_env in [make_blocks_env, make_train_env, make_ycb_env, make_composer_env]:
        env = make_env(**make_env_args)

        obs = safe_reset_env(env)
        assert not np.allclose(obs["obj_pos"], obs["goal_obj_pos"])
        assert not np.allclose(obs["obj_rot"], obs["goal_obj_rot"])

        obs, _, _, _ = env.step(env.action_space.sample())
        assert not np.allclose(obs["obj_pos"], obs["goal_obj_pos"])
        assert not np.allclose(obs["obj_rot"], obs["goal_obj_rot"])

        obs = safe_reset_env(env, only_reset_goal=True)
        assert not np.allclose(obs["obj_pos"], obs["goal_obj_pos"])
        assert not np.allclose(obs["obj_rot"], obs["goal_obj_rot"])


def test_object_gripper_contact():
    env = make_blocks_env(
        parameters={
            "simulation_params": {"num_objects": 3},
            "n_random_initial_steps": 0,
        }
    )
    env.reset()
    contact = env.unwrapped.mujoco_simulation.get_object_gripper_contact()
    assert np.array_equal(contact, np.zeros((3, 2)))
