from unittest import TestCase

import numpy as np

from robogym.envs.rearrange.blocks import make_env
from robogym.utils.testing import assert_dict_match
from robogym.wrappers.util import DiscretizeActionWrapper


class TestMultiGoalsEnv(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.env = self._create_env(
            starting_seed=0,
            constants={
                "max_timesteps_per_goal_per_obj": 5,
                "successes_needed": 5,
                "success_reward": 100.0,
                "success_pause_range_s": (0, 0),
            },
            parameters={"simulation_params": {"num_objects": 2}},
        )

        assert self.env.unwrapped.multi_goal_tracker.max_timesteps_per_goal == 10
        assert self.env.unwrapped.constants.max_timesteps_per_goal == 10
        self.zero_action = np.zeros(self.env.action_space.shape)

    def _create_env(self, **env_params):
        # Create a test env with 2 blocks and position-only goals.
        env = make_env(**env_params)
        assert isinstance(env, DiscretizeActionWrapper)

        env = env.env  # remove discretized action so that we can use zero action.
        env.reset()

        return env

    def test_basic_info(self):
        assert_dict_match(self.env.unwrapped.get_info(), self.env.unwrapped.get_info())

        for step in range(9):
            _, reward, done, info = self.env.step(self.env.action_space.sample())
            assert_dict_match(info, self.env.unwrapped.get_info())
            assert_dict_match(
                self.env.unwrapped.get_info(), self.env.unwrapped.get_info()
            )

            assert not done
            assert list(reward) == [0.0] * 3

            assert info["goals_so_far"] == 1
            assert info["goal_reachable"]
            assert not info["goal_terminally_unreachable"]
            assert not info["solved"]

            assert not info["sub_goal_is_successful"]
            assert not info["trial_success"]

            assert info["steps_since_last_goal"] == step + 1
            assert info["sub_goal_type"] == "generic"
            assert info["steps_by_goal_type"] == {"generic": step + 1}
            assert info["successes_so_far"] == 0
            assert info["successes_so_far_by_goal_type"] == {"generic": 0}
            assert not info["env_crash"]

            assert info["steps_per_success"] == 10
            assert info["steps_per_success_by_goal_type"] == {"generic": 10}

        _, _, done, _ = self.env.step(self.env.action_space.sample())
        assert done

    def _fake_one_success(self):
        obj_pos = self.env.unwrapped._goal["obj_pos"]
        self.env.unwrapped.mujoco_simulation.set_object_pos(obj_pos)
        self.env.unwrapped.mujoco_simulation.forward()

    def test_reset(self):
        for _ in range(5):
            for _ in range(5):
                self.env.step(self.zero_action)

            self._fake_one_success()
            assert self.env.step(self.zero_action)[-1]["sub_goal_is_successful"]

        self.env.reset()
        info = self.env.step(self.zero_action)[-1]
        assert info["goals_so_far"] == 1
        assert info["steps_since_last_goal"] == 1
        assert info["successes_so_far"] == 0
        assert info["successes_so_far_by_goal_type"] == {"generic": 0}
        assert info["steps_per_success"] == 10
        assert info["steps_per_success_by_goal_type"] == {"generic": 10}

    def test_multi_successes_full(self):
        for goal_idx in range(5):
            for step in range(5):
                _, reward, done, info = self.env.step(self.zero_action)
                assert not done
                assert not info["sub_goal_is_successful"]
                assert not info["trial_success"]
                assert info["steps_since_last_goal"] == step + 1
                assert info["successes_so_far"] == goal_idx
                assert info["goals_so_far"] == goal_idx + 1
                assert info["steps_per_success"] == (10 if goal_idx == 0 else 6)

            self._fake_one_success()
            _, reward, done, info = self.env.step(self.zero_action)
            assert reward[0] == 0
            assert reward[-1] == 100.0

            assert info["sub_goal_is_successful"]
            assert info["successes_so_far"] == goal_idx + 1
            assert info["steps_per_success"] == 6
            assert info["steps_per_success_by_goal_type"] == {"generic": 6}

            if goal_idx == 4:
                # When the last goal is successful.
                assert done
                assert info["trial_success"]
                assert info["goals_so_far"] == 5
            else:
                # When the episode should continue with more goals.
                assert not done
                assert not info["trial_success"]
                assert info["goals_so_far"] == goal_idx + 2

    def test_multi_successes_fail(self):
        for goal_idx in range(3):
            # We need 2, 4, 6 steps to achieve one success.
            steps_to_success = (goal_idx + 1) * 2
            for step in range(steps_to_success - 1):
                _, reward, done, info = self.env.step(self.zero_action)
                assert not done

                if goal_idx == 0:
                    expected_steps_per_success = 10
                elif goal_idx == 1:
                    expected_steps_per_success = 2
                else:
                    expected_steps_per_success = (2 + 4) / 2

                assert info["steps_per_success"] == expected_steps_per_success

            self._fake_one_success()
            _, reward, done, info = self.env.step(self.zero_action)
            assert info["sub_goal_is_successful"]

        # failed on the 4th goal.
        for step in range(9):
            _, reward, done, info = self.env.step(self.zero_action)
            assert not done

        _, reward, done, info = self.env.step(self.zero_action)
        assert done
        assert not info["sub_goal_is_successful"]
        assert info["steps_per_success"] == (2 + 4 + 6) / 3
        assert not info["trial_success"]

    def test_max_timesteps(self):
        self.env = self._create_env(
            starting_seed=0,
            constants={
                "max_timesteps_per_goal_per_obj": 50,
                "success_reward": 100.0,
                "successes_needed": 1,
                "success_pause_range_s": (1.0, 1.0),
            },
            parameters={"simulation_params": {"num_objects": 1, "max_num_objects": 8}},
        )

        for num_objects in range(1, 9):
            self.env.unwrapped.randomization.update_parameter(
                "parameters:num_objects", num_objects
            )
            self.env.reset()
            assert self.env.unwrapped.mujoco_simulation.num_objects == num_objects
            assert (
                self.env.unwrapped.multi_goal_tracker.max_timesteps_per_goal
                == 50 * num_objects
            )
            assert (
                self.env.unwrapped.constants.max_timesteps_per_goal == 50 * num_objects
            )

    def test_consecutive_success_steps_required(self):
        self.env = self._create_env(
            starting_seed=0,
            constants={
                "max_timesteps_per_goal_per_obj": 50,
                "success_reward": 100.0,
                "successes_needed": 1,
                "success_pause_range_s": (1.0, 1.0),
            },
            parameters={"simulation_params": {"num_objects": 2}},
        )

        mj_sim = self.env.unwrapped.mujoco_simulation.mj_sim
        step_duration_s = mj_sim.nsubsteps * mj_sim.model.opt.timestep
        success_steps_required = int(1.0 / step_duration_s)
        assert (
            self.env.unwrapped.multi_goal_tracker._success_steps_required
            == success_steps_required
        )

        # take 4 steps without achieving the goal.
        for i in range(4):
            _, _, done, info = self.env.step(self.zero_action)
            assert not done
            assert not info["sub_goal_is_successful"]
            assert info["steps_since_last_goal"] == i + 1
            assert (
                self.env.unwrapped.multi_goal_tracker._consecutive_steps_with_success
                == 0
            )

        # achieve the goal!
        self._fake_one_success()

        # remain the state for successful goal achievement for some steps.
        for j in range(success_steps_required - 1):
            _, _, done, info = self.env.step(self.zero_action)

            assert (
                self.env.unwrapped.multi_goal_tracker._consecutive_steps_with_success
                == j + 1
            )

            assert not done
            assert not info["sub_goal_is_successful"]
            assert not info["trial_success"]
            assert info["steps_since_last_goal"] == j + 5

        _, _, done, info = self.env.step(self.zero_action)

        assert (
            self.env.unwrapped.multi_goal_tracker._consecutive_steps_with_success
            == success_steps_required
        )

        assert done
        assert info["sub_goal_is_successful"]
        assert info["trial_success"]
