from collections import OrderedDict

import gym
import numpy as np
from gym.spaces import Box, Dict

from robogym.wrappers import randomizations
from robogym.wrappers.randomizations import loguniform
from robogym.wrappers.util import update_obs_space


class RandomizedCubeSizeWrapper(randomizations.RandomizedBodyWrapper):
    def __init__(self, env=None, cube_size_range=[0.95, 1.05]):
        super().__init__(env)
        self._cube_size_range = cube_size_range

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("cube_size", (1,))])

    def _get_field(self, sim):
        cube_idx = sim.model.geom_name2id("cube:middle")
        cube_size = sim.model.geom_size[
            cube_idx
        ]  # the other unnamed geom is target cube
        ret = {"cube_size": cube_size}

        if "cube:top" in sim.model.body_names:
            for name in ["cube:top", "cube:bottom"]:
                idx = sim.model.body_name2id(name)
                ret[name] = sim.model.body_pos[idx]
        return ret

    def _set_field(self, sim):
        cube_geom_idxs = [sim.model.geom_name2id("cube:middle")]
        if "cube:top" in sim.model.geom_names:
            cube_geom_idxs += [
                sim.model.geom_name2id(name) for name in ["cube:top", "cube:bottom"]
            ]
        random_state = self.unwrapped._random_state
        scale = random_state.uniform(
            self._cube_size_range[0], self._cube_size_range[1], size=[1]
        )
        val = self._orig_value["cube_size"] * scale
        for cube_geom_idx in cube_geom_idxs:
            sim.model.geom_size[cube_geom_idx] = val

        # For face cube, we have to move bodies for rescaling to work.
        if "cube:top" in sim.model.body_names:
            for name in ["cube:top", "cube:bottom"]:
                idx = sim.model.body_name2id(name)
                sim.model.body_pos[idx] = self._orig_value[name] * scale

        return OrderedDict([("cube_size", val.copy())])


class RandomizedWindWrapper(gym.Wrapper):
    def __init__(self, env=None, force_std=1.0, max_mean_time_between=0.8):
        super().__init__(env)
        self._force_std = force_std
        self._max_mean_time_between = max_mean_time_between

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        sim = self.unwrapped.sim
        self._hit_prob = loguniform(
            random_state=self.unwrapped._random_state,
            low=0.01
            * sim.nsubsteps
            * sim.model.opt.timestep
            / self._max_mean_time_between,
            high=sim.nsubsteps * sim.model.opt.timestep / self._max_mean_time_between,
        )
        return obs

    def step(self, action):
        ret = self.env.step(action)
        sim = self.unwrapped.sim
        i = sim.model.body_name2id("cube:middle")
        cube_mass = sim.model.body_mass[i]
        sim.data.xfrc_applied[i, :3] *= 0.99  # TODO: make substeps dependent
        if self.unwrapped._random_state.random_sample() < self._hit_prob:
            sim.data.xfrc_applied[i, :3] = (
                self.unwrapped._random_state.randn(3) * cube_mass * self._force_std
            )
        return ret


class CubeFreezingPhasespaceBody(randomizations.FreezingPhasespaceBody):
    def __init__(self, env=None, disappear_p_1s=0.02, freeze_scale_s=1.0):
        super().__init__(
            env,
            keys=[
                "noisy_relative_goal_pos",
                "noisy_relative_goal_quat",
                "noisy_relative_goal_face_angle",
                "noisy_achieved_goal_pos",
                "noisy_achieved_goal_quat",
                "noisy_achieved_goal_face_angle",
                "noisy_cube_pos",
            ],
            disappear_p_1s=disappear_p_1s,
            freeze_scale_s=freeze_scale_s,
        )


class StopOnFallWrapper(gym.Wrapper):
    def __init__(self, env=None, drop_reward=-20.0, min_episode_length=-1):
        super().__init__(env)
        self.observation_space = update_obs_space(env, {"fell_down": (1,)})
        self.steps = 0
        self.drop_reward = drop_reward
        self.env.unwrapped.reward_names.append("drop")
        self.min_episode_length = min_episode_length
        self.drops_so_far = 0
        self.first_drop = 0

    def reset(self, *args, **kwargs):
        self.drops_so_far = 0
        self.first_drop = 0
        self.steps = 0
        return self.observation(self.env.reset(*args, **kwargs))

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # Handle dropping.
        current_drop_reward = 0.0
        if self._is_fallen():
            done = True
            self.drops_so_far += 1
            if not self.first_drop:
                # Penalize only first frame where cube dropped.
                current_drop_reward = self.drop_reward
                self.first_drop = info["successes_so_far"] + 1

        if self.steps < self.min_episode_length:
            # If we require a minimum episode length, do not return a terminal state until
            # we have reached the minimum.
            done = False

        rew = rew + [current_drop_reward]
        info["fell_down"] = self._is_fallen()
        info["drops_so_far"] = self.drops_so_far
        info["first_drop"] = self.first_drop

        self.steps += 1
        return self.observation(obs), rew, done, info

    def observation(self, observation):
        observation["fell_down"] = np.array([self._is_fallen()])
        return observation

    def _is_fallen(self):
        cube_middle_idx = self.unwrapped.sim.model.site_name2id("cube:center")
        cube_middle_pos = self.unwrapped.sim.data.site_xpos[cube_middle_idx]
        return cube_middle_pos[2] < 0.04


class AngleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        """Change angles to sines and cosines"""
        super().__init__(env)
        new_spaces = {}
        for name, value in self.env.observation_space.spaces.items():
            if name.endswith("_angle"):
                new_spaces[name] = Box(
                    -np.inf, np.inf, [value.shape[0] * 2], value.dtype
                )
            else:
                new_spaces[name] = value
        self.observation_space = Dict(new_spaces)

    def observation(self, observation):
        extended_observation = OrderedDict()
        for key in observation:
            if key.endswith("_angle"):
                extended_observation[key] = np.concatenate(
                    [np.cos(observation[key]), np.sin(observation[key])]
                )
            else:
                extended_observation[key] = observation[key]
        return extended_observation
