import enum
from collections import OrderedDict
from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Box, Dict


def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, (np.prod(shape),), np.float32)
    return Dict(spaces)


class BinSpacing(enum.Enum):
    """
    An Enum class ti generate action bin spacing arrays.
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"  # Exponential binning. Expects a symmetric action space centered around zero

    def get_bin_array(self, lower_bound, upper_bound, n_bins) -> np.ndarray:
        if self is BinSpacing.LINEAR:
            return np.linspace(lower_bound, upper_bound, n_bins)
        else:
            assert (
                lower_bound == -upper_bound and n_bins % 2 == 1
            ), "Exponential binning is only supported on symmetric action space with an odd number of bins"
            half_range = np.array([2 ** (-n) for n in range(n_bins // 2)]) * lower_bound
            return np.concatenate([half_range, [0], -half_range[::-1]])


class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A wrapper that maps a continuous gym action space into a discrete action space.
    """

    # default action bins for DiscretizeActionWrapper
    DEFAULT_BINS = 11

    def __init__(
        self, env=None, n_action_bins=DEFAULT_BINS, bin_spacing=BinSpacing.LINEAR
    ):
        """
        n_action_bins: can be int or None
          if None is passed, then DEFAULT_BINS will be used.
        """
        super().__init__(env)
        assert isinstance(env.action_space, Box)
        self._disc_to_cont = []
        if n_action_bins is None:
            n_action_bins = self.DEFAULT_BINS

        for low, high in zip(env.action_space.low, env.action_space.high):
            self._disc_to_cont.append(
                bin_spacing.get_bin_array(low, high, n_action_bins)
            )

        temp = [n_action_bins for _ in self._disc_to_cont]
        self.action_space = gym.spaces.MultiDiscrete(temp)
        self.action_space.seed(env.action_space.np_random.randint(0, 2 ** 32 - 1))

    def action(self, action):
        assert len(action) == len(self._disc_to_cont)
        return np.array(
            [m[a] for a, m in zip(action, self._disc_to_cont)], dtype=np.float32
        )


class RewardNameWrapper(gym.Wrapper):
    """ Sets the default reward name on the environment """

    def __init__(self, env):
        super().__init__(env)

        unwrapped = self.env.unwrapped

        if not hasattr(unwrapped, "reward_names"):
            self.env.unwrapped.reward_names = ["env"]

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipObservationWrapper(gym.ObservationWrapper):
    """
    Clips observations into a fixed range.
    """

    def __init__(self, env=None, clip=100.0):
        super().__init__(env)
        self._clip = clip

    def observation(self, observation):
        clipped_observation = OrderedDict()
        for key in observation:
            clipped_observation[key] = np.clip(
                observation[key], -self._clip, self._clip
            )
        return clipped_observation

    def compute_relative_goals(self, *args, **kwargs):
        self.env.compute_relative_goals(*args, **kwargs)

    def compute_goal_reward(self, *args, **kwargs):
        return self.env.compute_goal_reward(*args, **kwargs)


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clips reward values into a fixed range.
    """

    def __init__(self, env=None, clip=100.0):
        super().__init__(env)
        self._clip = clip

    def reward(self, reward):
        clipped_reward = np.clip(reward, -self._clip, self._clip)
        return clipped_reward

    def compute_relative_goals(self, *args, **kwargs):
        self.env.compute_relative_goals(*args, **kwargs)

    def compute_goal_reward(self, *args, **kwargs):
        return self.env.compute_goal_reward(*args, **kwargs)


class ClipActionWrapper(gym.ActionWrapper):
    """ Clips action values into a normalized space between -1 and 1"""

    def action(self, action):
        return np.clip(a=action, a_min=-1.0, a_max=1.0)


class IncrementalExpAvg(object):
    """ A generic exponential moving average filter. """

    def __init__(self, alpha, intial_value=None):
        self._value = 0
        self._t = 0
        self._alpha = alpha
        if intial_value is not None:
            self.update(intial_value)

    def update(self, observation):
        self._value = self._value * self._alpha + (1 - self._alpha) * observation
        self._t += 1

    def get(self):
        if self._value is None:
            return None
        else:
            return self._value / (1 - self._alpha ** self._t)


class PreviousActionObservationWrapper(gym.Wrapper):
    """
    Wrapper that annotates observations with a cached previous action.
    """

    def __init__(self, env=None):
        super().__init__(env)
        env.observation_space.spaces["previous_action"] = deepcopy(env.action_space)

    def reset(self, *args, **kwargs):
        self.previous_action = np.zeros(self.env.action_space.shape)
        return self.observation(self.env.reset(*args, **kwargs))

    def observation(self, observation):
        observation["previous_action"] = self.previous_action.copy()
        return observation

    def step(self, action):
        self.previous_action = action.copy()
        ob, rew, done, info = self.env.step(action)
        return self.observation(ob), rew, done, info

    def compute_relative_goals(self, *args, **kwargs):
        self.env.compute_relative_goals(*args, **kwargs)

    def compute_goal_reward(self, *args, **kwargs):
        return self.env.compute_goal_reward(*args, **kwargs)


class SmoothActionWrapper(gym.Wrapper):
    """
    Applies smoothing to the current action using an Exponential Moving Average filter.
    """

    def __init__(self, env, alpha=0.0):
        super().__init__(env)
        self._alpha = alpha
        delta = OrderedDict([("action_ema", self.env.action_space.shape)])
        self.observation_space = update_obs_space(self.env, delta)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        sim = self.unwrapped.sim
        adjusted_alpha = np.power(
            self._alpha, (sim.model.opt.timestep * sim.nsubsteps) / 0.08
        )
        self._ema = IncrementalExpAvg(alpha=adjusted_alpha)
        obs["action_ema"] = np.zeros(self.env.action_space.shape)
        return obs

    def step(self, action):
        self._ema.update(action)
        action = self._ema.get()
        obs, rew, done, info = self.env.step(action)
        obs["action_ema"] = action
        return obs, rew, done, info


class RelativeGoalWrapper(gym.ObservationWrapper):
    """
    Wrapper that computes the 'relative goal' and 'achieved goal' observations for
    environments.
    """

    def __init__(self, env, obs_prefix=""):

        # Prefix to map goal observation to state observation. This is a hack to
        # handle inconsistent naming convention for cube environment observations
        # e.g. goal_pos goal observation maps to cube_pos state observation.
        self.obs_prefix = obs_prefix
        super().__init__(env)

        self.goal_obs_names = []
        delta = OrderedDict()
        for name, space in self.env.observation_space.spaces.items():
            if name.startswith("goal_"):
                delta[f"achieved_{name}"] = space.shape
                delta[f"relative_{name}"] = space.shape
                delta[f"noisy_achieved_{name}"] = space.shape
                delta[f"noisy_relative_{name}"] = space.shape

                obs_name = name[len("goal_"):]
                assert (
                    f"{self.obs_prefix}{obs_name}" in self.env.observation_space.spaces
                ), (
                    f"Found {name} but not {self.obs_prefix}{obs_name} in observation space. "
                    f"RelativeGoalWrapper won't work. Available observation space: "
                    f"{sorted(self.env.observation_space.spaces.keys())}"
                )

                self.goal_obs_names.append(obs_name)

        self.observation_space = update_obs_space(self.env, delta)

    def observation(self, observation):
        """ Calculate 'relative goal' and 'achieved goal' """
        current_state = {
            f"{self.obs_prefix}{n}": observation[f"{self.obs_prefix}{n}"]
            for n in self.goal_obs_names
        }
        noisy_goal_state = {
            f"{self.obs_prefix}{n}": observation[f"noisy_{self.obs_prefix}{n}"]
            for n in self.goal_obs_names
        }

        relative_goal = self.env.unwrapped.goal_generation.relative_goal(
            self.env.unwrapped._goal, current_state
        )

        noisy_relative_goal = self.env.unwrapped.goal_generation.relative_goal(
            self.env.unwrapped._goal, noisy_goal_state
        )

        for name in self.goal_obs_names:
            obs_name = f"{self.obs_prefix}{name}"
            observation[f"achieved_goal_{name}"] = observation[obs_name].copy()
            observation[f"relative_goal_{name}"] = relative_goal[obs_name]
            observation[f"noisy_achieved_goal_{name}"] = observation[
                f"noisy_{obs_name}"
            ].copy()
            observation[f"noisy_relative_goal_{name}"] = noisy_relative_goal[obs_name]

        return observation


class UnifiedGoalObservationWrapper(gym.ObservationWrapper):
    """Concatenates the pieces of every goal type"""

    def __init__(
        self, env, goal_keys=["relative_goal", "achieved_goal", "goal"], goal_parts=[],
    ):
        super().__init__(env)
        self.delta = OrderedDict()

        for goal_key in goal_keys:
            goal_len = sum(
                [
                    self.observation_space.spaces[key].shape[0]
                    for key in self.observation_space.spaces.keys()
                    if key.startswith(goal_key)
                ]
            )

            self.delta[goal_key] = (goal_len,)

            if any(
                [
                    key.startswith("noisy_" + goal_key + "_")
                    for key in self.observation_space.spaces.keys()
                ]
            ):
                self.delta["noisy_" + goal_key] = (goal_len,)

        self.goal_parts = goal_parts
        self.observation_space = update_obs_space(self.env, self.delta)

    def observation(self, observation):
        new_obs = OrderedDict()
        for key, value in observation.items():
            new_obs[key] = value

        # It's a bit hacky to hard code observation key here but we have to do it now
        # because we need to keep old policy backward compatible by keep observation order
        # the same.
        for goal_key in self.delta.keys():
            goal_parts = [goal_key + "_" + part for part in self.goal_parts]
            goal = np.concatenate(
                [observation[key] for key in goal_parts if key in observation]
            )
            new_obs[goal_key] = goal

        return new_obs


class SummedRewardsWrapper(gym.RewardWrapper):
    """
    Ensures that reward is a scalar.
    """

    def reward(self, reward):
        return np.sum([reward])
