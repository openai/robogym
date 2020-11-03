import copy
import math
from collections import OrderedDict, deque

import gym
import numpy as np
from gym.spaces import Box, Dict

from robogym.utils.dactyl_utils import actuated_joint_range
from robogym.utils.rotation import (
    normalize_angles,
    quat_average,
    quat_from_angle_and_axis,
    quat_mul,
    quat_normalize,
)
from robogym.wrappers.util import update_obs_space


def loguniform(random_state, low, high, size=[]):
    return np.exp(random_state.uniform(np.log(low), np.log(high), size=size))


class RandomizedBodyWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Randomize properties of some bodies.
        """
        super().__init__(env)
        self._orig_value = None
        assert hasattr(self.unwrapped, "sim")

        delta = self._get_observation_space_delta(self.unwrapped.sim)
        self.observation_space = update_obs_space(self.env, delta)

    def reset(self, *args, **kwargs):
        sim = self.unwrapped.sim
        if self._orig_value is None:
            self._orig_value = copy.deepcopy(self._get_field(sim))
        self._obs_delta = self._set_field(sim)
        # We want to reset only after updating the parameters, the reason is that reset will run
        # the simulation a bit until the cube is on the palm, so, if we reset before, it will run
        # with parameters from the previous episode which is problematic for ADR
        obs = self.env.reset(*args, **kwargs)
        return self.observation(obs)

    def observation(self, obs):
        new_obs = OrderedDict()
        for key, value in obs.items():
            new_obs[key] = value
        for key, value in self._obs_delta.items():
            new_obs[key] = copy.deepcopy(value)
            if isinstance(new_obs[key], np.ndarray):
                new_obs[key] = new_obs[key].ravel()
        return new_obs

    def _get_observation_space_delta(self, sim):
        return {}

    def _get_field(self, sim):
        """Called once per environments existance to establish initial value
        that simulator assigns to some quantity, e.g. gravity"""
        return None

    def _set_field(self, sim):
        """Called every time episode is reset to update the new value for quantity
        code can reference self._orig_value to get original value returned by
        self._get_field."""
        raise NotImplementedError()


class RandomizedBodyInertiaWrapper(RandomizedBodyWrapper):
    def __init__(self, env=None, mass_range=[0.5, 1.5]):
        super().__init__(env)
        self._mass_range = mass_range

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("body_inertia", sim.model.body_inertia.shape)])

    def _get_field(self, sim):
        return sim.model.body_inertia

    def _set_field(self, sim):
        val = sim.model.body_inertia[:] = (
            self._orig_value
            * self.unwrapped._random_state.uniform(
                low=self._mass_range[0],
                high=self._mass_range[1],
                size=(sim.model.body_inertia.shape[0], 1),
            )
        )
        return OrderedDict([("body_inertia", val.copy())])


class RandomizedFrictionBaseWrapper(RandomizedBodyWrapper):
    def __init__(self, env, multiplier_ranges, geom_name_prefix=None):
        """
        Contact friction parameters for dynamically generated contact pairs.
        1. the sliding friction, acting along both axes of the tangent plane.
        2. the torsional friction, acting around the contact normal.
        3. the rolling friction, acting around both axes of the tangent plane.

        We expect the multiplier_ranges is a numpy array of shape (3, 2)
        multiplier_ranges[0], ...[1], ...[2] correspond to multiplier range that should
        be applied to three types of contact friction irrespectively.
        """
        super().__init__(env)
        self._multiplier_ranges = np.array(multiplier_ranges).copy()
        assert self._multiplier_ranges.shape == (3, 2)

        if geom_name_prefix is None:
            self._geom_names = list(self.env.unwrapped.sim.model.geom_names).copy()
        else:
            self._geom_names = [
                name
                for name in self.env.unwrapped.sim.model.geom_names
                if name.startswith(geom_name_prefix)
            ]

        self._geom_ids = [
            self.unwrapped.sim.model.geom_name2id(name) for name in self._geom_names
        ]
        self._geom_ids = np.array(self._geom_ids)

        # Used by ADR
        self._multiplier_values = None

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("friction", sim.model.geom_friction.shape)])

    def _get_field(self, sim):
        return sim.model.geom_friction

    def _set_field(self, sim):
        if self._multiplier_values is not None:
            assert len(self._multiplier_values) == self._orig_value.shape[-1]
            for col, multiplier in enumerate(self._multiplier_values):
                val = self._orig_value[self._geom_ids, col] * multiplier
                sim.model.geom_friction[self._geom_ids, col] = val
        else:
            assert len(self._multiplier_ranges) == self._orig_value.shape[-1]
            for col, multi_range in enumerate(self._multiplier_ranges):
                # use a single multiplier for each type of friction. Avoids "averaging" out
                # friction.
                multiplier = self.unwrapped._random_state.uniform(
                    multi_range[0], multi_range[1]
                )
                val = self._orig_value[self._geom_ids, col] * multiplier
                sim.model.geom_friction[self._geom_ids, col] = val
        return OrderedDict([("friction", sim.model.geom_friction.copy())])

    def update_parameters(self, slide_multiplier, spin_multiplier, roll_multiplier):
        self._multiplier_values = [slide_multiplier, spin_multiplier, roll_multiplier]


class RandomizedFrictionWrapper(RandomizedFrictionBaseWrapper):
    def __init__(self, env=None, multiplier_range=[0.7, 1.3]):
        multiplier_ranges = [multiplier_range] * 3
        super().__init__(env, multiplier_ranges, "robot0:")


class RandomizedRobotFrictionWrapper(RandomizedFrictionBaseWrapper):
    def __init__(
        self, env=None, multiplier_ranges=[[0.7, 1.3], [0.5, 1.5], [0.5, 1.5]]
    ):
        super().__init__(env, multiplier_ranges, "robot0:")


class RandomizedCubeFrictionWrapper(RandomizedFrictionBaseWrapper):
    def __init__(
        self, env=None, multiplier_ranges=[[0.5, 1.5], [0.2, 5.0], [0.2, 5.0]]
    ):
        super().__init__(env, multiplier_ranges, "cube:")


class RandomizedGravityWrapper(RandomizedBodyWrapper):
    def __init__(self, env=None, gravity_std=0.4):
        super().__init__(env)
        self._gravity_std = gravity_std

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("gravity", sim.model.opt.gravity.shape)])

    def _get_field(self, sim):
        return sim.model.opt.gravity

    def _set_field(self, sim):
        val = sim.model.opt.gravity[
            :
        ] = self._orig_value + self._gravity_std * self.unwrapped._random_state.randn(3)
        return OrderedDict([("gravity", val.copy())])


class RandomizedTimestepWrapper(RandomizedBodyWrapper):
    def __init__(
        self,
        env=None,
        min_lambda=125 * 10,
        max_lambda=1000 * 10,
        adr_bias_magic=0.6,
        adr_variance_magic=1.0,
    ):
        """Randomize the environment timestep by a value from an exponential distribution
        with the parameter lambda sampled once per episode from [min_lambda,max_lambda]."""
        super().__init__(env)

        self._min_lambda = min_lambda
        self._max_lambda = max_lambda

        self._adr_bias_magic = adr_bias_magic
        self._adr_variance_magic = adr_variance_magic

        self._adr_bias = 0.0
        self._adr_variance = 0.0

        self._side = 1.0  # positive or negative

        self._p_flip_pos = 0.5
        self._p_flip_neg = 0.5

        self._positive_lambda = 0.0
        self._negative_lambda = 0.0

        self._positive_lambda = 0.0
        self._negative_lambda = 0.0

        self._bias_multiplier = 0.0
        self._variance_multiplier = 0.0

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("timestep_lambda", (2,)), ("timestep_multipliers", (2,))])

    def _get_field(self, sim):
        return sim.model.opt.timestep

    def update_adr_bias(self, adr_bias):
        self._adr_bias = adr_bias

    def update_adr_variance(self, adr_variance):
        self._adr_variance = adr_variance

    def _set_field(self, *args, **kwargs):
        self._bias_multiplier = np.exp(self._adr_bias * self._adr_bias_magic)
        self._variance_multiplier = np.exp(
            self._adr_variance * self._adr_variance_magic
        )

        self._positive_lambda = self.unwrapped._random_state.uniform(
            self._min_lambda, self._max_lambda
        )

        self._negative_lambda = self.unwrapped._random_state.uniform(
            self._min_lambda, self._max_lambda
        )

        self._side = self.unwrapped._random_state.choice([-1.0, 1.0])

        self._p_flip_pos = self.unwrapped._random_state.uniform()
        self._p_flip_neg = self.unwrapped._random_state.uniform()

        return OrderedDict(
            [
                ("timestep_lambda", [self._positive_lambda, self._negative_lambda]),
                (
                    "timestep_multipliers",
                    [self._bias_multiplier, self._variance_multiplier],
                ),
            ]
        )

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # Simulate flipping somehow
        if self._side > 0:
            if self.unwrapped._random_state.uniform() > self._p_flip_pos:
                self._side = -self._side
        else:
            if self.unwrapped._random_state.uniform() > self._p_flip_neg:
                self._side = -self._side

        if self._side > 0:
            noise = self.unwrapped._random_state.exponential(
                1.0 / self._positive_lambda
            )
        else:
            noise = self.unwrapped._random_state.exponential(
                1.0 / self._negative_lambda
            )

        noise *= self._variance_multiplier

        if self._side < 0:
            # Rescale
            fraction = noise / self._orig_value
            noise = self._orig_value * (fraction / (1 + fraction))

        if self._side < 0:
            # Clip the noise if it's negative so that the simulation is stable
            noise = np.clip(noise, 0.0, self._orig_value / 2)

        self.unwrapped.sim.model.opt.timestep = self._bias_multiplier * (
            self._orig_value + self._side * noise
        )

        return self.observation(obs), rew, done, info


# empirical constant to allow quaternion noise to be specified at same level as
# Euler angle additive perturbation measured in radians
QUAT_NOISE_CORRECTION = 1.96


class RandomizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, levels=None):
        super().__init__(env)

        self._correlated_multiplier = 1.0
        self._uncorrelated_multipler = 1.0

        self._levels = levels
        self._additive_bias = {}
        self._multiplicative_bias = {}

        new_spaces = self.env.observation_space.spaces.copy()

        new_spaces.update(
            {f"noisy_{k}": self.env.observation_space.spaces[k] for k in self._levels}
        )
        self.observation_space = Dict(new_spaces)
        self.random_state = self.unwrapped._random_state

    def key_length(self, key):
        if not key.endswith("_quat"):
            return self.env.observation_space.spaces[key].shape[0]
        else:
            assert self.env.observation_space.spaces[key].shape[0] == 4
            return 1

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)

        for key in sorted(self._levels):
            key_len = self.key_length(key)
            self._additive_bias[key] = (
                self.random_state.randn(key_len)
                * self._levels[key].get("additive", 0.0)
                * self._correlated_multiplier
            )
            self._multiplicative_bias[key] = (
                1.0
                + self.random_state.randn(key_len)
                * self._levels[key].get("multiplicative", 0.0)
                * self._correlated_multiplier
            )
        return self.observation(observation)

    def observation(self, observation):
        randomized_observation = OrderedDict()
        for key in observation:
            randomized_observation[key] = observation[key]
        for key in sorted(self._levels):
            key_len = self.key_length(key)
            uncorrelated_bias = (
                self.random_state.randn(key_len)
                * self._levels[key].get("uncorrelated", 0.0)
                * self._uncorrelated_multipler
            )
            additive_bias = self._additive_bias[key] + uncorrelated_bias

            if f"noisy_{key}" in observation:
                # There is already noisy value available for this observation key,
                # we apply noise on top of the noisy value.
                obs_key = f"noisy_{key}"
            else:
                # Apply noise on top of noiseless observation if no noisy value available.
                obs_key = key

            new_value = observation[obs_key].copy()

            if not key.endswith("_quat"):
                new_value *= self._multiplicative_bias[key]
                new_value += additive_bias
            else:
                assert np.allclose(self._multiplicative_bias[key], 1.0)
                noise_axis = self.random_state.uniform(-1.0, 1.0, size=(3,))
                additive_bias *= QUAT_NOISE_CORRECTION
                noise_quat = quat_from_angle_and_axis(additive_bias, noise_axis)
                new_value = quat_normalize(quat_mul(new_value, noise_quat))

            randomized_observation[f"noisy_{key}"] = new_value

        return randomized_observation

    def update_parameters(self, correlated_multiplier, uncorrelated_multiplier):
        self._correlated_multiplier = correlated_multiplier
        self._uncorrelated_multipler = uncorrelated_multiplier


class FreezingPhasespaceMarkers(gym.ObservationWrapper):
    def __init__(self, env=None, key=None, disappear_p_1s=None, freeze_scale_s=None):
        """Make phasespace markers disappear sometimes (which is simulated by returning old values)

        Parameters
        ----------
        key: str
            Name of key in obs that representes the markers.
            must be (3 * n_markers) array )
        disappear_p_1s: float
            Probability that one of markers will disappear during period of 1 second
        freeze_scale_s: float
            For how long does a marker disappear
        """
        super().__init__(env)
        n_substeps = self.unwrapped.sim.nsubsteps
        substep_duration_s = self.unwrapped.sim.model.opt.timestep
        step_duration_s = n_substeps * substep_duration_s
        self._key = key
        self._disappear_p = 1.0 - (1.0 - disappear_p_1s) ** step_duration_s
        self._freeze_scale_steps = freeze_scale_s / step_duration_s

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        assert len(obs[self._key]) % 3 == 0
        self._n_markers = len(obs[self._key]) // 3
        self._freeze_left = np.array([0 for _ in range(self._n_markers)])
        self._obs_buffer = obs[self._key].copy()
        return obs

    def observation(self, observation):
        new_observation = OrderedDict()
        for key in observation:
            new_observation[key] = observation[key]

        # update nonfrozen observations for all markers.
        for i, left in enumerate(self._freeze_left[: self._n_markers]):
            if left <= 0:
                self._obs_buffer[3 * i: 3 * (i + 1)] = observation[self._key][
                    3 * i: 3 * (i + 1)
                ]
        new_observation[self._key] = self._obs_buffer.copy()

        # update freeze_left
        self._freeze_left = np.maximum(self._freeze_left - 1, 0)
        does_freeze = (
            self.unwrapped._random_state.random_sample(size=self._n_markers)
            < self._disappear_p
        )
        new_freeze_len = np.round(
            self.unwrapped._random_state.exponential(
                scale=self._freeze_scale_steps, size=self._n_markers
            )
        )
        self._freeze_left = (
            1 - does_freeze
        ) * self._freeze_left + does_freeze * new_freeze_len

        return new_observation


class FreezingPhasespaceBody(gym.ObservationWrapper):
    def __init__(self, env=None, keys=None, disappear_p_1s=None, freeze_scale_s=None):
        """Make some keys disappear sometimes (which is simulated by returning old values)

        Parameters
        ----------
        keys: str
            Names of keys to be frozen.
        disappear_p_1s: float
            Probability that one of markers will disappear during period of 1 second
        freeze_scale_s: float
            For how long does a marker disappear
        """
        super().__init__(env)

        keys = [k for k in keys if k in env.observation_space.spaces]

        n_substeps = self.unwrapped.sim.nsubsteps
        substep_duration_s = self.unwrapped.sim.model.opt.timestep
        step_duration_s = n_substeps * substep_duration_s
        self._keys = keys
        self._disappear_p = 1.0 - (1.0 - disappear_p_1s) ** step_duration_s
        self._freeze_scale_steps = freeze_scale_s / step_duration_s

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._freeze_left = 0
        self._obs_buffer = {k: obs[k].copy() for k in self._keys}
        return obs

    def observation(self, observation):
        # update nonfrozen observations for all markers.
        if self._freeze_left <= 0:
            for k in self._keys:
                self._obs_buffer[k] = observation[k].copy()

        # update freeze_left
        self._freeze_left = max(self._freeze_left - 1, 0)
        does_freeze = self.unwrapped._random_state.random_sample() < self._disappear_p
        new_freeze_len = np.round(
            self.unwrapped._random_state.exponential(scale=self._freeze_scale_steps)
        )
        self._freeze_left = (
            1 - does_freeze
        ) * self._freeze_left + does_freeze * new_freeze_len

        new_observation = OrderedDict()
        for key in observation:
            new_observation[key] = (
                self._obs_buffer[key].copy() if key in self._keys else observation[key]
            )

        return new_observation


class RandomizedActionLatency(RandomizedBodyWrapper):
    def __init__(self, env, max_delay=1):
        """For random coordinates of action space return old values"""
        super().__init__(env)
        self._max_delay = max_delay
        assert (
            isinstance(self.env.action_space, Box)
            and len(self.env.action_space.shape) == 1
        )
        self._action_size = self.env.action_space.shape[0]
        delta = OrderedDict(
            [
                ("performed_action", self.env.action_space.shape),
                ("action_history", (self._max_delay + 1, self._action_size)),
                ("action_delay", self.env.action_space.shape),
            ]
        )
        self.observation_space = update_obs_space(self.env, delta)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._action_history = np.zeros((self._max_delay + 1, self._action_size))
        self._action_delay = self.unwrapped._random_state.randint(
            low=0, high=self._max_delay + 1, size=self._action_size
        )
        obs["action_history"] = self._action_history[:-1]
        obs["action_delay"] = self._action_delay
        return obs

    def step(self, action):
        self._action_history[0], self._action_history[1:] = (
            action,
            self._action_history[:-1],
        )
        new_action = self._action_history[
            self._action_delay, list(range(self._action_size))
        ]
        obs, rew, done, info = self.env.step(new_action.copy())
        obs["action_history"] = self._action_history[:-1]
        obs["action_delay"] = self._action_delay
        return obs, rew, done, info

    def update_parameters(self, max_delay):
        self._max_delay = max_delay


class RandomizedDampingWrapper(RandomizedBodyWrapper):
    def __init__(self, env=None, damping_range=[0.3, 3.0], joint_names=[]):
        self._damping_range = damping_range
        self._joint_names = joint_names
        super().__init__(env)

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("joint_damping", (len(self._joint_names),))])

    def _get_field(self, sim):
        joint_ids = [sim.model.joint_name2id(name) for name in self._joint_names]
        dof_ids = [
            idx for idx in range(sim.model.nv) if sim.model.dof_jntid[idx] in joint_ids
        ]
        return [sim.model.dof_damping[idx] for idx in dof_ids]

    def _set_field(self, sim):
        joint_ids = [sim.model.joint_name2id(name) for name in self._joint_names]
        dof_ids = [
            idx for idx in range(sim.model.nv) if sim.model.dof_jntid[idx] in joint_ids
        ]
        val = self._orig_value * loguniform(
            self.unwrapped._random_state,
            self._damping_range[0],
            self._damping_range[1],
            size=[len(dof_ids)],
        )
        sim.model.dof_damping[dof_ids] = val
        return OrderedDict([("joint_damping", val.copy())])


class RandomizedJointLimitWrapper(RandomizedBodyWrapper):
    def __init__(self, env=None, joint_names=[], relative_std=0.15):
        """Randomize the joint limit and update joint range and actuator ctrl range accordingly.
        """
        self._joint_names = joint_names or env.unwrapped.sim.model.joint_names
        self._relative_std = relative_std

        assert set(self._joint_names).issubset(set(env.unwrapped.sim.model.joint_names))
        self._joint_ids = [
            env.unwrapped.sim.model.joint_name2id(name) for name in self._joint_names
        ]

        super().__init__(env)

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("joint_limit", (len(self._joint_names),))])

    def _get_field(self, sim):
        jnt_limits = actuated_joint_range(sim)[self._joint_ids]
        assert all(jnt_limits[:, 1] - jnt_limits[:, 0] >= 0.0)
        return jnt_limits

    def _set_field(self, sim):
        limit_widths = self._orig_value[:, 1] - self._orig_value[:, 0]
        stds = limit_widths * self._relative_std
        stds_reshaped = np.repeat(stds, 2).reshape(
            len(self._joint_names), 2
        ) * self._random_noises(len(self._joint_names))

        new_jnt_limits = self._orig_value.copy()

        for idx, jnt_id in enumerate(self._joint_ids):
            min_width = limit_widths[idx] * 0.001
            # Let's go through the joint limit range one by one to handle special case,
            # i.e., if the lower bound is 0.0, we should not lower it to be negative
            low, high = new_jnt_limits[idx]
            if low == 0.0 and high > 0:
                low = max(0.0, low + stds_reshaped[idx][0])
                high = max(low + min_width, high + stds_reshaped[idx][1])
            elif low < 0 and high == 0.0:
                high = min(0.0, high + stds_reshaped[idx][1])
                low = min(high - min_width, low + stds_reshaped[idx][0])
            else:
                low += stds_reshaped[idx][0]
                high = max(low + min_width, high + stds_reshaped[idx][1])

            new_jnt_limits[idx][0] = low
            new_jnt_limits[idx][1] = high

        # Apply the new joint limit to the joint range and actuator control range.
        sim.model.jnt_range[self._joint_ids] = new_jnt_limits.copy()

        for jnt_id, jnt_name in zip(self._joint_ids, self._joint_names):
            actuator_name = jnt_name.replace(":", ":A_")
            if actuator_name not in sim.model.actuator_names:
                continue

            actuator_id = sim.model.actuator_name2id(actuator_name)

            if actuator_name[-3:] == "FJ1":
                # This actuator should control the unactuated "*FJ0' joint as well.
                other_jnt_name = jnt_name.replace("FJ1", "FJ0")
                other_jnt_id = sim.model.joint_name2id(other_jnt_name)
                fj0_range = sim.model.jnt_range[other_jnt_id]
                fj1_range = sim.model.jnt_range[jnt_id]
                sim.model.actuator_ctrlrange[actuator_id] = np.array(
                    [min(fj0_range[0], fj1_range[0]), fj0_range[1] + fj1_range[1]]
                )
            else:
                sim.model.actuator_ctrlrange[actuator_id] = sim.model.jnt_range[jnt_id]

        return OrderedDict([("joint_limit", new_jnt_limits)])

    def _random_noises(self, n_jnt):
        return self.unwrapped._random_state.randn(n_jnt, 2)

    def update_parameters(self, relative_std):
        self._relative_std = relative_std


class RandomizedTendonRangeWrapper(RandomizedBodyWrapper):
    """Randomize and update all tendon ranges."""

    def __init__(self, env=None, relative_std=0.15):
        self._relative_std = relative_std
        super().__init__(env)

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("tendon_range", (len(sim.model.tendon_names),))])

    def _get_field(self, sim):
        assert all(sim.model.tendon_range[:, 1] - sim.model.tendon_range[:, 0] >= 0.0)
        return sim.model.tendon_range

    def _set_field(self, sim):
        widths = self._orig_value[:, 1] - self._orig_value[:, 0]
        assert widths.shape == (len(sim.model.tendon_names),)

        bounds_change = np.repeat(widths * self._relative_std, 2)
        bounds_change = bounds_change.reshape(len(sim.model.tendon_names), 2)
        bounds_change *= self.unwrapped._random_state.randn(
            len(sim.model.tendon_names), 2
        )

        new_tendon_ranges = self._orig_value.copy()
        for tendon in sim.model.tendon_names:
            tendon_id = sim.model.tendon_name2id(tendon)

            lower, upper = new_tendon_ranges[tendon_id]
            assert lower >= 0.0, "tendon range should have nonnegative lower bound"
            assert upper >= 0.0, "tendon range should have nonnegative upper bound"

            lower = max(0.0, lower + bounds_change[tendon_id][0])
            upper = max(
                lower + (widths[tendon_id] * 0.001), upper + bounds_change[tendon_id][1]
            )

            new_tendon_ranges[tendon_id][0] = lower
            new_tendon_ranges[tendon_id][1] = upper

        sim.model.tendon_range[:] = new_tendon_ranges.copy()
        return OrderedDict([("tendon_range", new_tendon_ranges)])

    def update_parameters(self, relative_std):
        self._relative_std = relative_std


class RandomizedKpWrapper(RandomizedBodyWrapper):
    def __init__(self, env=None, kp_range=[0.75, 1.5], actuator_names=[]):
        self._kp_range = kp_range
        self._actuator_names = actuator_names
        super().__init__(env)

    def _get_observation_space_delta(self, sim):
        return OrderedDict([("actuator_kp", (len(self._actuator_names),))])

    def _get_field(self, sim):
        actuator_ids = [
            sim.model.actuator_name2id(name) for name in self._actuator_names
        ]
        return [sim.model.actuator_gainprm[idx, 0] for idx in actuator_ids]

    def _set_field(self, sim):
        actuator_ids = [
            sim.model.actuator_name2id(name) for name in self._actuator_names
        ]
        val = self._orig_value * loguniform(
            self.unwrapped._random_state,
            self._kp_range[0],
            self._kp_range[1],
            size=[len(actuator_ids)],
        )
        sim.model.actuator_gainprm[actuator_ids, 0] = val.copy()
        return OrderedDict([("actuator_kp", val.copy())])


class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env=None, multiplicative=0.03, additive=0.03, uncorrelated=0.1):
        super().__init__(env)
        self._multiplicative = multiplicative
        self._additive = additive
        self._uncorrelated = uncorrelated

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)

        self._multiplicative_bias = (
            1.0
            + self.unwrapped._random_state.randn(self.action_space.shape[0])
            * self._multiplicative
        )
        self._additive_bias = (
            self.unwrapped._random_state.randn(self.action_space.shape[0])
            * self._additive
        )

        return observation

    def action(self, action):
        new_action = action * self._multiplicative_bias + self._additive_bias
        new_action += (
            self.unwrapped._random_state.randn(self.action_space.shape[0])
            * self._uncorrelated
        )
        return new_action

    def update_parameters(self, multiplicative, additive, uncorrelated):
        self._multiplicative = multiplicative
        self._additive = additive
        self._uncorrelated = uncorrelated


class BacklashWrapper(gym.Wrapper):
    """
    Simulates bashlash. coef controls how much of a backlash we have
    coef=0 - there is an infinite backlash
    coef=np.inf - backlash is ineffective. coef=exp(4.25) ~ 70. acts almost like np.inf.

    There is a different coefficient on tendon pulling up vs tensdon pulling down (as there
    are two tendons). Both coef_down_log and coef_up_log of actuators are in log scale. The
    actual backlash is np.exp(coef_down_log) and np.exp(coef_up_log).

    Args:
    - std (float): we sample coef_log per episode with this standard deviation.
    """

    def __init__(self, env, std=0.1):
        super().__init__(env)

        self.coef_down_log = np.array(
            [
                4.25,  # A_WRJ1
                4.25,  # A_WRJ0
                2.93,  # A_FFJ3
                4.25,  # A_FFJ2
                4.25,  # A_FFJ1
                4.25,  # A_MFJ3
                4.25,  # A_MFJ2
                1.92,  # A_MFJ1
                4.25,  # A_RFJ3
                3.35,  # A_RFJ2
                4.25,  # A_RFJ1
                4.25,  # A_LFJ4
                4.25,  # A_LFJ3
                3.87,  # A_LFJ2
                1.39,  # A_LFJ1
                4.25,  # A_THJ4
                1.25,  # A_THJ3
                4.25,  # A_THJ2
                4.25,  # A_THJ1
                4.25,
            ]  # A_THJ0
        )
        self.coef_up_log = np.array(
            [
                4.25,  # A_WRJ1
                4.25,  # A_WRJ0
                4.25,  # A_FFJ3
                4.25,  # A_FFJ2
                1.86,  # A_FFJ1
                4.25,  # A_MFJ3
                4.25,  # A_MFJ2
                1.44,  # A_MFJ1
                4.25,  # A_RFJ3
                2.98,  # A_RFJ2
                2.07,  # A_RFJ1
                4.25,  # A_LFJ4
                4.25,  # A_LFJ3
                2.94,  # A_LFJ2
                1.41,  # A_LFJ1
                2.82,  # A_THJ4
                1.53,  # A_THJ3
                4.25,  # A_THJ2
                2.86,  # A_THJ1
                2.10,
            ]  # A_THJ0
        )
        self.slack = None
        self.std = std

    def reset(self, *args, **kwargs):
        self.slack = np.zeros(len(self.env.unwrapped.sim.model.actuator_names))
        ob = self.env.reset(*args, **kwargs)

        rand = self.unwrapped._random_state
        shape = self.coef_up_log.shape

        self.episode_coef_down = np.exp(
            self.coef_down_log * (1.0 + rand.randn(*shape) * self.std)
        )
        self.episode_coef_up = np.exp(
            self.coef_up_log * (1.0 + rand.randn(*shape) * self.std)
        )

        # Otherwise, backlash is so huge that robot is useless.
        self.episode_coef_down = np.maximum(self.episode_coef_down, 2.0)
        self.episode_coef_up = np.maximum(self.episode_coef_up, 2.0)
        return ob

    def step(self, action):
        # ctrl in space.
        sim = self.env.unwrapped.sim
        self.env.unwrapped._set_action(action)
        ctrl = sim.data.ctrl

        qpos_as_ctrl = self._qpos2ctrl(sim, sim.data.qpos)
        # use kp and vel.
        dt = sim.model.opt.timestep * sim.nsubsteps
        diff = ctrl - qpos_as_ctrl
        eps = 1e-5
        incr = (diff < -eps) * diff * self.episode_coef_down * dt + (
            diff > eps
        ) * diff * self.episode_coef_up * dt
        alpha = np.abs(np.sign(diff) - self.slack) / (np.abs(incr) + 1e-12)
        alpha = np.clip(alpha, 0.0, 1.0)
        ctrl = alpha * qpos_as_ctrl + (1.0 - alpha) * ctrl

        # Ensures that backlash behaves proportionally to elapsed time.
        self.slack += incr
        self.slack = np.clip(self.slack, -1.0, 1.0)

        action = self._ctrl2action(
            sim, ctrl, self.env.unwrapped.constants.relative_action
        )
        return self.env.step(action)

    def _get_actuation_center(self, sim, ctrl, relative_action=False):
        ctrlrange = sim.model.actuator_ctrlrange

        if relative_action:
            actuation_center = np.zeros_like(ctrl)
            for i in range(sim.data.ctrl.shape[0]):
                actuation_center[i] = sim.data.get_joint_qpos(
                    sim.model.actuator_names[i].replace(":A_", ":")
                )
            for joint_name in ["FF", "MF", "RF", "LF"]:
                act_idx = sim.model.actuator_name2id("robot0:A_%sJ1" % joint_name)
                actuation_center[act_idx] += sim.data.get_joint_qpos(
                    "robot0:%sJ0" % joint_name
                )
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

        return actuation_center

    def _ctrl2action(self, sim, ctrl, relative_action=False):
        ctrlrange = sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_center = self._get_actuation_center(sim, ctrl, relative_action)
        action = (ctrl - actuation_center) / actuation_range
        return action

    def _qpos2ctrl(self, sim, qpos):
        action = np.zeros(len(sim.model.actuator_names))
        for act_idx, act_name in enumerate(sim.model.actuator_names):
            joint_name = act_name.replace(":A_", ":")
            jnt_idx = sim.model.get_joint_qpos_addr(joint_name)
            action[act_idx] += qpos[jnt_idx]
            for suffix in ["FFJ1", "MFJ1", "RFJ1", "LFJ1"]:
                if suffix in joint_name:
                    jnt_idx = sim.model.get_joint_qpos_addr(
                        joint_name.replace("J1", "J0")
                    )
                    action[act_idx] += qpos[jnt_idx]
        return action

    def update_parameters(self, std):
        self.std = std


class ActionDelayWrapper(gym.Wrapper):
    """
    Delay in miliseconds, and its standard deviation.
    """

    def __init__(
        self,
        env,
        delay=30.0,
        per_episode_std=0.1,
        per_step_std=0.002,
        random_state=None,
    ):
        """
        :param env: Env to be wrapped.
        :param delay: Amount of delay in milisecond.
        :param std: standard deviation of the delay.
        """
        super().__init__(env)
        self.delay = delay
        self.per_episode_std = per_episode_std
        self.per_step_std = per_step_std
        self.nsubsteps = None  # Default value of nsubsteps from a simulator
        self.timestep = None  # Default value of timestep from a simulator.
        self.total_length_ms = None  # How long takes the entire step.
        self.last_action = None
        self.random_state = random_state or self.unwrapped._random_state
        self.reset()

    def reset(self, *args, **kwargs):
        self.last_action = None
        sim = self.env.unwrapped.sim
        random_normal = self.random_state.normal()
        self.per_episode_delay = self.delay * (
            1.0 + random_normal * self.per_episode_std
        )
        if self.nsubsteps is None or self.timestep is None:
            self.nsubsteps = sim.nsubsteps
            self.timestep = sim.model.opt.timestep
            self.total_length_ms = self.timestep * self.nsubsteps * 1000

        return self.env.reset(*args, **kwargs)

    def step(self, action):
        if self.last_action is None:
            self.last_action = action.copy()
        sim = self.env.unwrapped.sim
        random_normal = self.random_state.normal()
        delay = self.per_episode_delay * (1.0 + random_normal * self.per_step_std)
        if delay > 1e-4:
            delay = self._clip_delay(delay)
            self._set_delay(delay)
            self.env.step(self.last_action.copy())  # This step takes 'delay' time
        else:
            delay = 0.0

        remaining_delay = self.total_length_ms - delay
        self._set_delay(self._clip_delay(remaining_delay))
        obs = self.env.step(action)  # This step takes 'normal length' - delay time.

        self.last_action = action.copy()
        # Set back values.
        sim.nsubsteps = self.nsubsteps
        sim.model.opt.timestep = self.timestep
        return obs

    def _clip_delay(self, delay):
        delay = max(0.05 * self.total_length_ms, delay)
        delay = min(self.total_length_ms, delay)
        return delay

    def _set_delay(self, delay):
        """
        sets nsubsteps and timestep to for step to take 'delay'
        """
        sim = self.env.unwrapped.sim
        delay_nsubsteps = int(delay / self.timestep / 1000)
        assert delay_nsubsteps >= 1, "This delay cannot be modeled within step."
        delay_timestep = delay / (delay_nsubsteps * 1000)
        assert np.abs((delay_nsubsteps * delay_timestep * 1000 - delay) / delay) < 1e-3
        sim.nsubsteps = delay_nsubsteps
        sim.model.opt.timestep = delay_timestep

    def update_parameters(self, delay, per_episode_std, per_step_std):
        self.delay = delay
        self.per_episode_std = per_episode_std
        self.per_step_std = per_step_std


class ObservationDelayWrapper(gym.ObservationWrapper):
    """
    Wrapper to simulate observation delay which is defined as
    delay between true observation timestamp and the timestamp
    when observation is obtained and used to calculate action.
    """

    class Interpolator:
        def interpolate(self, x1, x2, t):
            raise NotImplementedError

    class LinearInterpolator(Interpolator):
        def interpolate(self, x1, x2, t):
            assert 0 <= t <= 1
            return x1 * t + x2 * (1 - t)

    class QuatInterpolator(Interpolator):
        def interpolate(self, x1, x2, t):
            return quat_average([x1, x2], [t, 1 - t])

    class RadianInterpolator(Interpolator):
        def interpolate(self, x1, x2, t):
            assert 0 <= t <= 1
            diff = normalize_angles(x2 - x1)
            return normalize_angles(x2 - t * diff)

    def __init__(self, env, levels):
        """
        :param env: Env to be wrapped.
        :param levels: Delay levels for each observation. Example structure is:
        {
            "interpolators" {
                "cube_quat": "QuatInterpolator"
            },

            "groups": {
                "vision": {
                    # Group of observations which same delay will be applied.
                    "obs_names": ["cube_pos", "cube_pot"],

                    # mean for delay in number of steps.
                    "mean": 2,

                    # std for delay in number steps.
                    "std": 1,
                },
                "giiker": {
                    "obs_names": ["cube_face_angle"],
                    "mean": 3,
                    "std": 1.5,
                }
            }
        }
        """
        super().__init__(env)

        self.groups = levels["groups"]

        self.interpolators = {
            obs_name: getattr(self, interpolator)()
            for obs_name, interpolator in levels["interpolators"].items()
        }

        new_spaces = self.env.observation_space.spaces.copy()

        for name in self.group_names:
            new_spaces.update(
                {
                    f"noisy_{k}": self.env.observation_space.spaces[k]
                    for k in self.groups[name]["obs_names"]
                }
            )

        self.observation_space = Dict(new_spaces)
        self.default_interpolator = self.LinearInterpolator()
        self.random_state = self.unwrapped._random_state
        self.prev_obs = deque(maxlen=10)

    @property
    def group_names(self):
        return sorted(self.groups.keys())

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.prev_obs.clear()
        return self.observation(obs)

    def observation(self, observation):
        self.prev_obs.append(observation)

        obs = observation.copy()

        for name in self.group_names:
            group = self.groups[name]
            delay = self.random_state.normal(group["mean"], group["std"])

            delay = np.clip(delay, 0.0, len(self.prev_obs) - 1)
            delay_l = math.floor(delay)
            delay_h = math.ceil(delay)
            t = delay - delay_l

            obs_l = self.prev_obs[-1 - delay_l]
            obs_h = self.prev_obs[-1 - delay_h]

            for obs_name in group["obs_names"]:
                new_obs_name = f"noisy_{obs_name}"
                assert new_obs_name not in obs, (
                    f"Noisy value for {obs_name} already exists. Please make sure "
                    f"observation delay wrapper is applied before other observation "
                    f"noise wrappers."
                )

                interpolator = self._get_interpolator(obs_name)
                obs[new_obs_name] = interpolator.interpolate(
                    obs_h[obs_name], obs_l[obs_name], t
                )

        return obs

    def _get_interpolator(self, obs_name):
        if obs_name in self.interpolators:
            return self.interpolators[obs_name]
        else:
            return self.default_interpolator

    def update_parameters(self, params):
        for name, (mean, std) in params.items():
            self.groups[name]["mean"] = mean
            self.groups[name]["std"] = std


class RandomizedBrokenActuatorWrapper(gym.ActionWrapper):
    def __init__(
        self, env=None, proba_broken=0.001, max_broken_actuators=2, uncorrelated=0.05
    ):
        """We mark whether an actuator as broken at each reset.
        The probability of all actuators being healthy is ~ 0.98 = (1-0.001) ** 20.
        We fake the broken actuator effect by overwriting the action for that actuator to
        0.0 + white noise.

        Args:
        - proba_broken (float): probability of one actuator being broken.
        - max_broken_actuators (int): only this number of actuators can be broken at the same
            time at maximum.
        - uncorrelated (float): white noise on the zero action for broken actuators.
        """
        super().__init__(env)

        self._uncorrelated = uncorrelated
        self._proba_broken = proba_broken
        self._max_broken_actuators = max_broken_actuators

        self._broken_aids = []
        self._broken_action = 0.0

    def reset(self, *args, **kwargs):
        # Potentially we can change the default actions for broken actuators here.
        observation = self.env.reset(*args, **kwargs)

        n_actuators = len(self.unwrapped.sim.model.actuator_names)
        self._broken_aids = [
            i
            for i in range(n_actuators)
            if self.unwrapped._random_state.rand() < self._proba_broken
        ]

        if len(self._broken_aids) > self._max_broken_actuators:
            self._broken_aids = self.unwrapped._random_state.choice(
                self._broken_aids, self._max_broken_actuators, replace=False
            )

        return observation

    def action(self, action):
        # in make_env(), relative_action=True by default.
        new_action = action.copy()
        for i in self._broken_aids:
            white_noise = self.unwrapped._random_state.rand() * self._uncorrelated
            new_action[i] = self._broken_action + white_noise

        return new_action
