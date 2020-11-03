from collections import OrderedDict

import gym
import numpy as np

from robogym.robot.shadow_hand.hand_forward_kinematics import (
    FINGERTIP_SITE_NAMES,
    REFERENCE_SITE_NAMES,
)
from robogym.utils.sensor_utils import check_occlusion, occlusion_markers_exist
from robogym.wrappers import randomizations


class RandomizedPhasespaceFingersWrapper(randomizations.RandomizedBodyWrapper):
    def __init__(self, env=None, fingertips_noise=0.003, reference_noise=0.001):
        """Randomize position of phasespace markers on fingers. Units in meters."""
        super().__init__(env)
        self._all_sites = [
            (f"robot0:{name}", fingertips_noise) for name in FINGERTIP_SITE_NAMES
        ]
        self._all_sites += [
            (f"robot0:{name}", reference_noise) for name in REFERENCE_SITE_NAMES
        ]

    def _get_observation_space_delta(self, sim):
        site_idxes = [
            sim.model.site_name2id(f"robot0:{c}")
            for c in FINGERTIP_SITE_NAMES + REFERENCE_SITE_NAMES
        ]
        return OrderedDict(
            [("randomized_phasespace", sim.model.site_pos[site_idxes, :].shape)]
        )

    def _get_field(self, sim):
        orig_pos = [None for _ in self._all_sites]
        for idx, (name, noise) in enumerate(self._all_sites):
            sensor_idx = sim.model.site_name2id(name)
            orig_pos[idx] = sim.model.site_pos[sensor_idx, :].copy()
        return np.array(orig_pos)

    def _set_field(self, sim):
        randomized_phasespace = []
        for idx, (name, noise) in enumerate(self._all_sites):
            sensor_idx = sim.model.site_name2id(name)
            sim.model.site_pos[sensor_idx, :] = self._orig_value[
                idx
            ] + self.unwrapped._random_state.uniform(-noise, noise, size=(3,))
            randomized_phasespace.append(sim.model.site_pos[sensor_idx, :])
        randomized_phasespace = np.array(randomized_phasespace, copy=True)
        return OrderedDict([("randomized_phasespace_fingers", randomized_phasespace)])


class FingersOccludedPhasespaceMarkers(gym.ObservationWrapper):
    def __init__(self, env):
        """Make phasespace markers disappear when the occlusion detectors have collision,
        which is simulated by returning old phasespace values.

        This relies on `RandomizeObservationWrapper` with "fingertip_pos" in the input
        "levels".
        """
        super().__init__(env)
        self._key = "noisy_fingertip_pos"
        self._n_markers = 5
        self._obs_buffer = None

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._occlusion_markers_exist = occlusion_markers_exist(self.unwrapped.sim)
        assert len(obs[self._key]) % 3 == 0
        assert len(obs[self._key]) // 3 == self._n_markers

        self._obs_buffer = obs[self._key].copy()
        return obs

    def observation(self, observation):
        if not self._occlusion_markers_exist:
            return observation
        else:
            new_observation = OrderedDict()
            for key in observation:
                new_observation[key] = observation[key]

            # Freeze the fingertip_pos read if the finger is occluded.
            is_occluded_list = check_occlusion(self.unwrapped.sim)
            for i, is_occluded in enumerate(is_occluded_list):
                if not is_occluded:
                    self._obs_buffer[3 * i: 3 * (i + 1)] = observation[self._key][
                        3 * i: 3 * (i + 1)
                    ]

            new_observation[self._key] = self._obs_buffer.copy()
            self._obs_buffer = new_observation[self._key].copy()
            return new_observation


class FingersFreezingPhasespaceMarkers(randomizations.FreezingPhasespaceMarkers):
    def __init__(
        self,
        env=None,
        key="noisy_fingertip_pos",
        disappear_p_1s=0.2,
        freeze_scale_s=1.0,
    ):
        super().__init__(
            env, key=key, disappear_p_1s=disappear_p_1s, freeze_scale_s=freeze_scale_s
        )


class FingerSeparationWrapper(gym.Wrapper):
    """ Immobilize and separate all fingers other than active finger. """

    def __init__(self, env, active_finger):
        super().__init__(env)
        self.active_finger = active_finger
        self.FINGERS = ("TH", "FF", "MF", "RF", "LF", "WR")

    def reset(self, *args, **kwargs):
        # Spreads fingers apart
        finger_i = self.FINGERS.index(self.active_finger)
        for i in range(len(self.FINGERS)):
            if "F" in self.FINGERS[i] and i != finger_i:
                if i < finger_i:
                    limit = 0
                elif i > finger_i:
                    limit = 1
                self._freeze_joint("{}J4".format(self.FINGERS[i]), 1)
                self._freeze_joint("{}J3".format(self.FINGERS[i]), limit)
                self._freeze_joint("{}J2".format(self.FINGERS[i]), 1)
                self._freeze_joint("{}J1".format(self.FINGERS[i]), 1)
                self._freeze_joint("{}J0".format(self.FINGERS[i]), 1)
            if "TH" in self.FINGERS[i] and i != finger_i:
                self._freeze_joint("{}J4".format(self.FINGERS[i]), 0)
                self._freeze_joint("{}J3".format(self.FINGERS[i]), 1)
                self._freeze_joint("{}J2".format(self.FINGERS[i]), 1)
                self._freeze_joint("{}J1".format(self.FINGERS[i]), 0)
                self._freeze_joint("{}J0".format(self.FINGERS[i]), 0)

        return self.env.reset(*args, **kwargs)

    def _freeze_joint(self, joint_name, limit):
        if limit == 0:
            diff = -0.01
        else:
            diff = 0.01
        model = self.env.unwrapped.sim.model
        if "robot0:" + joint_name in model.joint_names:
            joint_id = model.joint_name2id("robot0:" + joint_name)
            model.jnt_range[joint_id, limit] = (
                model.jnt_range[joint_id, 1 - limit] + diff
            )


class RandomizedRobotDampingWrapper(randomizations.RandomizedDampingWrapper):
    def __init__(self, env=None, damping_range=[1 / 1.5, 1.5], robot_name="robot0"):
        joint_names = [
            name
            for name in env.unwrapped.sim.model.joint_names
            if name.startswith(robot_name + ":")
        ]
        super().__init__(env, damping_range, joint_names)


class RandomizedRobotKpWrapper(randomizations.RandomizedKpWrapper):
    def __init__(self, env=None, kp_range=[0.5, 2.0], robot_name="robot0"):
        actuator_names = [
            name
            for name in env.unwrapped.sim.model.actuator_names
            if name.startswith(robot_name + ":")
        ]
        super().__init__(env, kp_range, actuator_names)


class FixedWristWrapper(gym.Wrapper):
    def __init__(self, env=None, wrj0_pos=0.0):
        self.wrj0_pos = wrj0_pos
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        a_wrj0_id = self.env.unwrapped.sim.model.actuator_name2id("robot0:A_WRJ0")
        ctrlrange = self.env.unwrapped.sim.model.actuator_ctrlrange[a_wrj0_id]
        actuation_range = (ctrlrange[1] - ctrlrange[0]) / 2.0
        joint_pos = self.env.unwrapped.sim.data.get_joint_qpos("robot0:WRJ0")
        action[a_wrj0_id] = (self.wrj0_pos - joint_pos) / actuation_range
        return self.env.step(action)


class RewardObservationWrapper(gym.Wrapper):
    def __init__(self, env=None, reward_inds=None):
        super().__init__(env)
        self.reward_inds = reward_inds
        self.shape = (len(reward_inds),) if reward_inds is not None else (1,)
        env.observation_space.spaces["reward"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self.observation(obs, None)

    def observation(self, observation, reward):
        observation["reward"] = self._reward_obs(reward)
        return observation

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        return self.observation(ob, rew), rew, done, info

    def _reward_obs(self, reward):
        if reward is None:  # this should only be the case on reset
            obs = np.zeros(self.shape)
        else:
            if (
                self.reward_inds is None
            ):  # This should only be the case when reward is a scalar
                obs = np.array([reward])
            else:
                obs = np.array(reward[self.reward_inds])
        return obs


DEFAULT_NOISE_LEVELS = {
    "achieved_goal": {"uncorrelated": 0.001, "additive": 0.001},
}
