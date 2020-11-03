import gym
import numpy as np


class RewardDiffWrapper(gym.RewardWrapper):
    """
    This wrapper is meant to work only with goal envs.
    It returns the difference in reward between steps instead of the reward.
    For distance based rewards, this turns the reward into a potential function.
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.previous_rew = None

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.previous_rew = self.unwrapped.compute_reward(
            obs["achieved_goal"], self.unwrapped.goal, {}
        )
        return obs

    def reward(self, reward):
        reward_diff = reward - self.previous_rew
        self.previous_rew = reward
        return reward_diff


class RobotStateObservationWrapper(gym.ObservationWrapper):
    """
    Adds robot state observations (i.e. the qpos values for all robot joints)
    without leaking information about the goal. This is necessary because the original
    Gym environment does not include this information and also concatenates everything
    into a single observation vector--which leaks the goal if using vision training.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._robot_joint_names = [
            n
            for n in self.env.unwrapped.sim.model.joint_names
            if n.startswith("robot0:")
        ]

        self.observation_space.spaces["robot_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self._robot_joint_names),)
        )

    def observation(self, obs):
        obs["robot_state"] = np.array(
            [
                self.env.unwrapped.sim.data.get_joint_qpos(n)
                for n in self._robot_joint_names
            ]
        )
        return obs


class CompatibilityWrapper(gym.Wrapper):
    """
    successes_so_far is use to track success in evaluator.
    This field has the same meaning as is_success coming from
    gym fetch environment.
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["successes_so_far"] = info["is_success"]
        return obs, reward, done, info
