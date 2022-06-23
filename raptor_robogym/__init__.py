import gym
import numpy as np
from rl.utils.make_callable import make_callable

class NormReward(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, sum(reward), done, info

class DictToArray(gym.Wrapper):
    def __init__(self, env):
        super(DictToArray, self).__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Dict), \
            f"expected wrapped env's observation space to be a gym.spaces.Dict but got {self.env.observation_space}."
        ignored_keys = {"performed_action", "action_history", "action_delay", "randomized_phasespace", "randomized_phasespace_fingers"}
        self._space_names = [n for n in self.env.observation_space.spaces.keys() if n not in ignored_keys]
        self._space_shapes = [np.prod(s.shape) for n, s in self.env.observation_space.spaces.items() if n not in ignored_keys]
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(sum(self._space_shapes),))

    def reset(self):
        return self._concate_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._concate_obs(obs), reward, done, info

    def _concate_obs(self, observation):
        out = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        so_far = 0
        for key, size in zip(self._space_names, self._space_shapes):
            if key in observation:
                obs = observation[key].reshape(-1)
                assert len(obs) == size, \
                    f"expected observation named {key} to have size {size} but found {len(obs)} instead (observation[key].shape = {observation[key].shape} -- {self.env.observation_space[key]})."
                out[so_far:so_far + size] = obs
                so_far += size
            else:
                so_far += size
                print(f"wtf! {key}")
        return out



def wrap_env(env):
    env = env if isinstance(env, gym.Env) else make_callable(env)()
    return NormReward(DictToArray(env))
