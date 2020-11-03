import gym


class EnvParameterWrapper(gym.Wrapper):
    """ Generic parameter that modifies environment parameters on each reset """

    def __init__(self, env, parameter_name: str):
        super().__init__(env)

        self.parameter_name = parameter_name
        self.original_value = getattr(self.unwrapped.parameters, self.parameter_name)

    def step(self, action):
        return self.env.step(action)

    def new_value(self):
        raise NotImplementedError

    def reset(self, **kwargs):
        setattr(self.unwrapped.parameters, self.parameter_name, self.new_value())
        return self.env.reset(**kwargs)


class RandomizedPerpendicularCubeSizeWrapper(EnvParameterWrapper):
    """ Randomize size of the "perpendicular" cube """

    def __init__(self, env=None, cube_size_range=None):
        super().__init__(env, "cube_size_multiplier")

        if cube_size_range is None:
            cube_size_range = [0.95, 1.05]

        self._cube_size_range = cube_size_range

    def new_value(self):
        return self.unwrapped._random_state.uniform(
            self._cube_size_range[0], self._cube_size_range[1]
        )
