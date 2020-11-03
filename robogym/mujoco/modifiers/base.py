class Modifier:
    """ Base class for various MuJoCo modifiers """

    def __init__(self):
        self.sim = None

    def initialize(self, sim):
        self.sim = sim

    def __call__(self, parameter_value):
        """ Apply given parameter to the sim """
        raise NotImplementedError
