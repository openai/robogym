from robogym.mujoco.modifiers.base import Modifier


class TimestepModifier(Modifier):
    """ Modify simulation timestep """

    def __call__(self, timestep):
        self.sim.model.opt.timestep = timestep
