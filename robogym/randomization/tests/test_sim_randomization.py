import numpy as np

from robogym.envs.rearrange.blocks import BlockRearrangeEnv
from robogym.randomization.sim import (
    GenericSimRandomizer,
    GeomSolimpRandomizer,
    GeomSolrefRandomizer,
    GravityRandomizer,
    JointMarginRandomizer,
    PidRandomizer,
)


class TestEnv(BlockRearrangeEnv):
    @classmethod
    def build_simulation_randomizers(cls, constants):
        return [
            GravityRandomizer(),
            JointMarginRandomizer(),
            PidRandomizer("pid_kp"),
            GenericSimRandomizer(
                name="dof_frictionloss_robot",
                field_name="dof_frictionloss",
                dof_jnt_prefix="robot0:",
                apply_mode="uncoupled_mean_variance",
            ),
            GeomSolimpRandomizer(),
            GeomSolrefRandomizer(),
        ]


def test_sim_randomization():
    def get_mujoco_values(mj_sim):
        return [
            value_getter(mj_sim).copy()
            for value_getter in [
                lambda sim: sim.model.opt.gravity,
                lambda sim: sim.model.jnt_margin,
                lambda sim: sim.model.actuator_gainprm,
                lambda sim: sim.model.dof_frictionloss,
                lambda sim: sim.model.geom_solref,
                lambda sim: sim.model.geom_solimp,
            ]
        ]

    env = TestEnv.build()
    env.reset()

    original_values = get_mujoco_values(env.sim)
    parameters = env.unwrapped.randomization.simulation_randomizer.get_parameters()

    initial_param_values = [param.get_value() for param in parameters]

    # Update parameters.
    for param in parameters:
        low, high = param.get_range()
        param.set_value(np.random.uniform(low, high))

    for _ in range(3):
        env.reset()

        new_values = get_mujoco_values(env.sim)

        for original_value, new_value in zip(original_values, new_values):
            assert not np.allclose(original_value, new_value)

    # Reset parameter back to original values.
    for param, initial_value in zip(parameters, initial_param_values):
        param.set_value(initial_value)

    env.reset()

    new_values = get_mujoco_values(env.sim)

    # Make sure parameter value doesn't drift away.
    for original_value, new_value in zip(original_values, new_values):
        assert np.allclose(original_value, new_value)
