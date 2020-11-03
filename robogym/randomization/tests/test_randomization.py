import unittest

import attr
import numpy as np

from robogym.randomization.env import (
    EnvActionRandomizer,
    EnvObservationRandomizer,
    EnvParameterRandomizer,
    EnvRandomization,
    EnvSimulationRandomizer,
    build_randomizable_param,
)
from robogym.randomization.observation import ObservationRandomizer
from robogym.randomization.parameters import FloatRandomizerParameter


class DummyRandomizerParameter(FloatRandomizerParameter):
    def __init__(self, name, val):
        super().__init__(
            name, val, value_range=(-1.0, 1.0), delta=1.0,
        )


@attr.s(auto_attribs=True)
class DummyNestedEnvParameter:
    c: int = build_randomizable_param(1, low=-3, high=3)


@attr.s(auto_attribs=True)
class DummyEnvParameter:
    a: int = build_randomizable_param(0, low=-5, high=5)
    b: float = build_randomizable_param(0.0, low=-1.0, high=1.0)

    x: int = 0  # Non randomizable parameter.

    nested: DummyNestedEnvParameter = DummyNestedEnvParameter()


class DummyObservationRandomizer(ObservationRandomizer):
    def __init__(self, name, val):
        super().__init__(name)
        self.val = self.register_parameter(val)

    def _randomize(self, target, random_state):
        target[self.val.name] = self.val.get_value()
        return target


class TestRandomization(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.random_state = np.random.RandomState()

    def test_randomizer_parameters(self):
        parameter = DummyRandomizerParameter("foo", 0.0)

        assert parameter.get_value() == 0.0
        assert parameter.get_range() == (-1.0, 1.0)
        assert parameter.get_delta() == 1.0

        parameter.set_value(1.0)
        assert parameter.get_value() == 1.0

    def test_randomizer_basic(self):
        """
        Test functionality of basic randomizer.
        """
        randomizer = EnvParameterRandomizer(DummyEnvParameter())

        assert len(randomizer.get_parameters()) == 3

        # Make sure register duplicate parameter is not allowed.
        with self.assertRaises(AssertionError):
            randomizer.register_parameter(DummyRandomizerParameter("a", 1))

        randomizer.register_parameter(DummyRandomizerParameter("d", 1))

        assert len(randomizer.get_parameters()) == 4

        randomizer.get_parameter("a").set_value(1)
        randomizer.get_parameter("b").set_value(0.5)
        randomizer.get_parameter("c").set_value(2)

        parameters = randomizer.randomize(DummyEnvParameter(), self.random_state)
        assert parameters.a == 1
        assert parameters.b == 0.5
        assert parameters.nested.c == 2

        randomizer.disable()

        parameters = randomizer.randomize(DummyEnvParameter(), self.random_state)
        randomizer.get_parameter("a").set_value(1)
        assert parameters.a == 0

    def test_observation_randomizer(self):
        randomizer = EnvObservationRandomizer(
            [
                DummyObservationRandomizer("r1", DummyRandomizerParameter("foo", 0.0)),
                DummyObservationRandomizer("r2", DummyRandomizerParameter("bar", 1.0)),
            ]
        )

        assert len(randomizer.get_randomizers()) == 2
        assert len(randomizer.get_parameters()) == 2
        obs = randomizer.randomize({}, self.random_state)
        assert obs["foo"] == 0.0
        assert obs["bar"] == 1.0

    def test_env_randomization(self):
        randomization = EnvRandomization(
            parameter_randomizer=EnvParameterRandomizer(DummyEnvParameter()),
            observation_randomizer=EnvObservationRandomizer(
                [
                    DummyObservationRandomizer(
                        "r1", DummyRandomizerParameter("foo", 0.0)
                    ),
                ]
            ),
            action_randomizer=EnvActionRandomizer([]),
            simulation_randomizer=EnvSimulationRandomizer([]),
        )

        randomization.update_parameter("observation.r1:foo", 0.5)

        parameter = randomization.get_parameter("observation.r1:foo")
        assert parameter.get_value() == 0.5
