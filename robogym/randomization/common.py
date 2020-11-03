import abc
from collections import OrderedDict
from enum import Enum
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

VType = TypeVar("VType", int, float)


class DType(Enum):
    INT = (1,)
    FLOAT = 2


class RandomizerParameter(Generic[VType], abc.ABC):
    """
    Base interface for randomizer parameter.
    """

    INT = DType.INT
    FLOAT = DType.FLOAT

    def __init__(
        self,
        name: str,
        initial_value: VType,
        value_range: Tuple[VType, VType],
        delta: Optional[VType] = None,
    ):
        self.name = name
        self._value_range: Tuple[VType, VType] = self._convert_range(value_range)
        self._value: VType = self._convert_value(initial_value)
        self._delta: Optional[VType] = self._convert_delta(delta)

    ################################################
    # External APIs to interact with domain randomization.
    def get_value(self) -> VType:
        return self._value

    def set_value(self, value: VType):
        self._value = self._convert_value(value)

    def get_range(self) -> Tuple[VType, VType]:
        return self._value_range

    def get_delta(self) -> Optional[VType]:
        return self._delta

    @property
    @abc.abstractmethod
    def dtype(self):
        pass

    ################################################
    # Internal methods.
    def _convert_value(self, value: VType) -> VType:
        low, high = self.get_range()
        value = self._convert_type(value)
        assert (
            low <= value <= high
        ), (  # type: ignore
            f"Value {value} is not within range of [{low}, {high}]"
        )

        return value

    def _convert_range(self, value_range: Tuple[VType, VType]) -> Tuple[VType, VType]:
        assert (
            len(value_range) == 2
        ), f"Invalid range {value_range}, must tuple of two values."
        low, high = value_range

        return self._convert_type(low), self._convert_type(high)

    def _convert_delta(self, delta: Optional[VType]):
        if delta is not None:
            return self._convert_type(delta)
        else:
            return None

    @classmethod
    @abc.abstractmethod
    def _convert_type(cls, val: VType) -> VType:
        pass

    def __repr__(self):
        return (
            f"{self.__class__}(\n"
            f"value={self.get_value()}\n"
            f"range={self.get_range()}\n"
            f")"
        )


TType = TypeVar("TType")


class Randomizer(abc.ABC, Generic[TType]):
    """
    Base interface for a randomizer.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self._parameters: Dict[str, RandomizerParameter] = OrderedDict()
        self._enabled = enabled

    def randomize(self, target: TType, random_state: np.random.RandomState) -> TType:
        if self._enabled:
            return self._randomize(target, random_state)
        else:
            return target

    @abc.abstractmethod
    def _randomize(self, target: TType, random_state: np.random.RandomState) -> TType:
        pass

    def get_parameters(self) -> List[RandomizerParameter]:
        """
        Return all parameters for this randomizer.
        """
        return list(self._parameters.values())

    def get_parameter(self, name: str) -> RandomizerParameter:
        """
        Get parameter by name.
        """
        assert (
            name in self._parameters
        ), f"Parameter {name} does not exist in randomizer {self.name}."

        return self._parameters[name]

    def register_parameter(self, parameter: RandomizerParameter):
        """
        Register a parameter for this randomizer.
        """
        assert (
            parameter.name not in self._parameters
        ), f"Parameter with name {parameter.name} already exists."

        self._parameters[parameter.name] = parameter

        return parameter

    def enable(self):
        """
        Enable the randomizer.
        """
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def enabled(self):
        return self._enabled

    def reset(self):
        """
        Reset state of the randomizer. Called during environment reset.
        """
        pass


RType = TypeVar("RType", bound=Randomizer)


class RandomizerCollection(Generic[RType]):
    """
    Interface for collection of randomizers, it provides functionality
    to register child randomizers and retrieve their parameters.
    """

    def __init__(self):
        self._randomizers = OrderedDict()

    def register_randomizer(self, randomizer: RType) -> RType:
        """
        Add a randomizer to the collection.
        """
        assert (
            randomizer.name not in self._randomizers
        ), f"Randomizer with name {randomizer.name} already exists."
        self._randomizers[randomizer.name] = randomizer
        return randomizer

    def get_randomizers(self) -> List[RType]:
        """
        Get all randomizers.
        """
        return list(self._randomizers.values())

    def get_randomizer(self, name) -> RType:
        """
        Get randomizer by name.
        """
        assert name in self._randomizers, f"Randomizer {name} does not exist"

        return self._randomizers[name]

    def _get_randomizer_parameters(self) -> List[RandomizerParameter]:
        parameters = []

        for randomizer in self.get_randomizers():
            parameters.extend(randomizer.get_parameters())

        return parameters


class ChainedRandomizer(
    Randomizer[TType], RandomizerCollection[RType], Generic[TType, RType],
):
    """
    Base class for randomizer which is composition of multiple randomizers.

    During randomize, it will each randomizer in order on given target, for example

    ChainedRandomizer('cr', [r1, r2, r3]).randomize(target) is equivalent to

    r1.randomize(r2.randomize(r3.randomize(target)))
    """

    def __init__(self, name, randomizers: List[RType]):
        Randomizer.__init__(self, name, enabled=True)
        RandomizerCollection.__init__(self)  # type: ignore

        for randomizer in randomizers:
            self.register_randomizer(randomizer)

    def _randomize(self, target: TType, random_state: np.random.RandomState) -> TType:
        for randomizer in self.get_randomizers():
            target = randomizer.randomize(target, random_state)

        return target

    def get_parameters(self):
        return self._get_randomizer_parameters()

    def reset(self):
        for randomizer in self.get_randomizers():
            randomizer.reset()
