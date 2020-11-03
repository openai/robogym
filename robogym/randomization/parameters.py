from typing import Optional, Tuple

import numpy as np

from robogym.randomization.common import RandomizerParameter

MAX_INT = int(1e9)  # This is reasonably large enough for any integer parameter.


class IntRandomizerParameter(RandomizerParameter[int]):
    """
    Randomizer parameter of scalar int data type.
    """

    def __init__(
        self,
        name: str,
        initial_value: int,
        value_range: Tuple[int, int] = (-MAX_INT, MAX_INT),
        delta: Optional[int] = None,
    ):
        super().__init__(name, initial_value, value_range, delta=delta)

    @property
    def dtype(self):
        return RandomizerParameter.INT

    @classmethod
    def _convert_type(cls, val: int):
        return int(val)


class FloatRandomizerParameter(RandomizerParameter[float]):
    """
    Randomizer parameter of scalar float data type.
    """

    def __init__(
        self,
        name: str,
        initial_value: float,
        value_range: Tuple[float, float] = (-np.inf, np.inf),
        delta: Optional[float] = None,
    ):
        super().__init__(name, initial_value, value_range, delta=delta)

    @property
    def dtype(self):
        return RandomizerParameter.FLOAT

    @classmethod
    def _convert_type(cls, val: float):
        return np.float32(val)
