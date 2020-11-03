import abc

import numpy as np

from robogym.randomization.common import Randomizer


class ActionRandomizer(Randomizer[np.ndarray], abc.ABC):
    """
    Randomizer which randomize action.
    """

    pass
