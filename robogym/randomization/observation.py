import abc
from typing import Dict

import numpy as np

from robogym.randomization.common import Randomizer


class ObservationRandomizer(Randomizer[Dict[str, np.ndarray]], abc.ABC):
    """
    Randomizer which randomize randomization.
    """

    pass
