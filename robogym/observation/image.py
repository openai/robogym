import abc
from typing import Dict

import numpy as np

from robogym.observation.common import Observation, ObservationProvider


class ImageObservationProvider(ObservationProvider, abc.ABC):
    """
    Interface for observation provider which can provide rendered image.
    """

    @property
    @abc.abstractmethod
    def images(self) -> Dict[str, np.ndarray]:
        pass


class ImageObservation(Observation[ImageObservationProvider]):
    """
    Observation class which provides image observation.
    """

    def get(self):
        return self.provider.images


class MobileImageObservationProvider(ObservationProvider, abc.ABC):
    """
    Interface for observation provider for mobile camera images.
    """

    @property
    @abc.abstractmethod
    def mobile_images(self) -> Dict[str, np.ndarray]:
        pass


class MobileImageObservation(Observation[MobileImageObservationProvider]):
    """
    Observation class which provides mobile image observation.
    """

    def get(self):
        return self.provider.mobile_images
