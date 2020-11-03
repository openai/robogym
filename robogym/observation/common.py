import abc
from enum import Enum
from typing import Generic, TypeVar

import numpy as np


class SyncType(Enum):
    """
    Enum which determines how frequent sync() will be called for given observation
    provider. This can be used to control when given observation will change. There
    are 3 different sync types:

    STEP: sync() will be called every step. This is the most common uses case and
        default. Observation provider with STEP sync type will sync observation
        data every step thus result corresponding observation to change every step.

    RESET_GOAL: sync() will be called only when a new goal is generated. This can be
        used to define observations which are constant for each goal. A typical use
        case of this sync type is goal related observations e.g. goal_images

    RESET: Sync() will be called only during environment reset. This can be used
        to define observations which are constant for each episode. Use cases for this
        sync type can be environment constants or per episode noises.

    Note: SyncType with smaller value is considered as subset of SyncType with larger
        value. For example, STEP sync type will also be called during goal reset and
        RESET_GOAL will also be called during env reset.
    """

    STEP = 0  # Sync every step.
    RESET_GOAL = 1  # Sync only during reset goal.
    RESET = 2  # Sync only during reset.


class ObservationProvider(abc.ABC):
    """
    Top level interface for observation provider which has two main responsibilities:

    1. Fetch raw data from observation source at beginning of each environment step.
    2. Provide cached data for observation interface.

    To implement this interface properly, you should make sure that you:

    1. Implement sync to fetch fresh data from observation source and cache it.
    2. Expose cached data via some public method which can be consumed by observation
       interface.

    There are two optional optional methods that can also be implemented to:
    1. Allow recording wrapper to read data for each step.
    2. Allow communication with simulation after each reset.

    No public method other than sync should be allowed fetch data from observation source.
    """

    SYNC_TYPE = SyncType.STEP  # Sync type for this observation provider.

    @abc.abstractmethod
    def sync(self) -> None:
        """
        Sync data from source. This method will only be called at the beginning of each
        environment step.

        This should be the only public method which involves data fetching from
        observation source.
        """
        pass

    def reset(self) -> None:
        """
        Reset internal state of this observation provider. Will be called during
        environment reset.
        """
        pass

    def record_data(self) -> dict:
        """
        Return data which is needed by recording wrapper.
        """
        return {}


PType = TypeVar("PType", bound=ObservationProvider)


class Observation(abc.ABC, Generic[PType]):
    """
    Top level interface for observation which is responsible for return data
    for each keyed environment observation.

    The interface can be subclassed to implement observation which comes from
    different sources.
    """

    def __init__(self, provider: PType):
        self.provider = provider

    @abc.abstractmethod
    def get(self) -> np.ndarray:
        """
        Returns data for this observation which will be provided directly to the policy.

        Note: This method should be implemented on the assumption that underlying raw
        data is already fetched, which means the implementation of this method should

        1. Fast i.e. no network communication.
        2. Readonly i.e. shouldn't change internal state of the environment.
        3. Idempotent i.e. calling this method twice in a row should always return same
           result

        Returns data for this observation.
        """
        pass

    def reset(self):
        """
        This method is called during environment initialization or reset. Most observations
        don't need to do anything here except those which need to update simulation state
        during reset.
        """
        pass

    def record_data(self) -> dict:
        """
        Return data which is needed by recording wrapper.
        """
        return {}
