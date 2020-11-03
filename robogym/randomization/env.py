from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr
import numpy as np

from robogym.randomization.action import ActionRandomizer
from robogym.randomization.common import (
    ChainedRandomizer,
    Randomizer,
    RandomizerCollection,
    RandomizerParameter,
    VType,
)
from robogym.randomization.observation import ObservationRandomizer
from robogym.randomization.parameters import (
    FloatRandomizerParameter,
    IntRandomizerParameter,
)
from robogym.randomization.sim import SimulationRandomizer

MYPY = False

if MYPY:
    from robogym.robot_env import RobotEnvParameters

    bound = RobotEnvParameters
else:
    bound = None

PType = TypeVar("PType", bound=bound)


def build_randomizable_param(
    default: Optional[VType] = None,
    low: Optional[VType] = None,
    high: Optional[VType] = None,
) -> VType:
    """
    Create parameter attribute which will be automatically registered as randomization
    parameters. If you create an attribute using this function you can directly
    randomize it using ADR config.

    :param default: Default value for this parameter.
    :param low: Low end of range of this parameter.
    :param high: High end of range for this parameter.
    :return: The parameter attribute.
    """
    if low is None:
        low = -np.inf

    if high is None:
        high = np.inf

    def value_in_range(_, attribute, value):
        assert (
            low - 1e-6 <= value <= high + 1e-6
        ), f"Wrong value for {attribute.name}: {value}, must be in [{low}, {high}]"

    kwargs: Dict[str, Any] = {"validator": value_in_range}

    if default is not None:
        kwargs["default"] = default

    return attr.ib(
        metadata={"randomizable": True, "low": low, "high": high}, **kwargs,
    )


class _RandomizableParam(NamedTuple, Generic[VType]):

    name: str

    value_type: Type[VType]

    default: VType

    value_range: Tuple[VType, VType]

    parent_instance: Any


def enumerate_randomizable_params(parameters: PType) -> Iterable[_RandomizableParam]:
    """
    Recursively enumerate all randomizable params under given parameters type.
    return iterable of _RandomizableParam for each randomizable parameter.

    :param parameters: The parameters instance.
    """
    parameters_type = type(parameters)

    for field in attr.fields(parameters_type):
        metadata = field.metadata

        name = field.name

        if metadata.get("randomizable", False):
            assert field.type
            assert field.default is not None

            yield _RandomizableParam(
                name=name,
                value_type=field.type,
                default=getattr(parameters, name),
                value_range=(metadata["low"], metadata["high"]),
                parent_instance=parameters,
            )

        assert field.type, f"No type available for field {field}"

        if attr.has(field.type):
            child_instance = getattr(parameters, name)
            for param in enumerate_randomizable_params(child_instance):
                yield param


class EnvParameterRandomizer(Randomizer[PType]):
    """
    Randomizer which randomize environment parameters which
    is used to initialize environment and simulation. This randomizer
    will be invoked once per environment reset.
    """

    VALUE_TYPE_TO_PARAMETER_TYPE: Dict[type, Type[RandomizerParameter]] = {
        int: IntRandomizerParameter,
        float: FloatRandomizerParameter,
    }

    def __init__(self, parameters: PType):
        super().__init__("parameters")

        for param in enumerate_randomizable_params(parameters):
            randomizer_parameter_type = self.VALUE_TYPE_TO_PARAMETER_TYPE[
                param.value_type
            ]
            self.register_parameter(
                randomizer_parameter_type(param.name, param.default, param.value_range)
            )

    def _randomize(self, target: PType, random_state: np.random.RandomState):
        for param in enumerate_randomizable_params(target):
            setattr(
                param.parent_instance,
                param.name,
                self.get_parameter(param.name).get_value(),
            )

        return target


class EnvObservationRandomizer(ChainedRandomizer[dict, ObservationRandomizer]):
    def __init__(self, randomizers: List[ObservationRandomizer]):
        super().__init__("observation", randomizers)


class EnvActionRandomizer(ChainedRandomizer[dict, ActionRandomizer]):
    def __init__(self, randomizers: List[ActionRandomizer]):
        super().__init__("action", randomizers)


class EnvSimulationRandomizer(ChainedRandomizer[dict, SimulationRandomizer]):
    def __init__(self, randomizers: List[SimulationRandomizer]):
        super().__init__("sim", randomizers)


RType = TypeVar("RType")


class EnvRandomization(RandomizerCollection[Randomizer], Generic[PType]):
    """
    Top level object which contains all randomizers for the environment.

    This class provides the interface for a Domain Randomization (DR) framework
    to interact with the environment and update its randomized parameters to new
    values.

    The top level flow is as below:

    1. Domain Randomization call get_parameters to get all randomized
      env parameters.
    2. Domain Randomization calculates new value for the randomized env parameters.
    3. Domain Randomization calls parameter.set_value to update randomized
      env parameter value.

    Parameter can be defined in jsonnet domain element with the following schema:

    domain_elements_configs: {
        <parameter_path>: {
            args : <parameter_args>
        },
    }

    where parameter path is defined as <dot separated list of randomizer chain>:<parameter name>

    e.g.

    parameters:num_objects  # num_objects parameter for parameters randomizer.
    observation.observation_delay:mean  # mean parameter for observation_delay randomizer under
        observation randomizer.
    sim.gravity:value  # value parameter for gravity randomizer under simulation randomizer.
    """

    def __init__(
        self,
        *,
        parameter_randomizer: EnvParameterRandomizer[PType],
        observation_randomizer: EnvObservationRandomizer,
        action_randomizer: EnvActionRandomizer,
        simulation_randomizer: EnvSimulationRandomizer,
    ):
        super().__init__()
        self.parameter_randomizer = self.register_randomizer(parameter_randomizer)
        self.observation_randomizer = self.register_randomizer(observation_randomizer)
        self.action_randomizer = self.register_randomizer(action_randomizer)
        self.simulation_randomizer = self.register_randomizer(simulation_randomizer)

    def get_parameters(self):
        """
        Get all randomization parameters for the environment.
        """
        return self._get_randomizer_parameters()

    def get_parameter(self, path: str) -> RandomizerParameter:
        parts = path.split(":")
        assert len(parts) == 2, f"Invalid parameter path {path}."
        path, param_name = parts

        randomizer_names = path.split(".")
        parent: Union[RandomizerCollection, Randomizer] = self

        for name in randomizer_names:
            assert isinstance(
                parent, RandomizerCollection
            ), f"{name} of randomizer path {path} is not a randomizer collection."
            parent = parent.get_randomizer(name)

        assert isinstance(parent, Randomizer)
        return parent.get_parameter(param_name)

    def update_parameter(
        self, path: str, value: VType,
    ):
        parameter = self.get_parameter(path)
        parameter.set_value(value)

    def reset(self):
        """
        Reset randomizer state. Will be called during environment reset.
        """
        for randomizer in self.get_randomizers():
            randomizer.reset()
