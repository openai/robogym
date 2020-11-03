import abc
import logging
import random
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import attr
import gym
import gym.spaces as spaces
import numpy as np

from robogym.goal.goal_generator import GoalGenerator
from robogym.mujoco.modifiers.base import Modifier
from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.mujoco.warning_buffer import MjWarningBuffer
from robogym.observation.common import Observation, ObservationProvider, SyncType
from robogym.randomization.action import ActionRandomizer
from robogym.randomization.env import (
    EnvActionRandomizer,
    EnvObservationRandomizer,
    EnvParameterRandomizer,
    EnvRandomization,
    EnvSimulationRandomizer,
)
from robogym.randomization.observation import ObservationRandomizer
from robogym.randomization.sim import SimulationRandomizer
from robogym.robot.robot_interface import Robot
from robogym.robot_exception import RobotException
from robogym.utils.env_utils import gym_space_from_arrays
from robogym.utils.multi_goal_tracker import MultiGoalTracker

logger = logging.getLogger(__name__)

T = TypeVar("T")


def build_nested_attr(t: Callable[..., T], default: Optional[Dict] = None) -> T:
    """
    Build attr.attrib for given user defined attr type which can be nested into
    another attr.attrib class. This is to overcome that issue that nested dict
    won't be automatically deserialized into nested attrib class. For example
    directly define the following attrib hierarchy

    @attrs.s()
    class B:
        name: str

    @attrs.s()
    class A:
        b: B

    and create a instance via A(b={'name': 'foo'}) doesn't work because {'name': 'foo'}
    won't be deserialized into class B. If you create a nested attrib using this function
    as

    @attrs.s()
    class A:
        b: build_nested_attr(B)

    The nested serialization will work correctly.
    """

    if default is None:
        default = {}

    def converter(val):
        new_val = default.copy()

        if isinstance(val, t):
            return val
        else:
            assert isinstance(val, dict)
            new_val.update(val)
            return t(**new_val)

    return attr.ib(converter=converter, default=t(**default))


def get_generic_param_type(type_: Type, i: int, expected_type: Type):
    """
    Get i-th generic type parameter for given class. For example

    if type_ extends Generic[A, B, C] then
        get_generic_param_type(type_, 0) -> A
        get_generic_param_type(type_, 1) -> B

    An error will be raised if the generic type is not a subclass of
    expected type.
    """
    type_param = type_.__orig_bases__[0].__args__[i]
    if isinstance(type_param, TypeVar):  # type: ignore
        type_param = type_param.__bound__

    assert issubclass(type_param, expected_type), (
        f"Parameter class {type_param} is not a subclass of {expected_type}"
        f"Please make sure type arguments for {type_} is correctly specified."
    )

    return type_param


@attr.s(auto_attribs=True)
class RobotEnvParameters:
    # How many steps with random action do we take when the environment is initialized.
    # This should be nonzero for sim2real training configs.
    n_random_initial_steps: int = 10


@attr.s(auto_attribs=True)
class RobotEnvConstants:
    """ Parameters of the env - set once and for all """

    # How many mujoco warnings we store max per episode
    mujoco_warning_capacity: int = 5

    # Whether policy actions represent target position of the fingers or change in positions of
    # the fingers
    relative_action: bool = True

    # Number of bins for discrete action.
    n_action_bins: int = 11

    # If this environment is used by physical robot.
    physical: bool = False

    #####################
    # Curriculum settings

    # Number of mujoco simulation steps per environment step
    mujoco_substeps: int = 10

    # timestep for each mujoco simulation step.
    mujoco_timestep: float = 0.002

    ####################
    # Observation related constants.

    # All enabled observation providers. Note that mujoco and goal doesn't
    # need to be specified here because they are assumed to exist for all
    # environments.
    observation_providers: list = []

    # Map between observation key to observation provider. This can used to
    # override predefined default_observation_map to support different
    # provider for each observation.
    observation_configs: dict = {}

    #####################
    # Wrapper settings.
    # Below are constants for commonly used wrappers e.g. recording wrapper.

    # If randomize the environment.
    randomize: bool = True

    #####################
    # Multiple successes settings

    # Threshold for success conditions
    success_threshold: dict = {}

    # How many successes needed to stop the episode.
    successes_needed: int = 5

    # Reward for a single success if all the distances are within the `success_threshold`
    success_reward: float = 5.0

    # Number of seconds to sample the amount of time (in seconds) that success needs to
    # stay in successful state to be counted. This defines the range (in seconds) and the
    # actual pause time is sampled uniformly from within this range.
    success_pause_range_s: Tuple[float, float] = (0.0, 0.0)

    # Max timesteps for each goal until timeout.
    max_timesteps_per_goal: Optional[int] = None

    # If true, check if goal is reachable from current state after each step.
    check_goal_reachable: bool = False

    # How many gym steps we can make before considering the goal is unreachable.
    max_steps_goal_unreachable: int = 10

    # If yes, include `goal_distance_reward` into the final env reward.
    # How `goal_distance_reward` is calculated depends on the specific env.
    use_goal_distance_reward: bool = True

    #####################
    # Randomizer settings

    # Path for all enabled randomizers. Randomizer path is defined as
    # dot separated list of randomizer names along the chain e.g.
    # parameters  # Top level parameters randomizer
    # observation.observation_delay  # Observation delay randomizer
    #   under top level observation randomizer
    randomizers: List[str] = []


class ObservationMapValue(Dict[str, Type[Observation]]):
    """
    Value type for observation map. Example ways to instantiate
    this type are:

    ObservationMapValue({'mujoco': 'MujocoCubePosObservation'})

    ObservationMapValue({
        'mujoco': MujocoCubePosObservation,
        'phasespace': PhasespaceCubePosObservation,
    }, default='phasespace')

    If there is only one provider available for given observation, default
    doesn't need to be explicitly specified.

    default provider will be used if no provider is explicitly specified for
    this observation key in RobotEnvConstants.observation_configs.
    """

    def __init__(
        self,
        providers_to_obs_class: Dict[str, Type[Observation]],
        default: Optional[str] = None,
    ):
        """
        :param providers_to_obs_class: Map between provider name to observation class.
        :param default: Name of default observation provider.
        """
        super().__init__(**providers_to_obs_class)
        self._default = default

    def update(self, target: "ObservationMapValue", **kwargs):  # type: ignore
        super().update(target, **kwargs)
        self._default = target._default or self._default

    def get_default(self):
        if self._default is None:
            assert len(self) == 1, (
                f"There are multiple providers available for this observation: "
                f"{self.keys()}, but no default."
            )
            return list(self.keys())[0]
        else:
            assert self._default in self, f"Invalid default provider {self._default}"
            return self._default


class RobotEnvObserver:
    """
    Class to encapsulate observation related logic.
    """

    def __init__(
        self,
        mujoco_simulation: SimulationInterface,
        providers: Dict[str, ObservationProvider],
        observations: Dict[str, Observation],
    ):
        self.mujoco_simulation = mujoco_simulation
        self.providers = providers
        self.observations = observations
        self._observation_space = None

        self.reset()

    def reset(self):
        """
        Reset providers and observations.
        """
        for provider in self.providers.values():
            provider.reset()

        for observation in self.observations.values():
            observation.reset()

    def sync(self, sync_type=SyncType.STEP):
        """
        Sync all observation providers. Ideally this should only be called once every step.
        """
        # Sync observation providers to make sure they return latest
        # information.
        for provider in self.providers.values():
            if provider.SYNC_TYPE.value <= sync_type.value:
                provider.sync()

    def observe(self) -> dict:
        """
        Return read only view of current observation of the environment. Calling this method
        multiple times in a row should yield same result without any side effect.
        """
        return OrderedDict((key, o.get()) for key, o in self.observations.items())

    def record_data(self) -> dict:
        """
        Return data for recording wrapper.
        """
        data = {}
        for provider in self.providers.values():
            data.update(provider.record_data())

        for obs in self.observations.values():
            data.update(obs.record_data())

        return data


class EnvMeta(abc.ABCMeta):
    """
    Metaclass for the environment to properly initialize environment after "basic" init has
    been done.

    This can ensure initialize() is called after __init__ of this class and all subclasses have
    finished. To follow the pattern please make sure:

    - Put logic in __init__ if it should be called before __init__of all subclasses are called.
    - Put logic in initialize if it should be called after __init__ of all subclasses are called.
    """

    def __call__(cls, *args, **kwargs):
        """Called when you call MyNewClass() """
        obj = type.__call__(cls, *args, **kwargs)
        obj.initialize()
        return obj


PType = TypeVar("PType", bound=RobotEnvParameters)
CType = TypeVar("CType", bound=RobotEnvConstants)
SType = TypeVar("SType", bound=SimulationInterface)


class RobotEnv(gym.Env, Generic[PType, CType, SType], metaclass=EnvMeta):
    """
    Base class for robot environments.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        parameters: PType,
        constants: CType,
        mujoco_simulation: SType,
        mujoco_modifiers: Dict[str, Modifier],
        robot: Robot,
        goal_generation: GoalGenerator,
        randomization: EnvRandomization,
        starting_seed: Optional[int] = None,
    ):
        self.parameters = parameters
        self.constants = constants
        self.mujoco_simulation = mujoco_simulation
        self._cached_robot = robot
        self.randomization = randomization

        self.latest_action_metadata: Dict[str, object] = {}

        # Container for potential mujoco errors/warnings
        self.warning_buffer = MjWarningBuffer(
            maxlen=self.constants.mujoco_warning_capacity
        )
        self.warning_buffer.enter()

        # Random seed state
        self._last_seed = (
            starting_seed
            if starting_seed is not None
            else random.randint(0, 2 ** 32 - 1)
        )

        self._random_state = np.random.RandomState(self._last_seed)

        # Episode step counter
        self.t = 0

        # "Goal" information
        self._previous_goal_distance = None
        self._goal = None  # type: ignore
        self._goal_info_cache = None
        self.goal_generation = goal_generation

        # Environment observation/action spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(robot.zero_control()),), dtype=np.float32
        )

        self.action_space.seed(self._last_seed)

        # Will be initialized later
        self.observer: Optional[RobotEnvObserver] = None

        self._observation_space = None

        # List of mujoco modifiers
        self.modifiers: List[Tuple[str, Any]] = []

        self.reward_names = ["env", "goal", "success"]

        self.last_update_time = time.time()

        self.mujoco_simulation.mj_sim.render_callback = self._render_callback

        for parameter_name, modifier in mujoco_modifiers.items():
            self.register_modifier(parameter_name, modifier)

        # Set up trackers for multi-successes goals
        self.multi_goal_tracker = MultiGoalTracker(
            mujoco_simulation=self.mujoco_simulation,
            reset_goal_generation_fn=self.reset_goal_generation,
            reset_goal_fn=self.reset_goal,
            max_timesteps_per_goal=self.constants.max_timesteps_per_goal,
            success_reward=self.constants.success_reward,
            successes_needed=self.constants.successes_needed,
            success_pause_range_s=self.constants.success_pause_range_s,
            max_steps_goal_unreachable=self.constants.max_steps_goal_unreachable,
            check_goal_reachable=self.constants.check_goal_reachable,
            use_goal_distance_reward=self.constants.use_goal_distance_reward,
            goal_types=self.goal_generation.goal_types(),
            random_state=self._random_state,
        )

    def initialize(self):
        """
        Initialization to be ran after __init__ of this class and all the subclasses is finished
        """
        self._setup_simulation_from_parameters()
        if "orrb" in self.constants.observation_providers:
            self._reset()
        self._goal = self._next_goal()
        self.update_goal_info()

        self.observer = self._build_observer()

    ###############################################################################################
    # Initialize observation related stuff.
    def _build_observer(self):
        """
        Initialize observation providers for the environment.
        """
        providers = self._build_observation_providers()
        observations = self._build_observations(providers)

        return RobotEnvObserver(self.mujoco_simulation, providers, observations)

    def _build_observations(self, providers: Dict[str, ObservationProvider]):
        observation_map = self._default_observation_map()

        observations: Dict[str, Observation] = {}

        for obs_key, obs_map in observation_map.items():
            provider_name = self.constants.observation_configs.get(
                obs_key, obs_map.get_default()
            )

            assert provider_name in providers, (
                f"Observation {obs_key} is configured to use provider {provider_name} "
                f"which is not enabled. Enabled providers are: "
                f"{providers.keys()}"
            )

            obs_class = obs_map[provider_name]
            provider = providers[provider_name]
            observations[obs_key] = obs_class(provider)

        return observations

    ###############################################################################################
    # Internal API - must be overridden - observation
    @abc.abstractmethod
    def _build_observation_providers(self) -> Dict[str, ObservationProvider]:
        """
        Build all observation providers for this environment.
        """
        pass

    @abc.abstractmethod
    def _default_observation_map(self) -> Dict[str, ObservationMapValue]:
        """
        Return map between observation key and map between observation provider
        and observation class. This map should only contain simulation based observations.

        See implementation of other environments for example.
        """
        pass

    ###############################################################################################
    # Internal API - may be overridden - mujoco simulation interface
    def _get_simulation_reward_with_done(self, info: dict) -> Tuple[float, bool]:
        """ Return current reward and whether episode is finished
        :param info: Current info dict for this env step.
        """
        return 0.0, False

    def _get_simulation_info(self) -> dict:
        """ Return extra information about the environment """
        # Just a stub for now
        return {}

    def _set_action(self, action):
        """ Set action for the hand"""
        action = np.asarray(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self.robot.denormalize_position_control(
            position_control=action, relative_action=self.constants.relative_action,
        )
        self.robot.set_position_control(ctrl)

    ###############################################################################################
    # Internal API - To be exposed to the child classes and wrappers
    def register_modifier(self, parameter_name: str, modifier_object: Modifier):
        """ Register given modifier for a given parameter value """
        modifier_object.initialize(self.sim)
        self.modifiers.append((parameter_name, modifier_object))

    def _setup_simulation_from_parameters(self):
        """
        Set all the simulation parameters from the current settings.

        You may override it or just leave it as it is for a very basic setup.
        """
        for param_name, modifier in self.modifiers:
            modifier(getattr(self.parameters, param_name))

    def _render_callback(self, _sim, _viewer):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _reset(self):
        """
        Custom reset logic that can be overloaded by subclasses.
        """
        pass

    def _act(self, action):
        """Perform a gym environment action on the simulation.

        By default, this just passes the action through to the simulation, but it can be
        overridden by subclasses for more complicated logic."""
        self._set_action(action)

    ###############################################################################################
    # Internal API - goal interface

    def _next_goal(self):
        """ Return next goal for the robot. Return a dictionary representing that goal """
        current_state = self.goal_generation.current_state()

        return self.goal_generation.next_goal(self._random_state, current_state)

    def _calculate_goal_distance_reward(
        self, previous_goal_distance, goal_distance
    ) -> float:
        dist_reward = sum(
            [
                previous_goal_distance[k] - goal_distance[k]
                for k in self.constants.success_threshold
            ]
        )
        return dist_reward

    def _calculate_goal_distance(self, current_state):
        goal_distance = self.goal_generation.goal_distance(self._goal, current_state)
        return goal_distance

    def _is_successful_state(self, current_state):
        goal_distance = self._calculate_goal_distance(current_state)
        return self._is_successful(goal_distance)

    def _is_successful(self, goal_distance):
        return all(
            [
                np.all(goal_distance[k] < self.constants.success_threshold[k])
                for k in self.constants.success_threshold
            ]
        )

    def _get_goal_info(self):
        """ Calculate information about current state of the goal """
        current_state = self.goal_generation.current_state()
        goal_distance = self._calculate_goal_distance(current_state)
        relative_goal = goal_distance.pop("relative_goal", None)

        goal_reachable = self.goal_generation.goal_reachable(self._goal, current_state)

        # In case it's the first time, just set it to current goal distance
        if self._previous_goal_distance is None:
            self._previous_goal_distance = goal_distance

        goal_distance_reward = self._calculate_goal_distance_reward(
            self._previous_goal_distance, goal_distance
        )

        self._previous_goal_distance = goal_distance

        optional_keys = {}
        is_successful = (
            self._is_successful(goal_distance)
            or self.goal_generation.reached_terminal_state
        )

        optional_keys["goal_max_dist"] = {
            k: np.max(goal_distance[k]) for k in self.constants.success_threshold
        }
        optional_keys["goal_failures"] = {
            k: np.sum(goal_distance[k] > self.constants.success_threshold[k])
            for k in self.constants.success_threshold
        }

        goal_info = {
            "current_state": current_state,
            "goal_dist": {key: np.sum(dist) for key, dist in goal_distance.items()},
            "goal_achieved": is_successful,
            "goal": self._goal,
            "penalty": current_state.get("penalty", 0.0),
            "goal_reachable": goal_reachable,
            "solved": self.goal_generation.reached_terminal_state,
        }

        goal_info.update(optional_keys)

        if relative_goal is not None:
            for key, val in relative_goal.items():
                goal_info[f"rel_goal_{key}"] = val.copy()

        return goal_distance_reward, is_successful, deepcopy(goal_info)

    @property
    def _is_goal_achieved(self) -> bool:
        """
        Return if current goal is achieved.
        """
        assert self._goal_info_cache
        return self._goal_info_cache[1]

    @property
    def _goal_info_dict(self) -> dict:
        """
        Return dict containing info e.g. goal state, relative goal state etc. for current goal
        """
        assert self._goal_info_cache
        return self._goal_info_cache[2]

    ###############################################################################################
    # Fully internal methods.
    def _synchronize_step_time(self):
        """
        Synchronize step time based on current threshold.
        """
        # Figure out the frequency with which we would step the underlying simulation (in seconds).
        delta_threshold_s = self._get_wall_clock_step_time_threshold()

        # Sleep until we hit the time step (delta threshold)
        current_time = time.time()
        wait_time = max(0.0, delta_threshold_s - (current_time - self.last_update_time))
        time.sleep(wait_time)

        update_time = time.time()

        self.last_update_time = update_time

    def _get_wall_clock_step_time_threshold(self):
        """
        Return the minimum threshold wall clock step time.
        """
        if self.constants.physical:
            sim = self.mujoco_simulation.mj_sim
            return float(sim.nsubsteps) * sim.model.opt.timestep
        else:
            # No minimum threshold for simulation.
            return 0

    def _observe_sync(self, sync_type=SyncType.STEP):
        """
        Sync all observation providers and return latest observation. Ideally this
        should only be called once every step.
        """
        self.mujoco_simulation.forward()
        self.update_goal_info()
        self.observer.sync(sync_type=sync_type)

        observations = self.observe()

        # Notify each robot of the new observations.
        # This path allows robots to do whatever they need with this observations update. For example, certain arms
        # that use sub-simulations can sync them with the new observations.
        self.robot.on_observations_updated(observations)

        return observations

    ###############################################################################################
    # External API - to establish communication with other parts of the system
    @property
    def observation_space(self):
        if self._observation_space is None:
            obs = self._observe_sync(sync_type=SyncType.RESET)
            self._observation_space = gym_space_from_arrays(obs)

        return self._observation_space

    @property
    def robot(self):
        return self._cached_robot

    @property
    def warnings(self):
        """ List of MuJoCo warnings """
        return self.warning_buffer.warnings

    @property
    def sim(self):
        """ Define this property so that it plays nicely with other parts of the system """
        return self.mujoco_simulation.sim

    def observe(self) -> dict:
        """
        Return read only view of current observation of the environment. Calling this method
        multiple times in a row should yield same result without any side effect.

        There are two ways to provide environment observations:

        1.  Via _observe_simple(): You can return observation using classic style gym observe
            method where you can directly return data associate with each observation. The downside
            of this approach it's not easy to change the observation based on different environment
            configuration e.g. sim vs physical.

            It's recommended to only return observations which are cheap to fetch and consistent
            across all environment configurations in _observe_simple().

        2.  Via observer.observe(): This comes with a bit extra overhead as it requires creating
            observation provider and observation classes. But it has better structured support for
            observations can vary based on environment configuration or needs to be updated not at
            every step.

            It's recommended to handle all polymorphic observations in observer.
        """
        assert self.observer

        obs = self._observe_simple()
        obs.update(self.observer.observe())

        return self.randomization.observation_randomizer.randomize(
            obs, self._random_state
        )

    def _observe_simple(self):
        """
        Returns simple observations which can always be fetched regardless of which observation
        providers are specified. This function is called every time observe is called so everything
        here needs to cheap to fetch.

        Good candidates to be included here are mujoco simulation state, info from goal dict etc.
        """
        return {}

    ###############################################################################################
    # External API - gym Env
    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Warning: Do not overload this method! It ensures operations happen in certain order.
        please add custom reset logic to _reset.

        Returns: observation (object): the initial observation of the
            space.
        """
        # Reset time counter
        self.t = 0

        # Reset randomization
        self.randomization.reset()

        # Randomize parameters.
        self.parameters = self.randomization.parameter_randomizer.randomize(
            self.parameters, self._random_state
        )

        self._reset()

        # Randomize simulation. Because sim is recreated in self._reset(),
        # simulation_randomizer.randomize should be called after the _reset.
        self.randomization.simulation_randomizer.randomize(
            self.mujoco_simulation.mj_sim, self._random_state
        )

        # reset observer.
        self.observer.reset()

        # Reset multi goal tracker for a new episode.
        self.multi_goal_tracker.reset()

        # Reset state of goal generation.
        return self.reset_goal_generation(sync_type=SyncType.RESET)

    def step_finalize(self, obs, env_reward, done, info):
        # Process the output by multiple goal tracker.
        goal_distance_reward, is_successful, goal_info = self.goal_info()
        obs, reward, done, info = self.multi_goal_tracker.process(
            obs, env_reward, done, info, goal_distance_reward, is_successful, goal_info
        )
        info.update(goal_info)

        return obs, reward, done, info

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will
            return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and
            sometimes learning)
        """
        action = self.randomization.action_randomizer.randomize(
            action, self._random_state
        )

        robot_exception = None
        try:
            self._act(action)
        except RobotException as re:
            logger.error(
                f"Robot raised exception: {str(re)}. This will finish the current episode."
            )
            robot_exception = re

        if not self.constants.physical:
            # We don't need to do stepping for physical roll out.
            self.mujoco_simulation.step()

        self._synchronize_step_time()
        self.t += 1

        obs, reward, done, info = self.get_observation(robot_exception=robot_exception)
        obs, reward, done, info = self.step_finalize(obs, reward, done, info)
        return obs, reward, done, info

    def get_info_finalize(self, info: dict) -> dict:
        """This is called to append more info into the original `info` dict, including stats
        from multi-goal tracker or self-play tracker.

        This should neither affect observation nor trigger any tracker process calls.
        Multiple calls without step() in-between should yield same results.
        """
        _, _, goal_info = self.goal_info()
        info = self.multi_goal_tracker.update_info(info, goal_info)
        info.update(goal_info)
        return info

    def get_observation(self, robot_exception=None):
        # Get current state to return to the user
        obs = self._observe_sync()

        if robot_exception is None:
            info, env_reward, done = self.get_simulation_info_reward_with_done()
        else:
            # Robot raised an exception, we can't continue with this tick, since we can't assume that the robot
            # performed the action.
            info = {"robot_raised_exception": True}
            done = True
            env_reward = 0.0  # TBD consider adding a penalty if useful

        info = self.get_info_finalize(info)
        return obs, env_reward, done, info

    def get_info(self):
        obs, reward, done, info = self.get_observation()
        return info

    def get_simulation_info_reward_with_done(self):
        info = self._get_simulation_info()
        env_reward, done = self._get_simulation_reward_with_done(info)
        assert isinstance(env_reward, float)

        return info, env_reward, done

    def update_goal_info(self):
        """
        Re-computes and caches the current goal_info. Usually you do not have to call this
        since `step` will automatically do so. However, if you manipulate the state externally,
        you will have to call this method after.
        """
        self._goal_info_cache = self._get_goal_info()

    def reset_goal(self, update_seed=False, sync_type=SyncType.RESET_GOAL):
        """ Reset the goal of the environment """

        # Reset stats for one goal in the same episode.
        self.multi_goal_tracker.reset_goal_steps()

        # Randomize a target for the robot
        self._goal = self._next_goal()
        self._previous_goal_distance = None

        return self._observe_sync(sync_type=sync_type)

    def reset_goal_generation(self, sync_type=SyncType.RESET_GOAL):
        """ Reset state of goal generation. """

        self.goal_generation.reset(self._random_state)
        return self.reset_goal(sync_type=sync_type)

    def goal_info(self):
        """ Return info about the goal """
        return self._goal_info_cache

    def render(self, mode="human", width=500, height=500):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            width (int): Width of the rendered image.
            height (int): Height of the rendered image.
        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == "human":
            return self.mujoco_simulation.mujoco_viewer.render()
        elif mode == "rgb_array":
            return self.mujoco_simulation.render(width=width, height=height)
        else:
            raise ValueError("Unsupported mode %s" % mode)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        if isinstance(seed, list):
            # Support list of seeds as required by Gym.
            seed = seed[0]
        elif isinstance(seed, int):
            pass
        elif seed is not None:
            # If seed is None, we just return current seed.
            raise ValueError("Seed must be an integer.")

        if seed is not None:
            self._last_seed = seed
            self._random_state.seed(seed)
            self.action_space.seed(seed)

        # Return list of seeds to conform to Gym specs
        return [self._last_seed]

    def apply_wrappers(self, **wrapper_params):
        """
        Apply wrappers to the environment.
        """
        return self

    @classmethod
    @abc.abstractmethod
    def build_robot(cls, mujoco_simulation: SType, physical: bool) -> Robot:
        """
        Build robot for this environment.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def build_goal_generation(
        cls, constants: CType, mujoco_simulation: SType
    ) -> GoalGenerator:
        """
        Build goal generation for this environment.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def build_simulation(cls, constants: CType, parameters: PType) -> SType:
        """
        Build simulation for this environment.
        """
        pass

    @classmethod
    def build_mujoco_modifiers(cls) -> Dict[str, Modifier]:
        """
        Build mujoco modifiers for this environment.
        """
        return OrderedDict()

    @classmethod
    def build_randomization(
        cls, constants: CType, parameters: PType
    ) -> EnvRandomization:
        """
        Build simulation for this environment.
        """
        return EnvRandomization(
            parameter_randomizer=cls.build_parameter_randomizer(constants, parameters),
            observation_randomizer=EnvObservationRandomizer(
                cls.build_observation_randomizers(constants)
            ),
            action_randomizer=EnvActionRandomizer(
                cls.build_action_randomizers(constants)
            ),
            simulation_randomizer=EnvSimulationRandomizer(
                cls.build_simulation_randomizers(constants)
            ),
        )

    @classmethod
    def build_parameter_randomizer(
        cls, constants: CType, parameters: PType
    ) -> EnvParameterRandomizer:
        """
        Build parameter randomizer for the environment.
        """
        return EnvParameterRandomizer(parameters)

    @classmethod
    def build_observation_randomizers(cls, constants) -> List[ObservationRandomizer]:
        """
        Build observation randomizers for the environment.
        """
        return []

    @classmethod
    def build_action_randomizers(cls, constants) -> List[ActionRandomizer]:
        """
        Build action randomizers for the environment.
        """
        return []

    @classmethod
    def build_simulation_randomizers(cls, constants) -> List[SimulationRandomizer]:
        """
        Build simulation randomizers for the environment.
        """
        return []

    @classmethod
    def build(
        cls,
        parameters=None,
        constants=None,
        wrapper_params=None,
        starting_seed=None,
        apply_wrappers=True,
    ):
        """
        Construct a dactyl environment together with a set of common wrappers.
        """
        if parameters is None:
            parameters = {}

        if constants is None:
            constants = {}

        if wrapper_params is None:
            wrapper_params = {}

        parameter_class = get_generic_param_type(cls, 0, RobotEnvParameters)
        constant_class = get_generic_param_type(cls, 1, RobotEnvConstants)

        if isinstance(parameters, dict):
            parameters = parameter_class(**parameters)

        if isinstance(constants, dict):
            constants = constant_class(**constants)

        mujoco_simulation = cls.build_simulation(constants, parameters,)

        mujoco_modifiers = cls.build_mujoco_modifiers()

        goal_generation = cls.build_goal_generation(constants, mujoco_simulation)

        randomization = cls.build_randomization(constants, parameters)

        for name in constants.randomizers:
            randomization.get_randomizer(name).enable()

        robot = cls.build_robot(
            mujoco_simulation=mujoco_simulation, physical=constants.physical
        )
        env = cls(
            parameters=parameters,
            constants=constants,
            mujoco_simulation=mujoco_simulation,
            mujoco_modifiers=mujoco_modifiers,
            robot=robot,
            goal_generation=goal_generation,
            randomization=randomization,
            starting_seed=starting_seed,
        )

        if apply_wrappers:
            env = env.apply_wrappers(**wrapper_params)

        return env

    @classmethod
    def _get_default_wrappers(cls):
        return None
