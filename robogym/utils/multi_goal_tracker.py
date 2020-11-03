import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from numpy.random import RandomState

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.utils.env_utils import InvalidSimulationError

logger = logging.getLogger(__name__)


def _sample_new_goal(goal_sample_func, _obs, _done, _env_crash, info):
    # A helper function for sampling a new goal with try-catch for handling
    # InvalidSimulationError.
    try:
        _obs = goal_sample_func()
        info["goal_reset"] = True
    except InvalidSimulationError:
        _done = True
        _env_crash = True
    return _obs, _done, _env_crash


class MultiGoalTracker:
    def __init__(
        self,
        *,
        mujoco_simulation: SimulationInterface,
        reset_goal_generation_fn: Callable,
        reset_goal_fn: Callable,
        max_timesteps_per_goal=None,
        min_timesteps_per_goal=0,
        success_reward: float = 5.0,
        successes_needed: int = 5,
        success_pause_range_s: Tuple[float, float] = (0.0, 0.0),
        max_steps_goal_unreachable: int = 10,
        check_goal_reachable=False,
        use_goal_distance_reward=True,
        goal_types: Optional[Set[str]] = None,
        random_state: Optional[RandomState] = None
    ):
        """
        Stats tracker for multiple goals.

        :param mj_sim: A mujoco sim object.
        :param max_timesteps_per_goal: How many gym steps we can make before giving up.
        :param min_timesteps_per_goal: How many gym step a goal should persist. This option is
            require to prevent reset from happening too frequently, which may slow down rollouts.
        :param success_reward: Reward when one goal is successful.
        :param successes_needed: Number of goals to achieve to consider the whole episode successful.
        :param success_pause_range_s: Number of seconds to sample the amount of time that success
            needs to stay in successful state to get the reward.
        :param max_steps_goal_unreachable: Number of gym steps we can make before considering the
            goal is unreachable.
        :param check_goal_reachable: If true, check whether goal is reachable from current state.
        :param use_goal_distance_reward: If true, use goal distance reward.
        :param random_state: A numpy random state object.
        """

        self.max_timesteps_per_goal = max_timesteps_per_goal
        self.min_timesteps_per_goal = min_timesteps_per_goal
        self.max_steps_goal_unreachable = max_steps_goal_unreachable
        self.success_pause_range_s = success_pause_range_s
        self.success_reward = success_reward
        self.successes_needed = successes_needed
        self.check_goal_reachable = check_goal_reachable
        self.use_goal_distance_reward = use_goal_distance_reward

        self.goal_types = goal_types if goal_types is not None else ["generic"]

        self.mujoco_simulation = mujoco_simulation
        self.reset_goal_generation_fn = reset_goal_generation_fn
        self.reset_goal_fn = reset_goal_fn

        if random_state is None:
            self._random_state = RandomState()
        else:
            self._random_state = random_state

        self.reset()

    def _set_success_step_range(self) -> List[float]:
        mj_sim = self.mujoco_simulation.mj_sim
        env_step_duration = mj_sim.nsubsteps * mj_sim.model.opt.timestep
        success_step_range = sorted(
            [max(1, s / env_step_duration) for s in self.success_pause_range_s]
        )
        assert len(success_step_range) == 2
        return success_step_range

    def reset(self):
        """Reset the state of MultiGoalTracker for starting a new episode,
        so the entire goal generation is reset.
        """
        self._success_step_range = self._set_success_step_range()
        self._success_steps_required = self._random_state.randint(
            self._success_step_range[0], self._success_step_range[1] + 1
        )

        self._steps = 0
        self._consecutive_steps_with_success = 0
        self._consecutive_steps_with_goal_unreachable = 0
        self._success_and_no_goal_reset = False

        self._goals_so_far = 0
        self._successes_so_far = 0
        self._successes_so_far_by_goal_type = {k: 0 for k in self.goal_types}

        self._steps_since_last_goal = 0
        self._steps_by_goal_type = {k: 0 for k in self.goal_types}

        self._trial_success = False
        self._env_crash = False
        self._sub_goal_is_successful = False

    def reset_goal_steps(self):
        """Reset the stats for a new goal within one episode.
        """
        self._goals_so_far += 1
        self._steps_since_last_goal = 0
        self._success_steps_required = self._random_state.randint(
            self._success_step_range[0], self._success_step_range[1] + 1
        )
        self._consecutive_steps_with_success = 0
        self._consecutive_steps_with_goal_unreachable = 0

    def _steps_per_success(self, total_steps, unsuccessful_steps, successes) -> float:
        # A helper function for computing avg steps per succeeded goals.
        if successes > 0:
            return float(total_steps - unsuccessful_steps) / successes
        else:
            return float(self.max_timesteps_per_goal)

    def initial_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}

        info["trial_success"] = False
        info["sub_goal_is_successful"] = False
        info["sub_goal_type"] = "generic"

        info["goals_so_far"] = 1
        info["successes_so_far"] = 0
        info["successes_so_far_by_goal_type"] = {k: 0 for k in self.goal_types}

        info["steps_since_last_goal"] = 0
        info["steps_by_goal_type"] = {k: 0 for k in self.goal_types}
        info["goal_terminally_unreachable"] = False
        info["steps_per_success"] = self.max_timesteps_per_goal
        info["steps_per_success_by_goal_type"] = {
            k: self.max_timesteps_per_goal for k in self.goal_types
        }

        info["env_crash"] = False

        return info

    def process(
        self,
        obs,
        env_reward,
        done,
        info,
        goal_distance_reward,
        is_successful,
        goal_info: dict,
    ):
        assert isinstance(env_reward, float)

        goal_reachable = goal_info.get("goal_reachable", True)
        solved = goal_info.get("solved", False)
        goal_type = goal_info.get("goal", {}).get("goal_type", "generic")

        success_reward = 0.0

        self._env_crash = False
        self._trial_success = False
        self._sub_goal_is_successful = False

        self._steps += 1
        self._steps_since_last_goal += 1
        self._steps_by_goal_type[goal_type] += 1

        if is_successful:
            self._consecutive_steps_with_success += 1
        else:
            self._consecutive_steps_with_success = 0

        if not goal_reachable:
            self._consecutive_steps_with_goal_unreachable += 1
        else:
            self._consecutive_steps_with_goal_unreachable = 0

        unreachable_state_persists = (
            self._consecutive_steps_with_goal_unreachable
            >= self.max_steps_goal_unreachable
        )

        if (
            self._consecutive_steps_with_success >= self._success_steps_required
            and not self._success_and_no_goal_reset
        ):

            success_reward = self.success_reward

            self._successes_so_far += 1
            self._successes_so_far_by_goal_type[goal_type] += 1

            self._success_and_no_goal_reset = True
            self._sub_goal_is_successful = True

        elif self._steps_since_last_goal >= self.max_timesteps_per_goal:
            # Even if env is not done, the wrapper ends the episode
            done = True

        elif self.check_goal_reachable and unreachable_state_persists:
            # If the goal is not reachable we reset goal generation state.
            obs, done, env_crash = _sample_new_goal(
                self.reset_goal_generation_fn, obs, done, self._env_crash, info
            )

        if (
            self._success_and_no_goal_reset
            and self._steps_since_last_goal >= self.min_timesteps_per_goal
        ):
            self._success_and_no_goal_reset = False
            if self._successes_so_far >= self.successes_needed or solved:
                # Get enough number of successes so it is time to end this episode.
                done = True
                self._trial_success = True
                self._steps_since_last_goal = 0
            else:
                obs, done, self._env_crash = _sample_new_goal(
                    self.reset_goal_fn, obs, done, self._env_crash, info
                )

        goal_reward = goal_distance_reward if self.use_goal_distance_reward else 0.0
        goal_reward -= goal_info.get("penalty", 0.0)
        reward = [env_reward, goal_reward, success_reward]
        info = self.update_info(info, goal_info)

        return obs, reward, done, info

    def update_info(self, info: dict, goal_info: dict) -> dict:
        goal_type = goal_info.get("goal", {}).get("goal_type", "generic")

        unreachable_state_persists = (
            self._consecutive_steps_with_goal_unreachable
            >= self.max_steps_goal_unreachable
        )

        # Extract this here since self.reset_goal() changes the counter. This wrapper is crazy.
        info["consecutive_steps_with_success"] = self._consecutive_steps_with_success
        info["sub_goal_is_successful"] = self._sub_goal_is_successful
        info["sub_goal_type"] = goal_type
        info["steps_since_last_goal"] = self._steps_since_last_goal

        info["trial_success"] = self._trial_success
        info["goals_so_far"] = self._goals_so_far
        info["successes_so_far"] = self._successes_so_far
        info["successes_so_far_by_goal_type"] = self._successes_so_far_by_goal_type
        info["steps_by_goal_type"] = self._steps_by_goal_type
        info["env_crash"] = self._env_crash
        info["goal_terminally_unreachable"] = unreachable_state_persists

        info["steps_per_success"] = self._steps_per_success(
            self._steps, self._steps_since_last_goal, self._successes_so_far
        )
        info["steps_per_success_by_goal_type"] = {
            goal: self._steps_per_success(
                self._steps_by_goal_type[goal],
                int(goal_type == goal) * self._steps_since_last_goal,
                self._successes_so_far_by_goal_type[goal],
            )
            for goal in self._steps_by_goal_type
        }

        return info
