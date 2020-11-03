import time
from copy import deepcopy
from typing import List, Union

import glfw
import numpy as np
from gym.spaces import Box, MultiDiscrete
from mujoco_py import MjViewer, const, ignore_mujoco_warnings

from robogym.utils.misc import pretty


class EnvViewer(MjViewer):
    def __init__(self, env):
        self.env = env
        self.elapsed = [0]
        self.env.reset()
        self.seed = strip_seed(self.env.seed())
        super().__init__(self.env.unwrapped.sim)
        self.num_action = self.env.action_space.shape[0]
        # Needed for gym 0.10.8.
        # TODO: Next 3 lines can be removed in gym >= 0.12.4
        if isinstance(self.num_action, tuple):
            assert len(self.num_action) == 1
            self.num_action = self.num_action[0]
        self.action_mod_index = 0
        self.action = self.zero_action(self.env.action_space)
        self.last_step_result = None

    def zero_action(self, action_space):
        if isinstance(action_space, Box):
            return np.zeros(action_space.shape[0])
        elif isinstance(action_space, MultiDiscrete):
            return action_space.nvec // 2  # assume middle element is "no action" action

    def env_reset(self):
        start = time.time()
        # get the seed before calling env.reset(), so we display the one
        # that was used for the reset.
        self.seed = strip_seed(self.env.seed())
        self.env.reset()
        self.elapsed.append(time.time() - start)
        self.update_sim(self.env.unwrapped.sim)

    def env_reset_goal(self):
        start = time.time()
        # get the seed before calling env.reset(), so we display the one
        # that was used for the reset.
        self.seed = strip_seed(self.env.seed())
        self.env.reset_goal()
        self.elapsed.append(time.time() - start)
        self.update_sim(self.env.unwrapped.sim)

    def key_callback(self, window, key, scancode, action, mods):
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        if key == glfw.KEY_ESCAPE:
            self.env.close()

        # Increment experiment seed
        elif key == glfw.KEY_N:
            self.seed += 1
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
        elif key == glfw.KEY_G:
            self.env_reset_goal()
            self.action = self.zero_action(self.env.action_space)
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
        if key == glfw.KEY_A:
            if isinstance(self.env.action_space, Box):
                self.action[self.action_mod_index] -= 0.05
            elif isinstance(self.env.action_space, MultiDiscrete):
                self.action[self.action_mod_index] = max(
                    0, self.action[self.action_mod_index] - 1
                )
        elif key == glfw.KEY_Z:
            if isinstance(self.env.action_space, Box):
                self.action[self.action_mod_index] += 0.05
            elif isinstance(self.env.action_space, MultiDiscrete):
                self.action[self.action_mod_index] = min(
                    self.env.action_space.nvec[self.action_mod_index] - 1,
                    self.action[self.action_mod_index] + 1,
                )
        elif key == glfw.KEY_K:
            self.action_mod_index = (self.action_mod_index + 1) % self.num_action
        elif key == glfw.KEY_J:
            self.action_mod_index = (self.action_mod_index - 1) % self.num_action
        elif key == glfw.KEY_B:
            if self._has_debug_option():
                self.env.unwrapped.parameters.debug = (
                    not self.env.unwrapped.parameters.debug
                )

        super().key_callback(window, key, scancode, action, mods)

    def render(self):
        super().render()

        # Display applied external forces.
        self.vopt.flags[8] = 1

    def _has_debug_option(self):
        if not hasattr(self.env.unwrapped, "parameters"):
            return False
        return hasattr(self.env.unwrapped.parameters, "debug")

    def _get_action(self):
        return self.action

    def process_events(self):
        """ A hook for subclasses to process additional events. This method is called right
        before updating the simulation. """
        pass

    def run(self, once=False):
        while True:
            self.process_events()
            self.update_sim(self.env.unwrapped.sim)
            self.add_extra_menu()

            with ignore_mujoco_warnings():
                obs, reward, done, info = self._run_step(self._get_action())
                self.last_step_result = deepcopy((obs, reward, done, info))

            self.add_overlay(const.GRID_BOTTOMRIGHT, "done", str(done))
            self.render()

            self.update_aux_display()

            if once:
                return

    def _run_step(self, action):
        return self.env.step(action)

    def add_extra_menu(self):
        self.add_overlay(
            const.GRID_TOPRIGHT,
            "Reset env; (current seed: {})".format(self.seed),
            "N - next / P - previous ",
        )
        self.add_overlay(const.GRID_TOPRIGHT, "Apply action", "A (-0.05) / Z (+0.05)")
        self.add_overlay(
            const.GRID_TOPRIGHT,
            "on action index %d out %d" % (self.action_mod_index, self.num_action),
            "J / K",
        )
        if self._has_debug_option():
            self.add_overlay(
                const.GRID_TOPRIGHT, "De[b]ug", str(self.env.unwrapped.parameters.debug)
            )
        self.add_overlay(
            const.GRID_BOTTOMRIGHT,
            "Reset took",
            "%.2f sec." % (sum(self.elapsed) / len(self.elapsed)),
        )
        self.add_overlay(const.GRID_BOTTOMRIGHT, "Action", pretty(self.action))

    def update_aux_display(self):
        """ Update an auxiliary display/output; called every step after sim has been updated. """
        pass


def strip_seed(seed: Union[List[int], int]) -> int:
    """
    Takes 1-element long list and returns it's element.
    Used to turn a single element list of seeds to its value.
    """
    if isinstance(seed, list):
        assert len(seed) == 1, "Expected list of length 1."
        return seed[0]
    return seed
