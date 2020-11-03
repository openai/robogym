import os
from collections import deque
from datetime import datetime

import glfw
import numpy as np
from mujoco_py import const

from robogym.envs.rearrange.holdouts import STATE_DIR
from robogym.viewer.robot_control_viewer import RobotControlViewer


class HoldoutCreationViewer(RobotControlViewer):
    """
    Viewer which can be used to create hold out env for rearrange.

    The key bindings are as follows:

    Key binds in RobotControlViewer +

    - K: Save current state as initial state.
    - L: Save current state as goal state.
    - <: Go back 1 second and pause.
    """

    def __init__(self, env, name):
        super().__init__(env)
        self.name = name
        self.state_dir = os.path.join(STATE_DIR, name)
        self.state_buffer = deque(maxlen=100)
        self.buffer_interval_s = 1

    def env_reset(self):
        super().env_reset()
        self.state_buffer.clear()

    def add_extra_menu(self):
        super().add_extra_menu()
        self.add_overlay(
            const.GRID_TOPRIGHT, "Save current state as initial state", "[K]"
        )
        self.add_overlay(const.GRID_TOPRIGHT, "Save current state as goal state", "[L]")
        self.add_overlay(const.GRID_TOPRIGHT, "Go back 1 second", "[<]")

    def _release_key_callback(self, window, key, scancode, mods):
        if key == glfw.KEY_K:
            self._save_state("initial_state")
        elif key == glfw.KEY_L:
            self._save_state("goal_state")
        elif key == glfw.KEY_COMMA:
            self._revert()
        else:
            super()._release_key_callback(window, key, scancode, mods)

    def _save_state(self, filename_prefix):
        mujoco_simulation = self.env.unwrapped.mujoco_simulation
        data = {
            "obj_pos": mujoco_simulation.get_object_pos(pad=False),
            "obj_quat": mujoco_simulation.get_object_quat(pad=False),
        }

        os.makedirs(self.state_dir, exist_ok=True)

        path = os.path.join(
            self.state_dir,
            f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
        )

        np.savez(path, **data)
        print(f"State saved to {path}")

    def _run_step(self, action):
        # Don't push new checkpoint immediately after revert.
        if (
            not self.state_buffer
            or self.state_buffer[-1].time < self.sim.data.time - self.buffer_interval_s
        ):
            self.state_buffer.append(self.sim.get_state())

        return super()._run_step(action)

    def _revert(self):
        """
        Revert one checkpoint back.
        """

        if self.state_buffer:
            state = self.state_buffer.pop()
            self.sim.set_state(state)
            self.sim.forward()
            self._paused = True

    def render(self):
        with self.env.unwrapped.mujoco_simulation.hide_target():
            super().render()
