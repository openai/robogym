import logging

import glfw
import numpy as np
from gym.spaces import MultiDiscrete
from mujoco_py import const

from robogym.robot.composite.controllers.ur_gripper_arm import (
    Direction,
    URGripperArmController,
)
from robogym.utils.misc import pretty
from robogym.viewer.env_viewer import EnvViewer

logger = logging.getLogger(__name__)


class RobotControlViewer(EnvViewer):
    """
    A viewer which support controlling the robot via keyboard control.

    The key bindings are as follows:

    Key binds in EnvViewer (unless override) +

    - UP/DOWN/LEFT/RIGHT: Go backward/forward/left/right.
    - Z/X: Go down/up.
    - C/V: Close/open gripper.
    - Q/W: Rotate wrist CW/CCW.
    - Y/U: Tilt the wrist.
    - -/=: Slow down/speed up gripper moving.
    """

    def __init__(self, env):
        assert isinstance(env.action_space, MultiDiscrete), (
            f"This viewer only works with env with discrete action space, "
            f"got {env.action_space} instead."
        )

        self.controller = URGripperArmController(env.unwrapped)

        super().__init__(env)

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self._press_key_callback(window, key, scancode, mods)
        elif action == glfw.RELEASE:
            self._release_key_callback(window, key, scancode, mods)

    def _press_key_callback(self, window, key, scancode, mods):
        """
        Key callback on press action.
        """
        if key == glfw.KEY_X:
            self.action = self._discretize(self.controller.move_z(Direction.NEG))
        elif key == glfw.KEY_Z:
            self.action = self._discretize(self.controller.move_z(Direction.POS))
        elif key == glfw.KEY_V:
            self.action = self._discretize(self.controller.move_gripper(Direction.NEG))
        elif key == glfw.KEY_C:
            self.action = self._discretize(self.controller.move_gripper(Direction.POS))
        elif key == glfw.KEY_UP:
            self.action = self._discretize(self.controller.move_x(Direction.NEG))
        elif key == glfw.KEY_DOWN:
            self.action = self._discretize(self.controller.move_x(Direction.POS))
        elif key == glfw.KEY_LEFT:
            self.action = self._discretize(self.controller.move_y(Direction.NEG))
        elif key == glfw.KEY_RIGHT:
            self.action = self._discretize(self.controller.move_y(Direction.POS))
        elif key == glfw.KEY_Q:
            self.action = self._discretize(self.controller.rotate_wrist(Direction.POS))
        elif key == glfw.KEY_W:
            self.action = self._discretize(self.controller.rotate_wrist(Direction.NEG))
        elif key == glfw.KEY_Y:
            self.action = self._discretize(self.controller.tilt_gripper(Direction.POS))
        elif key == glfw.KEY_U:
            self.action = self._discretize(self.controller.tilt_gripper(Direction.NEG))
        else:
            super().key_callback(window, key, scancode, glfw.PRESS, mods)

    def _release_key_callback(self, window, key, scancode, mods):
        self.action = self.zero_action(self.env.action_space)

        if key == glfw.KEY_MINUS:
            self.controller.speed_down()
        elif key == glfw.KEY_EQUAL:
            self.controller.speed_up()
        elif key in [
            glfw.KEY_Z,
            glfw.KEY_X,
            glfw.KEY_C,
            glfw.KEY_V,
            glfw.KEY_UP,
            glfw.KEY_DOWN,
            glfw.KEY_LEFT,
            glfw.KEY_RIGHT,
        ]:
            # Don't respond on release for sticky control keys.
            return
        else:
            super().key_callback(window, key, scancode, glfw.RELEASE, mods)

    def _discretize(self, action):
        """
        This assumes the env uses discretized action which is true for all rearrang envs.
        """
        return ((action + 1) * self.env.action_space.nvec / 2).astype(np.int32)

    def add_extra_menu(self):
        self.add_overlay(
            const.GRID_TOPRIGHT, "De[b]ug", str(self.env.unwrapped.parameters.debug)
        )
        self.add_overlay(
            const.GRID_TOPRIGHT,
            "Go backward/forward/left/right",
            "[up]/[down]/[left]/[right] arrow",
        )
        self.add_overlay(const.GRID_TOPRIGHT, "Go up/down", "[Z]/[X]")
        self.add_overlay(const.GRID_TOPRIGHT, "Open/Close gripper", "[V]/[C]")
        self.add_overlay(const.GRID_TOPRIGHT, "Rotate wrist CW/CCW", "[Q]/[W]")
        self.add_overlay(const.GRID_TOPRIGHT, "Tilt Wrist", "[Y]/[U]")
        self.add_overlay(const.GRID_TOPRIGHT, "Slow down/Speed up", "[-]/[=]")

        self.add_overlay(
            const.GRID_BOTTOMRIGHT,
            "Reset took",
            "%.2f sec." % (sum(self.elapsed) / len(self.elapsed)),
        )
        self.add_overlay(const.GRID_BOTTOMRIGHT, "Action", pretty(self._get_action()))
        self.add_overlay(
            const.GRID_BOTTOMRIGHT,
            "Control Mode",
            str(self.env.unwrapped.parameters.robot_control_params.control_mode.value),
        )
        self.add_overlay(
            const.GRID_BOTTOMRIGHT,
            "TCP Solver Mode",
            str(
                self.env.unwrapped.parameters.robot_control_params.tcp_solver_mode.value
            ),
        )
        gripper_force = self.controller.get_gripper_actuator_force()
        if gripper_force is not None:
            self.add_overlay(
                const.GRID_BOTTOMRIGHT, "Gripper Force", "%.2f" % gripper_force
            )
        # - - - - > re-grasp
        regrasp = self.controller.get_gripper_regrasp_status()
        regrasp_str = "Unknown"
        if regrasp is not None:
            regrasp_str = "ON!" if regrasp else "Off"
        self.add_overlay(const.GRID_BOTTOMRIGHT, "Gripper Auto Re-grasp", regrasp_str)
        # < - - - - re-grasp
        self.add_overlay(
            const.GRID_BOTTOMRIGHT,
            "Wrist Angle",
            str(np.rad2deg(self.env.observe()["robot_joint_pos"][5])),
        )

        if self.last_step_result:
            obs, reward, done, info = self.last_step_result
            self.add_overlay(
                const.GRID_TOPRIGHT,
                "Is goal achieved?",
                str(bool(obs["is_goal_achieved"].item())),
            )
            for k, v in info.get("goal_max_dist", {}).items():
                self.add_overlay(const.GRID_TOPRIGHT, f"max_goal_dist_{k}", f"{v:.2f}")

            if info.get("wrist_cam_contacts", {}):
                contacts = info["wrist_cam_contacts"]
                self.add_overlay(
                    const.GRID_TOPRIGHT,
                    "wrist_cam_collision",
                    str(any(contacts.values())),
                )
                active_contacts = [
                    contact_obj
                    for contact_obj, contact_st in contacts.items()
                    if contact_st
                ]
                self.add_overlay(
                    const.GRID_TOPRIGHT, "wrist_cam_contacts", str(active_contacts)
                )
