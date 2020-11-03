import itertools as it
from typing import Dict, List

import attr
from mujoco_py import MjSimState, cymj

from robogym.mujoco.helpers import (
    joint_qpos_ids,
    joint_qpos_ids_from_prefix,
    joint_qvel_ids,
    joint_qvel_ids_from_prefix,
)
from robogym.mujoco.mujoco_xml import MjSim


@attr.s(auto_attribs=True)
class SimulationParameters:
    """
    Containing all parameters needed to build the mujoco simulation.
    """

    pass


class SimulationInterface:
    """
    Base class for domain-specific simulation interfaces tied to particular XML.

    Goal is to transform code interfacing with generic `MjSim` that looks like that:

    hand_angles = sim.data.qpos[hand_angle_idx]
    cube_pos = sim.data.qpos[cube_pos_idx]
    sim.model.actuator_gainprm[actuator_idx] = actuator_kps
    sim.model.actuator_biasprm[actuator_idx] = actuator_kps

    Into more high-level and domain-specific version:

    hand_angles = sim.hand.get_angles()
    cube_pos = sim.get_cube_pos()
    sim.set_actuator_kp(actuator_kps)

    Etc.

    This is a base class that just exposes a few generic utilities to help the subclasses
    implement the abovementioned functionality. By convention, the subclasses should be named
    <Something>Simulation.
    """

    __slots__ = [
        "sim",
        "qpos_idxs",
        "qvel_idxs",
        "synchronization_points",
        "_mujoco_viewer",
    ]

    def __init__(self, sim: MjSim):
        self.sim = sim

        self.qpos_idxs: Dict[str, List[int]] = {}
        self.qvel_idxs: Dict[str, List[int]] = {}

        self.synchronization_points = []  # type: ignore

        self._mujoco_viewer = None

    @property
    def mj_sim(self):
        """ MuJoCo simulation object - alias to make it clearer """
        return self.sim

    @property
    def mujoco_viewer(self):
        """
        Get a nicely-interactive version of the mujoco viewer
        """
        if self._mujoco_viewer is None:
            # Inline import since this is only relevant on platforms
            # which have GLFW support.
            from mujoco_py.mjviewer import MjViewer  # noqa

            self._mujoco_viewer = MjViewer(self.sim)

        return self._mujoco_viewer

    def enable_pid(self):
        """ Enable our custom PID controller code for the actuators with 'user' type """
        cymj.set_pid_control(self.sim.model, self.sim.data)

    ###############################################################################################
    # SUBCLASS REGISTRATION
    def register_joint_group(self, group_name, prefix):
        """ Finds and collect joint ids for given joint name prefix or a list of prefixes. """
        if isinstance(prefix, str):
            self.qpos_idxs[group_name] = joint_qpos_ids_from_prefix(
                self.sim.model, prefix
            )
            self.qvel_idxs[group_name] = joint_qvel_ids_from_prefix(
                self.sim.model, prefix
            )
        elif isinstance(prefix, list):
            self.qpos_idxs[group_name] = list(
                it.chain.from_iterable(
                    joint_qpos_ids_from_prefix(self.sim.model, p) for p in prefix
                )
            )
            self.qvel_idxs[group_name] = list(
                it.chain.from_iterable(
                    joint_qvel_ids_from_prefix(self.sim.model, p) for p in prefix
                )
            )

    def register_joint_group_by_name(self, group_name, name):
        """ Finds and collect joint ids for given joint name or list of names. """
        if isinstance(name, str):
            self.qpos_idxs[group_name] = joint_qpos_ids(self.sim.model, name)
            self.qvel_idxs[group_name] = joint_qvel_ids(self.sim.model, name)
        elif isinstance(name, list):
            self.qpos_idxs[group_name] = list(
                it.chain.from_iterable(joint_qpos_ids(self.sim.model, n) for n in name)
            )
            self.qvel_idxs[group_name] = list(
                it.chain.from_iterable(joint_qvel_ids(self.sim.model, n) for n in name)
            )

    ###############################################################################################
    # GET DATA OUT OF SIM
    def get_qpos(self, group_name):
        """ Gets qpos for a particular group. """
        return self.sim.data.qpos[self.qpos_idxs[group_name]]

    def get_qpos_dict(self, group_names):
        """ Gets qpos dictionary for multiple groups. """
        return {k: self.get_qpos(k) for k in group_names}

    def get_qvel(self, group_name):
        """ Gets qvel for a particular group. """
        return self.sim.data.qvel[self.qvel_idxs[group_name]]

    def get_qvel_dict(self, group_names):
        """ Gets qpos dictionary for multiple groups. """
        return {k: self.get_qvel(k) for k in group_names}

    @property
    def qpos(self):
        """ Returns. copy of full sim qpos. """
        return self.sim.data.qpos.copy()

    @property
    def qvel(self):
        """ Returns copy of full sim qvel. """
        return self.sim.data.qvel.copy()

    def get_state(self) -> MjSimState:
        return self.sim.get_state()

    ###############################################################################################
    # SET DATA IN SIM
    def set_qpos(self, group_name, value):
        """ Sets qpos for a given group. """
        self.sim.data.qpos[self.qpos_idxs[group_name]] = value

    def set_qvel(self, group_name, value):
        """ Sets qpos for a given group. """
        self.sim.data.qvel[self.qvel_idxs[group_name]] = value

    def add_qpos(self, group_name, value):
        """ Sets qpos for a given group. """
        self.sim.data.qpos[self.qpos_idxs[group_name]] += value

    def set_state(self, state: MjSimState):
        self.sim.set_state(state)

    ###############################################################################################
    # INTERFACE TO UNDERLYING SIM
    def step(self, with_udd=True):
        """
        Advances the simulation by calling ``mj_step``.

        If ``qpos`` or ``qvel`` have been modified directly, the user is required to call
        :meth:`.forward` before :meth:`.step` if their ``udd_callback`` requires access to MuJoCo
        state set during the forward dynamics.
        """
        self.sim.step(with_udd=with_udd)
        self.sim.forward()

        # To potentially communicate with other processes
        for point in self.synchronization_points:
            point.synchronize()

    def reset(self):
        """
        Resets the simulation data and clears buffers.
        """
        self.sim.reset()

    def set_constants(self):
        """
        Sets the derived constants of the mujoco simulation.
        """
        self.sim.set_constants()

    def forward(self):
        """
        Computes the forward kinematics. Calls ``mj_forward`` internally.
        """
        self.sim.forward()

    def render(
        self,
        width=None,
        height=None,
        *,
        camera_name=None,
        depth=False,
        mode="offscreen",
        device_id=-1
    ):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.

        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).

        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
            depth=depth,
            mode=mode,
            device_id=device_id,
        )

    ###############################################################################################
    # PROPERTIES
    @property
    def n_substeps(self):
        """ Number of substeps in the mujoco sim """
        return self.sim.nsubsteps
