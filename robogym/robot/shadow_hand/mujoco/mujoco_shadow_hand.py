import numpy as np
from mujoco_py.generated import const

from robogym.mujoco.simulation_interface import SimulationInterface
from robogym.robot.shadow_hand.hand_forward_kinematics import (
    FINGERTIP_SITE_NAMES,
    REFERENCE_SITE_NAMES,
    get_relative_positions,
)
from robogym.robot.shadow_hand.hand_interface import ACTUATORS, Hand, Observation
from robogym.robot.shadow_hand.hand_utils import (
    denormalize_by_limit,
    normalize_by_limits,
)
from robogym.robot.shadow_hand.mujoco.parameter_manager import MuJoCoParameterManager


class MuJoCoObservation(Observation):
    """ Shadow Hand observation coming from the MuJoCo simulation """

    def __init__(
        self, simulation: SimulationInterface, hand_prefix: str, joint_group: str
    ):
        fingers = np.array(
            [
                simulation.mj_sim.data.get_site_xpos(hand_prefix + site)
                for site in FINGERTIP_SITE_NAMES
            ]
        )
        reference = np.array(
            [
                simulation.mj_sim.data.get_site_xpos(hand_prefix + site)
                for site in REFERENCE_SITE_NAMES
            ]
        )

        self._fingertip_positions = get_relative_positions(fingers, reference)
        self._joint_positions = simulation.get_qpos(joint_group).copy()
        self._joint_vel = simulation.get_qvel(joint_group).copy()
        self._time = simulation.mj_sim.data.time

        self._force_limits = simulation.mj_sim.model.actuator_forcerange.copy()

        self._actuator_force = normalize_by_limits(
            simulation.mj_sim.data.actuator_force, self._force_limits
        )

    def joint_positions(self) -> np.ndarray:
        return self._joint_positions

    def joint_velocities(self) -> np.ndarray:
        return self._joint_vel

    def actuator_effort(self) -> np.ndarray:
        return self._actuator_force

    def timestamp(self) -> float:
        return self._time

    def fingertip_positions(self) -> np.ndarray:
        return self._fingertip_positions


class MuJoCoShadowHand(Hand):
    """
    MuJoCo interface to interact with robotic Shadow Hand
    """

    def get_name(self) -> str:
        return "unnamed-mujoco-shadowhand"

    def __init__(
        self, simulation: SimulationInterface, hand_prefix="robot0:", autostep=False
    ):
        """
        :param simulation: simulation interface for the MuJoCo shadow hand xml
        :param hand_prefix: Prefix to add to the joint names while constructing the MuJoCo simulation
        :param autostep: When true, calls step() on the simulation whenever a control is set. This
        should only be used only when the MuJoCoShadowHand is being controlled without a
        SimulationRunner in the loop.
        """
        self.simulation = simulation
        self.hand_prefix = hand_prefix
        self.autostep = autostep
        self.joint_group = hand_prefix + "hand_joint_angles"
        self.simulation.register_joint_group(self.joint_group, prefix=hand_prefix)
        self._parameter_manager = MuJoCoParameterManager(self.mj_sim)

        assert self.mj_sim.model.nu == len(
            ACTUATORS
        ), "Action space must have compatible shape"

        # Are we in the joint control mode or in the force control mode?
        self.joint_control_mode = True
        self.force_limits = self.mj_sim.model.actuator_forcerange.copy()

        # Store copies of parameters in the initial state
        self.gainprm_copy = self.mj_sim.model.actuator_gainprm.copy()
        self.biasprm_copy = self.mj_sim.model.actuator_biasprm.copy()
        self.ctrlrange_copy = self.mj_sim.model.actuator_ctrlrange.copy()

    def parameter_manager(self):
        return self._parameter_manager

    def actuator_ctrl_range_upper_bound(self) -> np.ndarray:
        # We use control range in xml instead of constants to take into account
        # ADR randomization for joint limit.
        return self.mj_sim.model.actuator_ctrlrange[:, 1]

    def actuator_ctrl_range_lower_bound(self) -> np.ndarray:
        # We use control range in xml instead of constants to take into account
        # ADR randomization for joint limit.
        return self.mj_sim.model.actuator_ctrlrange[:, 0]

    @property
    def mj_sim(self):
        """ MuJoCo MjSim simulation object """
        return self.simulation.mj_sim

    def set_position_control(self, control: np.ndarray) -> None:
        assert self.is_position_control_valid(control), f"Invalid control: {control}"

        if not self.joint_control_mode:
            # Need to change the parameters of the motors
            # state.
            self.mj_sim.model.actuator_gaintype[:] = const.GAIN_USER
            self.mj_sim.model.actuator_biastype[:] = const.BIAS_USER

            self.mj_sim.model.actuator_gainprm[:] = self.gainprm_copy
            self.mj_sim.model.actuator_biasprm[:] = self.biasprm_copy
            self.mj_sim.model.actuator_ctrlrange[:] = self.ctrlrange_copy
            self.joint_control_mode = True

        self.mj_sim.data.ctrl[:] = control

        if self.autostep:
            self.mj_sim.step()

    def set_effort_control(self, control: np.ndarray) -> None:
        if self.joint_control_mode:
            # Need to change the parameters of the motors
            self.mj_sim.model.actuator_gaintype[:] = const.GAIN_FIXED
            self.mj_sim.model.actuator_biastype[:] = const.BIAS_NONE

            self.mj_sim.model.actuator_gainprm[:, 0] = 1.0
            self.mj_sim.model.actuator_biasprm[:] = 0
            self.mj_sim.model.actuator_ctrlrange[:] = np.array([[-1.0, 1.0]])
            self.joint_control_mode = False

        # Transform 0 and 1 into force limits
        force_applied = denormalize_by_limit(control, self.force_limits)

        self.mj_sim.data.ctrl[:] = force_applied

        if self.autostep:
            self.mj_sim.step()

    def observe(self) -> Observation:
        return MuJoCoObservation(self.simulation, self.hand_prefix, self.joint_group)
