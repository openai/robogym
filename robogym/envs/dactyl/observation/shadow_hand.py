import abc

import numpy as np

from robogym.observation.mujoco import MujocoObservation
from robogym.robot.shadow_hand.hand_forward_kinematics import FINGERTIP_SITE_NAMES
from robogym.robot.shadow_hand.hand_interface import JOINTS
from robogym.robot.shadow_hand.mujoco.mujoco_shadow_hand import MuJoCoShadowHand


def _update_qpos_and_qvel(sim, qpos=None, qvel=None):
    for i, joint_name in enumerate(JOINTS):
        name = "robot0:" + joint_name
        if name in sim.model.joint_names:
            if qpos is not None:
                sim.data.qpos[sim.model.get_joint_qpos_addr(name)] = qpos[i]

            if qvel is not None:
                sim.data.qvel[sim.model.get_joint_qvel_addr(name)] = qvel[i]


class MujocoShadowHandObservation(MujocoObservation, abc.ABC):
    def __init__(self, provider):
        super().__init__(provider)
        self.hand = MuJoCoShadowHand(self.provider.mujoco_simulation)


class MujocoShadowhandRelativeFingertipsObservation(MujocoShadowHandObservation):
    """
    Mujoco based relative fingertip position observation.
    """

    def get(self) -> np.ndarray:
        """
        Get relative fingertip positions.
        """
        return self.hand.observe().fingertip_positions().flatten()


class MujocoShadowhandAbsoluteFingertipsObservation(MujocoShadowHandObservation):
    """
    Mujoco based absolute fingertip position observation.
    """

    def get(self) -> np.ndarray:
        """
        Get relative fingertip positions.
        """
        fingertip_pos = np.array(
            [
                self.provider.mujoco_simulation.mj_sim.data.get_site_xpos(
                    f"robot0:{site}"
                )
                for site in FINGERTIP_SITE_NAMES
            ]
        )

        return fingertip_pos.flatten()


class MujocoShadowHandJointPosObservation(MujocoShadowHandObservation):
    """
    Mujoco based observation for shadowhand joint positions.
    """

    def get(self) -> np.ndarray:
        """
        Get shadowhand joint positions.
        """
        return self.hand.observe().joint_positions()


class MujocoShadowHandJointVelocityObservation(MujocoShadowHandObservation):
    """
    Mujoco based observation for shadowhand joint velocities.
    """

    def get(self) -> np.ndarray:
        """
        Get shadowhand joint velocities.
        """
        return self.hand.observe().joint_velocities()


class MujocoShadowhandAngleObservation(MujocoShadowHandObservation):
    """
    Mujoco based observation for shadowhand hand angle.
    """

    def get(self) -> np.ndarray:
        """
        Get shadowhand joint velocities.
        """
        return self.provider.mujoco_simulation.get_qpos("hand_angle")
