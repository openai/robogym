import numpy as np

from robogym.mujoco.forward_kinematics import ForwardKinematics
from robogym.mujoco.mujoco_xml import MujocoXML
from robogym.robot.shadow_hand import hand_interface

FINGERTIP_SITE_NAMES = [
    "S_fftip",
    "S_mftip",
    "S_rftip",
    "S_lftip",
    "S_thtip",
]

REFERENCE_SITE_NAMES = [
    "phasespace_ref0",
    "phasespace_ref1",
    "phasespace_ref2",
]

HAND_KINEMATICS = ForwardKinematics.prepare(
    MujocoXML.parse("robot/shadowhand/main.xml"),
    "hand_mount",
    [1.0, 1.25, 0.15],
    [np.pi / 2, 0, np.pi],
    REFERENCE_SITE_NAMES + FINGERTIP_SITE_NAMES,
    hand_interface.JOINTS,
)

ZERO_JOINT_POSITIONS = HAND_KINEMATICS.compute(np.zeros(len(hand_interface.JOINTS)))
REFERENCE_POSITIONS = ZERO_JOINT_POSITIONS[:3]


def hand_forward_kinematics(qpos, return_joint_pos=False):
    """ Calculate forward kinematics of the hand """
    return HAND_KINEMATICS.compute(qpos, return_joint_pos)[3:]


def get_relative_positions(fingertips_xpos, reference_xpos=REFERENCE_POSITIONS):
    """ Return positions relative to the reference points """
    fingertips_xpos = fingertips_xpos.copy()
    reference_xpos = reference_xpos.copy()
    fingertips_xpos -= reference_xpos[1]
    reference_xpos -= reference_xpos[1]  # This point makes other two orthogonal.
    for idx in [0, 2]:
        reference_xpos[idx] /= np.sqrt(np.sum(np.square(reference_xpos[idx])))

    ort = np.cross(reference_xpos[0], reference_xpos[2])
    m = np.transpose(np.array([reference_xpos[0], ort, reference_xpos[2]]))
    return np.matmul(fingertips_xpos, m)


def compute_forward_kinematics_fingertips(
    joint_positions, reference_xpos=REFERENCE_POSITIONS, return_joint_pos=False
) -> np.ndarray:
    """
    Compute fingertip positions using forward kinematics from joint angles.

    :returns 5x3-element array of current fingertip positions (5 fingertips in 3D)
    """
    fingertip_absolute_positions = hand_forward_kinematics(
        joint_positions, return_joint_pos=return_joint_pos
    )

    return get_relative_positions(
        fingertip_absolute_positions, reference_xpos=reference_xpos
    )
