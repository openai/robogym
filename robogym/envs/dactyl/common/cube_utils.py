import math

import numpy as np

from robogym.mujoco.helpers import joint_qpos_ids_from_prefix
from robogym.utils import rotation

PARALLEL_QUATS = [
    rotation.quat_normalize(rotation.euler2quat(r))
    for r in rotation.get_parallel_rotations()
]


DEFAULT_CAMERA_NAMES = ["vision_cam_top", "vision_cam_right", "vision_cam_left"]


def on_palm(sim):
    """ Determines if the cube is on the palm of the hand."""
    sim.forward()
    cube_middle_idx = sim.model.site_name2id("cube:center")
    cube_middle_pos = sim.data.site_xpos[cube_middle_idx]
    is_on_palm = cube_middle_pos[2] > 0.04
    return is_on_palm


def uniform_z_aligned_quat(random):
    """ Produces a random quaternion with the red face on top. """
    axis = np.asarray([0.0, 0.0, 1.0])
    angle = random.uniform(-np.pi, np.pi)
    quat = rotation.quat_from_angle_and_axis(angle, axis)
    return rotation.quat_normalize(quat)


def face_up(sim, geom_names):
    """ Return the index of the face which is oriented up. """
    face_geom_z = [sim.data.get_geom_xpos(name)[2] for name in geom_names]

    return np.argmax(face_geom_z)


def face_up_quats(sim, ball_joint, geom_names):
    """ Returns a dict of parallel quats in which the given faces are up. """
    goal_quat = {}

    cube_quat_idxs = joint_qpos_ids_from_prefix(sim.model, ball_joint)
    initial_quat = sim.data.qpos[cube_quat_idxs]

    sim.data.qpos[cube_quat_idxs] = rotation.quat_identity()

    for i, geom in enumerate(geom_names):
        geom_z = []
        for p in PARALLEL_QUATS:
            sim.data.qpos[cube_quat_idxs] = p
            sim.forward()
            geom_z.append(sim.data.get_geom_xpos(geom)[2])
        goal_quat[i] = PARALLEL_QUATS[np.argmax(geom_z)]

    assert len(goal_quat.keys()) == len(geom_names)

    sim.data.qpos[cube_quat_idxs] = initial_quat
    sim.forward()

    return goal_quat


def rotated_face(
    face_angles, face_to_shift, random, round_target_face, directions=["cw", "ccw"]
):
    """
    Return a new set of face angles, which correspond to the original but with the given
    face rotated.
    """
    clockwise = math.pow(-1, face_to_shift)

    rotation_directions = {
        "cw": clockwise,
        "ccw": -clockwise,
    }

    directions = [math.pi / 2 * rotation_directions[d] for d in directions]

    rotated_face = face_angles.copy()

    if random.uniform() < float(round_target_face):
        rotated_face[face_to_shift] += random.choice(directions)
        rotated_face = rotation.normalize_angles(rotated_face)
        rotated_face = rotation.round_to_straight_angles(rotated_face)
    else:
        directions += [0.0]
        rotated_face[face_to_shift] += random.uniform(min(directions), max(directions))
        rotated_face = rotation.normalize_angles(rotated_face)

    return rotated_face


def rotated_face_with_angle(
    face_angles, face_to_shift, random, round_target_face, directions=["cw", "ccw"]
):
    """
    Return a new set of face angles, which correspond to the original but with the given
    face rotated.

    :param face_angles: Numpy array of current angles of cube faces
    :param face_to_shift: Index of the face that we are about to rotate
    :param random: Random state used to sample pseudorandom numbers
    :param round_target_face: Boolean of floating point probability if the face should be rotated
           by 90 degrees or by any uniform angle within range
   :param directions: Specify which direction rotations are allowed, supported values are
           'cw' and 'ccw'
    """
    clockwise = math.pow(-1, face_to_shift)

    rotation_directions = {
        "cw": clockwise,
        "ccw": -clockwise,
    }

    directions = [math.pi / 2 * rotation_directions[d] for d in directions]

    rotated_face = face_angles.copy()

    if random.uniform() < float(round_target_face):
        rotation_angle = random.choice(directions)

        rotated_face[face_to_shift] += rotation_angle
        rotated_face = rotation.normalize_angles(rotated_face)
        rotated_face = rotation.round_to_straight_angles(rotated_face)
    else:
        directions += [0.0]
        rotation_angle = random.uniform(min(directions), max(directions))

        rotated_face[face_to_shift] += rotation_angle
        rotated_face = rotation.normalize_angles(rotated_face)

    return rotated_face, rotation_angle


def align_quat_up(cube_quat, normalize=True):
    """ Align quaternion so that the closest face to being up is actually up """
    z_up = np.array([0, 0, 1]).reshape(3, 1)
    mtx = rotation.quat2mat(cube_quat)
    # Axis that is the closest (by dotproduct) to z-up
    axis_nr = np.abs((z_up.T @ mtx)).argmax()

    # Axis of the cube pointing the closest to the top
    axis = mtx[:, axis_nr]
    axis = axis * np.sign(axis @ z_up)

    # Quaternion representing the rotation from "axis" that is almost up to
    # the actual "up" direction
    difference_quat = rotation.vectors2quat(axis, z_up[:, 0])

    angle = rotation.quat_mul(difference_quat, cube_quat)
    return rotation.quat_normalize(angle) if normalize else angle


def up_axis_with_sign(cube_quat):
    """ Return an axis number + sign of the cube that is the closest to pointing up """
    z_up = np.array([0, 0, 1]).reshape(3, 1)
    mtx = rotation.quat2mat(cube_quat)
    # Axis that is the closest (by dotproduct) to z-up
    axis_nr = np.abs((z_up.T @ mtx)).argmax()
    axis = mtx[:, axis_nr]
    sign = np.sign(axis @ z_up)
    return axis_nr, sign


def distance_quat_from_being_up(cube_quat, axis_nr, sign):
    """ How far is the cube from having given axis pointing upwards """
    mtx = rotation.quat2mat(cube_quat)

    axis = mtx[:, axis_nr]
    axis = axis * sign

    z_up = np.array([0, 0, 1]).reshape(3, 1)

    # Quaternion representing the rotation from "axis" that is almost up to
    # the actual "up" direction
    difference_quat = rotation.vectors2quat(axis, z_up[:, 0])

    return rotation.quat_normalize(difference_quat)
