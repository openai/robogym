import collections
from typing import Dict, Tuple

import numpy as np
import pycuber

from robogym.utils import rotation

PYCUBER_LOCATION_AXES: Dict[str, np.array] = {
    "L": np.array([-1, 0, 0]),
    "R": np.array([1, 0, 0]),
    "F": np.array([0, -1, 0]),
    "B": np.array([0, 1, 0]),
    "D": np.array([0, 0, -1]),
    "U": np.array([0, 0, 1]),
}


PYCUBER_COLOR_AXES: Dict[str, np.array] = {
    "red": np.array([-1, 0, 0]),
    "orange": np.array([1, 0, 0]),
    "blue": np.array([0, 1, 0]),
    "green": np.array([0, -1, 0]),
    "yellow": np.array([0, 0, 1]),
    "white": np.array([0, 0, -1]),
}


# Below mappings are represented as pairs (axis nr, sign)
# e.g. (0, -1) means -X and (2, 1) means +Z
PYCUBER_COLOR_AXES_DESCRIPTIONS: Dict[str, Tuple] = {
    "red": (0, -1),
    "orange": (0, 1),
    "blue": (1, 1),
    "green": (1, -1),
    "yellow": (2, 1),
    "white": (2, -1),
}


PYCUBER_REVERSE_LOCATIONS: Dict[Tuple, str] = {
    (0, -1): "L",
    (0, 1): "R",
    (1, -1): "F",
    (1, 1): "B",
    (2, -1): "D",
    (2, 1): "U",
}


PYCUBER_REVERSE_COLORS: Dict[Tuple, str] = {
    (0, -1): "red",
    (0, 1): "orange",
    (1, 1): "blue",
    (1, -1): "green",
    (2, 1): "yellow",
    (2, -1): "white",
}


class CubeManipulator:
    """
    Class for manipulating the perpendicular rubik's cube model in mujoco simulation.
    Translates face rotation commands into angle representation that can be input into
    mujoco qpos array
    """

    def __init__(self, prefix, sim):
        self.prefix = prefix
        self.sim = sim

        self.drivers = [
            x
            for x in self.sim.model.joint_names
            if x.startswith("{}cubelet:driver:".format(self.prefix))
        ]

        self.rotators = [
            x
            for x in self.sim.model.joint_names
            if x.startswith("{}cubelet:rot".format(self.prefix))
        ]

        self.joints = self.drivers + self.rotators

        self.joints_qpos_map = {}
        self.joints_qpos_idx = []

        for j in self.joints:
            current_index = self.sim.model.get_joint_qpos_addr(j)

            self.joints_qpos_map[j] = current_index
            self.joints_qpos_idx.append(current_index)

        # Info about the cubelets - indexed by cubelet index
        self.cubelet_meta_info = []

        # Populate cubelet meta information
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    indicators = collections.OrderedDict(
                        [("x", i - 1), ("y", j - 1), ("z", k - 1)]
                    )

                    # Build cubelet nam
                    name_pieces = []
                    keys = []

                    for key, value in indicators.items():
                        if value == -1:
                            keys.append(key)
                            name_pieces.append("neg_{}".format(key))
                        elif value == 1:
                            keys.append(key)
                            name_pieces.append("pos_{}".format(key))

                    name = "_".join(name_pieces)

                    data = {"name": name, "coords": np.array([i - 1, j - 1, k - 1])}

                    if len(name_pieces) > 1:
                        # Not a driver just a normal cubelet
                        idxs = [
                            self.joints.index(
                                "{}cubelet:rot{}:{}".format(prefix, key, name)
                            )
                            for key in indicators
                        ]

                        data["type"] = "cubelet"
                        data["euler_qpos"] = [self.joints_qpos_idx[i] for i in idxs]
                    elif len(name_pieces) == 1:
                        data["driver"] = self.drivers.index(
                            "{}cubelet:driver:{}".format(prefix, name)
                        )
                        data["type"] = "driver"
                    else:
                        data["type"] = "null"

                    self.cubelet_meta_info.append(data)

    def _cubelet_rotation_matrix(self, cubelet_meta_info, qpos_array):
        """ Find local coordinate axes for the cubelet """
        euler_angles = qpos_array[cubelet_meta_info["euler_qpos"]]
        return rotation.euler2mat(euler_angles)

    def rotate_face(self, axis, side, angle):
        """
        Rotate given face (identified by axis and side) by given angle.
        Cube should be in a reasonably aligned state for this to work well.
        """
        assert 0 <= axis <= 2
        assert 0 <= side <= 1

        angle = rotation.normalize_angles(np.array(angle))

        if np.abs(angle) < 1e-4:
            # No need to do anything, the angle is too small to care
            return

        side = side * 2 - 1

        qpos_copy = self.sim.data.qpos.copy()

        # For each cubelet
        for i in range(27):
            cubelet_meta = self.cubelet_meta_info[i]

            if cubelet_meta["type"] == "cubelet":
                mtx = self._cubelet_rotation_matrix(cubelet_meta, qpos_copy)
                current_coords = mtx @ cubelet_meta["coords"].astype(float)
                is_selected = np.take(current_coords, axis) * side > 0.5

                if is_selected:
                    euler = np.zeros(3)
                    euler[axis] = angle

                    combined_matrix = rotation.euler2mat(euler) @ mtx
                    new_euler = rotation.mat2euler(combined_matrix)

                    self.sim.data.qpos[cubelet_meta["euler_qpos"]] = new_euler
            elif cubelet_meta["type"] == "driver":
                # No transformation matrix really here
                current_coords = cubelet_meta["coords"]
                is_selected = np.take(current_coords, axis) * side > 0.5

                if is_selected:
                    joint_idx = self.joints_qpos_idx[cubelet_meta["driver"]]
                    self.sim.data.qpos[joint_idx] += angle

    def from_pycuber(self, cube: pycuber.Cube):
        """
        Set cubelet positions based on the pycuber cube state

        Image copied from rubik_utils.py

                        Z(+) Up (Yellow)                     Faces:
                        |                                    +X: Right (Orange)
                        |          / Y(+) Back (Blue)        -X: Left (Red)
                    _____________ /                          +Y: Back (Blue)
                   /            /|                           -Y: Front (Green)
                  /            / |                           +Z: Up (Yellow)
                 /            /  |                           -Z: Down (White)
                /____________/   |
                |            |   |____ X(+) Right (Orange)
       Left     |            |   /
       (Red)    |   Front    |  /
                |  (Green)   | /
                |____________|/
                         Down (White)
        """
        # First, we zero out the cubelet positions to reset all of the cube state
        self.sim.data.qpos[self.joints_qpos_idx] = 0.0

        for cubelet in cube.children:
            if isinstance(cubelet, pycuber.cube.Corner):

                mtx = np.zeros((3, 3))

                for element in cubelet.location:
                    # Example: Corner(B: [r], U: [y], L: [g])
                    # Original location: red, yellow, green: -X, +Z, -Y
                    # Current location back, up, left: +Y, +Z, -X
                    # Mapping: -X -> +Y, +Z -> +Z, -Y -> -X
                    axis, sign = PYCUBER_COLOR_AXES_DESCRIPTIONS[
                        cubelet[element].colour
                    ]
                    vector = PYCUBER_LOCATION_AXES[element]

                    mtx[:, axis] = sign * vector

                euler_angles = rotation.mat2euler(mtx)

                original_location: np.array = sum(
                    PYCUBER_COLOR_AXES[x.colour] for x in cubelet.children
                )
                idx = (
                    (original_location[0] + 1) * 9
                    + (original_location[1] + 1) * 3
                    + original_location[2]
                    + 1
                )

                # Set the euler angles
                self.sim.data.qpos[
                    self.cubelet_meta_info[idx]["euler_qpos"]
                ] = euler_angles
            elif isinstance(cubelet, pycuber.cube.Edge):
                # Example:
                # Edge(R: [o], B: [g])
                # original location: orange-green +X, -Y, (1, -1, 0)
                # current location: right-back, +X, +Y (1, 1, 0)
                original_location = sum(
                    PYCUBER_COLOR_AXES[x.colour] for x in cubelet.children
                )

                mtx = np.zeros((3, 3))

                axes = {0, 1, 2}

                for element in cubelet.location:
                    axis, sign = PYCUBER_COLOR_AXES_DESCRIPTIONS[
                        cubelet[element].colour
                    ]
                    vector = PYCUBER_LOCATION_AXES[element]

                    mtx[:, axis] = sign * vector
                    axes.remove(axis)

                remaining_axis = axes.pop()

                # Antisymmetric tensor
                if remaining_axis == 0:
                    mtx[:, 0] = np.cross(mtx[:, 1], mtx[:, 2])
                elif remaining_axis == 1:
                    mtx[:, 1] = -np.cross(mtx[:, 0], mtx[:, 2])
                elif remaining_axis == 2:
                    mtx[:, 2] = np.cross(mtx[:, 0], mtx[:, 1])

                euler_angles = rotation.mat2euler(mtx)

                idx = (
                    (original_location[0] + 1) * 9
                    + (original_location[1] + 1) * 3
                    + original_location[2]
                    + 1
                )

                # Set the euler angles
                self.sim.data.qpos[
                    self.cubelet_meta_info[idx]["euler_qpos"]
                ] = euler_angles

    def to_pycuber(self) -> pycuber.Cube:
        """ Return current cubelet state as a pycuber state """
        self.soft_align_faces()
        qpos_copy = self.sim.data.qpos.copy()

        cubies = []

        for i in range(27):
            cubelet_meta = self.cubelet_meta_info[i]

            if cubelet_meta["type"] == "cubelet":
                mtx = self._cubelet_rotation_matrix(cubelet_meta, qpos_copy)

                original_coords = cubelet_meta["coords"]
                # current_coords = (mtx @ cubelet_meta['coords'].astype(float)).round().astype(int)

                cubie_desc = {}

                for prev_axis, sign in enumerate(original_coords):
                    if sign != 0:
                        vec = mtx[:, prev_axis] * sign
                        new_axis = np.abs(vec).argmax()
                        new_sign = vec[new_axis]

                        color = PYCUBER_REVERSE_COLORS[prev_axis, sign]
                        loc = PYCUBER_REVERSE_LOCATIONS[new_axis, new_sign]

                        cubie_desc[loc] = pycuber.Square(color)

                if len(cubie_desc) == 3:
                    cubies.append(pycuber.Corner(**cubie_desc))
                elif len(cubie_desc) == 2:
                    cubies.append(pycuber.Edge(**cubie_desc))
            if cubelet_meta["type"] == "driver":
                original_coords = cubelet_meta["coords"]
                axis = np.abs(original_coords).argmax()
                sign = original_coords[axis]

                color = PYCUBER_REVERSE_COLORS[axis, sign]
                loc = PYCUBER_REVERSE_LOCATIONS[axis, sign]

                cubie_desc = {loc: pycuber.Square(color)}
                cubies.append(pycuber.Centre(**cubie_desc))

        return pycuber.Cube(cubies=cubies)

    def snap_rotate_face_with_threshold(self, axis, side, angle, threshold=0.1):
        """
        Rotate face of a cube in a "snapping" fashion, correcting the cube along the way.
        Underlying assumption: cube is already in a "snapped", physically-aligned state

        Threshold is threshold in radians which decides maximum angle we want to snap over.
        If the angle required to move the face to be snapped is larger than that, the cube
        will remain locked and won't rotate.
        """
        qpos = self.sim.data.qpos

        drivers = rotation.normalize_angles(
            qpos[[self.joints_qpos_map[x] for x in self.drivers]]
        )

        perpendicular_axes = sorted({0, 1, 2} - {axis})

        transaction = []
        abort = False

        for other_axis in perpendicular_axes:
            for other_side in range(2):
                other_driver_idx = other_axis * 2 + other_side

                other_angle = drivers[other_driver_idx]
                other_angle_aligned = rotation.round_to_straight_angles(other_angle)
                other_angle_diff = rotation.normalize_angles(
                    other_angle_aligned - other_angle
                )

                if (
                    np.abs(other_angle_diff) < np.abs(angle)
                    and np.abs(other_angle_diff) < threshold
                ):
                    transaction.append((other_axis, other_side, other_angle_diff))
                else:
                    abort = True

        if not abort:
            # Snap other faces
            for other_axis, other_side, angle_diff in transaction:
                self.rotate_face(other_axis, other_side, angle_diff)

            # rotate the actual face
            self.rotate_face(axis, side, angle)

    def soft_align_faces(self):
        """
        Align cube configuration to nearest set of straight angles.
        Should handle more corner cases than naive implementation
        """
        drivers_idx = [self.joints_qpos_map[x] for x in self.drivers]

        current_angles = self.sim.data.qpos[drivers_idx]
        straight_angles = rotation.round_to_straight_angles(current_angles)
        normalized_diff = rotation.normalize_angles(straight_angles - current_angles)

        # From the largest angle to the smallest
        for _, idx in reversed(
            sorted(zip(np.abs(normalized_diff), range(len(normalized_diff))))
        ):
            self.rotate_face(idx // 2, idx % 2, normalized_diff[idx])

        # Align all little cubelets at the end
        for i in range(27):
            info = self.cubelet_meta_info[i]

            if "euler_qpos" in info:
                mtx = self._cubelet_rotation_matrix(info, self.sim.data.qpos)
                # Much better alignment than in the euler angle representation
                # If the cube is close enough to the aligned state it should work
                mtx = mtx.round()
                self.sim.data.qpos[info["euler_qpos"]] = rotation.mat2euler(mtx)

    def align_angles(self):
        """
        Round all cube angles to the nearest straight angle
        Naive implementation that may easily cause the cube to end up in an incorrect state
        due to "gimbal lock" singularity in the euler angle representation
        """
        self.sim.data.qpos[self.joints_qpos_idx] = rotation.round_to_straight_angles(
            self.sim.data.qpos[self.joints_qpos_idx]
        )
