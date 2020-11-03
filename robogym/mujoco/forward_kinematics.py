import xml.etree.ElementTree as et
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import robogym.utils.rotation as rot
from robogym.mujoco.mujoco_xml import MujocoXML


def homogeneous_matrix_from_pos_mat(pos, mat):
    m = np.eye(4)
    m[:3, :3] = mat
    m[:3, 3] = pos
    return m


def get_joint_matrix(pos, angle, axis):
    def transform_rot_x_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the X axis by given angle in radians
        """
        m = np.eye(4)
        m[1, 1] = m[2, 2] = np.cos(angle)
        s = np.sin(angle)
        m[1, 2] = -s
        m[2, 1] = s
        m[:3, 3] = pos
        return m

    def transform_rot_y_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the Y axis by given angle in radians
        """
        m = np.eye(4)
        m[0, 0] = m[2, 2] = np.cos(angle)
        s = np.sin(angle)
        m[0, 2] = s
        m[2, 0] = -s
        m[:3, 3] = pos
        return m

    def transform_rot_z_matrix(pos, angle):
        """
        Optimization - create a homogeneous matrix where rotation submatrix
        rotates around the Z axis by given angle in radians
        """
        m = np.eye(4)
        m[0, 0] = m[1, 1] = np.cos(angle)
        s = np.sin(angle)
        m[0, 1] = -s
        m[1, 0] = s
        m[:3, 3] = pos
        return m

    if abs(axis[0]) == 1.0 and axis[1] == 0.0 and axis[2] == 0.0:
        return transform_rot_x_matrix(pos, angle * axis[0])
    elif axis[0] == 0.0 and abs(axis[1]) == 1.0 and axis[2] == 0.0:
        return transform_rot_y_matrix(pos, angle * axis[1])
    elif axis[0] == 0.0 and axis[1] == 0.0 and abs(axis[2]) == 1.0:
        return transform_rot_z_matrix(pos, angle * axis[2])
    else:
        return homogeneous_matrix_from_pos_mat(
            pos, rot.quat2mat(rot.quat_from_angle_and_axis(angle, axis))
        )


class ForwardKinematics:
    """
        Generic forward kinematics calculator for open chain rigid systems.
    """

    def __init__(self, site_computations, joint_info):
        self.site_computations = site_computations
        self.joint_info = joint_info

    def compute(self, qpos, return_joint_pos=False):
        """
            given joint positions, calculate the endpoint positions.

            currently only return positions, but orientation also computed
            but just not returned.
            later could be useful for other end effectors.

            currently only support hinge joints.
        """
        num_sites = len(self.site_computations)
        site_positions = []
        joint_positions = [None] * len(self.joint_info)

        def cached_joint_calculator(computations, cidx):
            joint_idx = computations[cidx]
            assert isinstance(
                joint_idx, int
            ), "computation should be body interweaving with joints."

            if joint_positions[joint_idx] is not None:
                return joint_positions[joint_idx]
            else:
                (joint_pos, joint_axis) = self.joint_info[joint_idx]
                joint_matrix = get_joint_matrix(joint_pos, qpos[joint_idx], joint_axis)

                if cidx == len(computations) - 2:
                    joint_pos = computations[-1] @ joint_matrix
                else:
                    joint_pos = (
                        cached_joint_calculator(computations, cidx + 2)
                        @ computations[cidx + 1]
                        @ joint_matrix
                    )

                joint_positions[joint_idx] = joint_pos
                return joint_pos

        for i in range(num_sites):
            computations = self.site_computations[i]
            m = computations[0]
            if len(computations) > 1:
                m = cached_joint_calculator(computations, 1) @ m

            # only return position for now
            site_positions.append(m[:3, 3])

        if return_joint_pos:
            joint_xpos = list(map(lambda v: v[:3, 3], joint_positions))
            return np.array(site_positions + joint_xpos)
        else:
            return np.array(site_positions)

    @classmethod
    def prepare(
        cls,
        mxml: MujocoXML,
        root_body_name: str,
        root_body_pos: np.array,
        root_body_euler: np.array,
        target_sites: List[str],
        joint_names: List[str],
    ):
        """
            parse mujoco xml to build up the kinematic tree, also does some
            static computations (e.g. fixed body/body connection).

            target_sites are the endpoints to compute later
            joint_names are sequence of joints passed at runtime for
            the endpoint position calculations.
        """
        IDENTITY_QUAT = rot.quat_identity()
        ROOT_BODY_PARENT = "NONE"

        target_sites_idx: Dict[str, int] = {
            v: idx for idx, v in enumerate(target_sites)
        }
        joint_names_idx: Dict[str, int] = {v: idx for idx, v in enumerate(joint_names)}

        num_sites = len(target_sites)
        site_info: List[Optional[Tuple]] = [None] * num_sites  # (4d matrix, parentBody)
        joint_info: List[Optional[Tuple]] = [None] * len(joint_names)  # (axis, pos)

        body_info: Dict[
            str, Any
        ] = dict()  # name => (4d homegeneous matrix, parentbody)
        body_joints: Dict[str, str] = dict()  # body => joints

        def get_matrix(x: et.Element):
            pos = np.fromstring(x.attrib.get("pos"), sep=" ")
            if "euler" in x.attrib:
                euler = np.fromstring(x.attrib.get("euler"), sep=" ")
                return homogeneous_matrix_from_pos_mat(pos, rot.euler2mat(euler))
            elif "axisangle" in x.attrib:
                axis_angle = np.fromstring(x.attrib.get("axisangle"), sep=" ")
                quat = rot.quat_from_angle_and_axis(
                    axis_angle[-1], np.array(axis_angle[:-1])
                )
                return homogeneous_matrix_from_pos_mat(pos, rot.quat2mat(quat))
            elif "quat" in x.attrib:
                quat = np.fromstring(x.attrib.get("quat"), sep=" ")
                return homogeneous_matrix_from_pos_mat(pos, rot.quat2mat(quat))
            else:
                quat = IDENTITY_QUAT
                return homogeneous_matrix_from_pos_mat(pos, rot.quat2mat(quat))

        def traverse(rt: et.Element, parent_body: str):
            assert rt.tag == "body", "only start from body tag in xml"
            matrix = get_matrix(rt)
            name = rt.attrib.get("name", "noname_body_%d" % len(body_info))
            body_info[name] = (matrix, parent_body)

            for x in rt.findall("joint"):
                joint_name = x.attrib.get("name", "")
                joint_idx: int = joint_names_idx.get(joint_name, -1)
                if joint_idx == -1:
                    continue

                assert (
                    x.attrib.get("type", "hinge") == "hinge"
                ), "currently only support hinge joints"

                pos = np.fromstring(x.attrib.get("pos"), sep=" ")
                axis = np.fromstring(x.attrib.get("axis"), sep=" ")

                joint_info[joint_idx] = (pos, axis)

                assert (
                    joint_name not in body_joints
                ), "Only support open chain system, unsupported rigid bodies"
                body_joints[name] = joint_name

            for x in rt.findall("site"):
                site_idx = target_sites_idx.get(x.attrib.get("name", ""), -1)
                if site_idx != -1:
                    matrix = get_matrix(x)
                    site_info[site_idx] = (matrix, name)

            for x in rt.findall("body"):
                # recursive scan through body parts
                traverse(x, name)

        rt = None
        for child in mxml.root_element.find("worldbody").findall("body"):  # type: ignore
            if child.attrib.get("name", "") == root_body_name:
                rt = child
                break

        assert rt is not None, "no root body found in xml"
        traverse(rt, ROOT_BODY_PARENT)

        root_matrix = homogeneous_matrix_from_pos_mat(
            root_body_pos, rot.euler2mat(root_body_euler)
        )

        # build the computation flow
        site_computations = [[] for i in range(num_sites)]  # type: ignore

        for i in range(num_sites):
            (matrix, parent_body) = site_info[i]  # type: ignore # Just have to trust the code
            while parent_body != ROOT_BODY_PARENT:
                parent_matrix, new_parent_body = body_info[parent_body]
                joint_name = body_joints.get(parent_body, "")

                if joint_name:
                    site_computations[i].append(matrix)
                    site_computations[i].append(joint_names_idx[joint_name])
                    matrix = parent_matrix
                else:
                    matrix = parent_matrix @ matrix
                parent_body = new_parent_body

            site_computations[i].append(root_matrix @ matrix)

        return cls(site_computations, joint_info)
