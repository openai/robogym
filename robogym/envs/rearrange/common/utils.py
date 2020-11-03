import collections
import glob
import itertools
import json
import logging
import os
from copy import deepcopy
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import _jsonnet
import numpy as np
import trimesh
from collision import Poly, Vector, collide
from mujoco_py import MjSim, const
from numpy.random import RandomState

from robogym.mujoco.mujoco_xml import ASSETS_DIR, MujocoXML
from robogym.utils.env_utils import InvalidSimulationError
from robogym.utils.mesh import get_vertices_bounding_box, subdivide_mesh
from robogym.utils.misc import robogym_path
from robogym.utils.rotation import mat2quat, quat2mat, quat_conjugate, uniform_quat

MATERIAL_DIR = robogym_path("envs", "rearrange", "materials")

NumType = Union[int, float]


class PlacementArea(NamedTuple):
    """The offset of the placement area, which is in the lower left corner"""

    offset: Tuple[float, float, float]

    """The full-size of the placement area (note that we do NOT use half-size convention here)"""
    size: Tuple[float, float, float]


def recursive_dict_update(dictionary, update):
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            dictionary[k] = recursive_dict_update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary


def sample_group_counts(
    random_state: RandomState, total: int, lam_low: float = 1.0, lam_high: float = 8.0
) -> List[int]:
    """
    Sample a list of integers which sum up to `total`.
    The probability of sampling an integer follows exponential decay, k ~ np.exp(-k * lam),
    where lam is a hyperparam sampled from a range [lam_low, lam_high).

    :param random_state: numpy random state
    :param total: the expected sum of sampled numbers.
    :param lam_low: lower bound for lambda in exponential decay.
    :param lam_high: higher bound for lambda in exponential decay.
    :return:
    """
    current_max = total
    counts = []
    while current_max > 0:
        candidates = range(1, current_max + 1)
        lam = random_state.uniform(lam_low, lam_high)
        probs = np.array([np.exp(-i * lam) for i in candidates])
        probs /= sum(probs)
        selected = random_state.choice(candidates, p=probs)
        counts.append(selected)
        current_max -= selected

    assert sum(counts) == total
    return counts


def stabilize_objects(mujoco_simulation, n_steps: int = 100):
    """
    Stabilize objects.
    """
    # Store original damping value for objects.
    damping = mujoco_simulation.get_object_damping()

    # Decrease damping value to make object stabilize faster.
    mujoco_simulation.set_object_damping(1e-3)

    # Step simulation to let object stabilize.
    for _ in range(n_steps):
        mujoco_simulation.step()

    # Restore damping value.
    mujoco_simulation.set_object_damping(damping)
    mujoco_simulation.forward()


def make_openai_block(name: str, object_size: np.ndarray) -> MujocoXML:
    """ Creates a block with OPENAI letters on it faces.

    :param name: The name of the block
    :param object_size: The size of the block (3-dimensional). This is half-size as per Mujoco
        convention
    """
    default_object_size = 0.0285
    default_letter_offset = 0.0009

    # scale face meshes properly
    scale = object_size / default_object_size
    letter_offset = default_letter_offset * scale

    def to_str(x: np.ndarray):
        return " ".join(map(str, x.tolist()))

    face_pos = {
        "top": {
            "body": to_str(np.array([0, 0, object_size[2]])),
            "geom": to_str(np.array([0, 0, -letter_offset[2]])),
        },
        "bottom": {
            "body": to_str(np.array([0, 0, -object_size[2]])),
            "geom": to_str(np.array([0, 0, letter_offset[2]])),
        },
        "back": {
            "body": to_str(np.array([0, object_size[1], 0])),
            "geom": to_str(np.array([0, -letter_offset[1], 0])),
        },
        "right": {
            "body": to_str(np.array([object_size[0], 0, 0])),
            "geom": to_str(np.array([-letter_offset[0], 0, 0])),
        },
        "front": {
            "body": to_str(np.array([0, -object_size[1], 0])),
            "geom": to_str(np.array([0, letter_offset[1], 0])),
        },
        "left": {
            "body": to_str(np.array([-object_size[0], 0, 0])),
            "geom": to_str(np.array([letter_offset[0], 0, 0])),
        },
    }
    face_euler = {
        "top": to_str(np.array([np.pi / 2, 0, np.pi / 2])),
        "bottom": to_str(np.array([np.pi / 2, 0, np.pi / 2])),
        "back": to_str(np.array([0, 0, np.pi / 2])),
        "right": to_str(np.array([0, 0, 0])),
        "front": to_str(np.array([0, 0, -np.pi / 2])),
        "left": to_str(np.array([0, 0, np.pi])),
    }

    def face_xml(_name: str, _face: str, _c: str):
        xml = f"""
        <body name="{_face}:{_name}" pos="{face_pos[_face]['body']}">
            <geom name="letter_{_c}:{_name}" mesh="{_name}{_c}" euler="{face_euler[_face]}"
             pos="{face_pos[_face]['geom']}" type="mesh" material="{_name}letter"
             conaffinity="0" contype="0" />
        </body>
        """
        return xml

    size_string = " ".join(map(str, list(object_size)))
    scale_string = " ".join(map(str, list(scale)))

    xml_source = f"""
    <mujoco>
        <asset>
            <material name="{name}letter" specular="1" shininess="0.3" rgba="1 1 1 1"/>
            <mesh name="{name}O" file="{ASSETS_DIR}/stls/openai_cube/O.stl"
             scale="{scale_string}" />
            <mesh name="{name}P" file="{ASSETS_DIR}/stls/openai_cube/P.stl"
             scale="{scale_string}" />
            <mesh name="{name}E" file="{ASSETS_DIR}/stls/openai_cube/E.stl"
             scale="{scale_string}" />
            <mesh name="{name}N" file="{ASSETS_DIR}/stls/openai_cube/N.stl"
             scale="{scale_string}" />
            <mesh name="{name}A" file="{ASSETS_DIR}/stls/openai_cube/A.stl"
             scale="{scale_string}" />
            <mesh name="{name}I" file="{ASSETS_DIR}/stls/openai_cube/I.stl"
             scale="{scale_string}" />
        </asset>
        <worldbody>
            <body name="{name}">
                <geom name="{name}" size="{size_string}" type="box"
                 rgba="0.0 0.0 0.0 0.0" material="block_mat"/>
                <joint name="{name}:joint" type="free"/>
                {face_xml(name, "top", "O")}
                {face_xml(name, "bottom", "P")}
                {face_xml(name, "back", "E")}
                {face_xml(name, "right", "N")}
                {face_xml(name, "front", "A")}
                {face_xml(name, "left", "I")}
            </body>
        </worldbody>
    </mujoco>
    """
    return MujocoXML.from_string(xml_source)


def make_block(name: str, object_size: np.ndarray) -> MujocoXML:
    """Creates a block.

    :param name: The name of the block
    :param object_size: The size of the block (3-dimensional). This is half-size as per Mujoco
        convention
    """
    xml_source = f"""
    <mujoco>
      <worldbody>
        <body name="{name}" pos="0.0 0.0 0.0">
          <geom type="box" rgba="0.0 0.0 0.0 0.0" material="block_mat"/>
          <joint name="{name}:joint" type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """
    xml = MujocoXML.from_string(xml_source).set_objects_attr(
        tag="geom", size=object_size
    )

    return xml


def make_blocks_and_targets(
    num_objects: int, block_size: Union[float, np.ndarray], appearance: str = "standard"
) -> List[Tuple[MujocoXML, MujocoXML]]:
    if isinstance(
        block_size, (int, float, np.integer, np.floating)
    ) or block_size.shape == (1,):
        block_size = np.tile(block_size, 3)
    assert block_size.shape == (
        3,
    ), f"Bad block_size: {block_size}, expected float, np.ndarray(1,) or np.ndarray(3,)"

    if appearance == "standard":
        make_block_fn = make_block
    elif appearance == "openai":
        make_block_fn = make_openai_block

    xmls: List[Tuple[MujocoXML, MujocoXML]] = []
    for i in range(num_objects):
        # add the block
        block_xml = make_block_fn(f"object{i}", block_size.copy())
        xmls.append((block_xml, make_target(block_xml)))

    return xmls


def get_combined_mesh(files: List[str]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(
        [trimesh.load(os.path.join(ASSETS_DIR, "stls", file)) for file in files]
    )


def make_mesh_object(name: str, files: List[str], scale: float) -> MujocoXML:
    # Center mesh properly by offsetting with center position of combined mesh.
    mesh = get_combined_mesh(files)

    pos = -mesh.center_mass * scale
    pos_string = " ".join(map(str, pos))

    scale_string = " ".join(map(str, [scale] * 3))
    assets = [
        f'<mesh file="{file}" name="{name}-{idx}" scale="{scale_string}" />'
        for idx, file in enumerate(files)
    ]
    geoms = [
        f'<geom type="mesh" mesh="{name}-{idx}" pos="{pos_string}"/>'
        for idx in range(len(files))
    ]
    assets_xml = "\n".join(assets)
    geoms_xml = "\n".join(geoms)
    xml_source = f"""
    <mujoco>
      <asset>
        {assets_xml}
      </asset>
      <worldbody>
        <body name="{name}" pos="0.0 0.0 0.0">
          {geoms_xml}
          <joint name="{name}:joint" type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return MujocoXML.from_string(xml_source)


def make_target(xml):
    xml = deepcopy(xml)
    xml = (
        xml.remove_objects_by_tag("joint")
        .add_name_prefix("target:", exclude_attribs=["material", "mesh", "class"])
        .set_objects_attr(tag="geom", contype=0, conaffinity=0)
    )
    return xml


def get_all_vertices(sim, object_name, subdivide_threshold=None) -> np.ndarray:
    """
    Return all vertices for given object.
    :param sim: The MjSim instance.
    :param object_name: The object name.
    :param subdivide_threshold: If provided, subdivide mesh into smaller faces.
      See subdivide_mesh for detail of this parameter.
    :return: Array of all vertices for this object.
    """
    all_verts: List[np.ndarray] = []
    all_faces: List[Optional[np.ndarray]] = []

    object_rot_mat = quat2mat(
        quat_conjugate(mat2quat(sim.data.get_body_xmat(object_name)))
    )
    geom_ids = geom_ids_of_body(sim, object_name)

    for geom_id in geom_ids:
        pos = sim.model.geom_pos[geom_id]
        quat = quat_conjugate(sim.model.geom_quat[geom_id])
        mat = quat2mat(quat)

        # Get all vertices associated with the current geom.
        verts = get_geom_vertices(sim, geom_id)
        faces = get_geom_faces(sim, geom_id)

        # Translate from geom's to body's coordinate frame.
        geom_ref_verts = verts @ mat
        geom_ref_verts = pos + geom_ref_verts

        all_verts.append(geom_ref_verts)
        all_faces.append(faces)

    if subdivide_threshold is not None and all(f is not None for f in all_faces):
        # We can only subdivide mesh with faces.
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=verts, faces=faces)
                for verts, faces in zip(all_verts, all_faces)
            ]
        )

        verts = subdivide_mesh(mesh.vertices, mesh.faces, subdivide_threshold)
    else:
        verts = np.concatenate(all_verts, axis=0)

    return verts @ object_rot_mat


def get_geom_vertices(sim, geom_id):
    geom_type = sim.model.geom_type[geom_id]
    geom_size = sim.model.geom_size[geom_id]

    if geom_type == const.GEOM_BOX:
        dx, dy, dz = geom_size
        return np.array(list(itertools.product([dx, -dx], [dy, -dy], [dz, -dz])))
    elif geom_type in (const.GEOM_SPHERE, const.GEOM_ELLIPSOID):
        if geom_type == const.GEOM_SPHERE:
            r = [geom_size[0]] * 3
        else:
            r = geom_size[:3]

        # https://stats.stackexchange.com/a/30622
        vertices = []
        phi = np.linspace(0, np.pi * 2, 20)
        cos_theta = np.linspace(-1, 1, 20)
        for p, c in itertools.product(phi, cos_theta):
            x = np.sqrt(1 - c ** 2) * np.cos(p)
            y = np.sqrt(1 - c ** 2) * np.sin(p)
            z = c
            vertices.append(np.array([x, y, z]))

        return np.array(vertices) * r
    elif geom_type in (const.GEOM_CYLINDER, const.GEOM_CAPSULE):
        # We treat cylinder and capsule the same.
        r, h = geom_size[0], geom_size[2]
        points = np.array(
            [[r * np.cos(x), r * np.sin(x), 0.0] for x in np.linspace(0, np.pi * 2, 50)]
        )

        return np.concatenate([points + h, points - h])
    elif geom_type == const.GEOM_MESH:
        return sim.model.mesh_vert[mesh_vert_range_of_geom(sim, geom_id)]
    else:
        raise AssertionError(f"Unexpected geom type {geom_type}")


def get_geom_faces(sim, geom_id):
    if sim.model.geom_type[geom_id] != const.GEOM_MESH:
        return None

    data_id = sim.model.geom_dataid[geom_id]
    face_adr = sim.model.mesh_faceadr[data_id]
    face_num = sim.model.mesh_facenum[data_id]
    return sim.model.mesh_face[range(face_adr, face_adr + face_num)]


def get_mesh_bounding_box(sim, object_name) -> Tuple[float, float]:
    """ Returns the bounding box of a mesh body. If the block is rotated in the world frame,
    the rotation is applied and the tightest axis-aligned bounding box is returned.
    """
    all_verts = get_all_vertices(sim, object_name)
    pos, size, _ = get_vertices_bounding_box(all_verts)
    return pos, size


def get_block_bounding_box(sim, object_name) -> Tuple[float, float]:
    """ Returns the bounding box of a block body. If the block is rotated in the world frame,
    the rotation is applied and the tightest axis-aligned bounding box is returned.
    """
    geom_ids = geom_ids_of_body(sim, object_name)
    assert len(geom_ids) == 1, f"More than 1 geoms in {object_name}."
    geom_id = geom_ids[0]
    size = sim.model.geom_size[geom_id]
    pos = sim.model.geom_pos[geom_id]

    quat = quat_conjugate(mat2quat(sim.data.get_body_xmat(object_name)))
    pos, size = rotate_bounding_box((pos, size), quat)
    return pos, size


class MeshGeom:
    """A little helper class for generating random mesh geoms."""

    def __init__(
        self,
        mesh_path: str,
        mesh: trimesh.Trimesh,
        pos: np.array,
        quat: np.array,
        scale: np.array,
        parent: Optional["MeshGeom"] = None,
    ):
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.pos = pos
        self.quat = quat
        self.scale = scale
        self.parent = parent
        self.children: List[MeshGeom] = []

    def to_geom_xml(self, name: str, idx: int):
        pos_str = " ".join([str(x) for x in self.pos])
        quat_str = " ".join([str(x) for x in self.quat])
        return f'<geom type="mesh" mesh="{name}:mesh-{idx}" name="{name}:geom-{idx}" pos="{pos_str}" quat="{quat_str}" />'

    def to_mesh_xml(self, name: str, idx: int):
        scale_str = " ".join([str(x) for x in self.scale])
        return f'<mesh file="{self.mesh_path}" name="{name}:mesh-{idx}" scale="{scale_str}" />'

    def min_max_xyz(self):
        # We already applied the scaling and rotation to the mesh vertices, so we only need
        # to apply the offset here.
        transformed_vertices = self.pos + self.mesh.vertices

        min_xyz = np.min(transformed_vertices, axis=0)
        max_xyz = np.max(transformed_vertices, axis=0)
        return min_xyz, max_xyz


def make_composed_mesh_object(
    name: str,
    primitives: List[str],
    random_state: np.random.RandomState,
    mesh_size_range: tuple = (0.01, 0.1),
    attachment: str = "random",
    object_size: Optional[float] = None,
) -> MujocoXML:
    """
    Composes an object out of mesh primitives by combining them in a random but systematic
    way. In the resulting object, all meshes are guaranteed to be connected.

    :param name: The name of the resulting object.
    :param primitives: A list of STL files that will be used as primitives in the provided order.
    :param random_state: The random state used for sampling.
    :param mesh_size_range: Each mesh is randomly resized (iid per dimension) but each side is
        guaranteed to be within this size. This is full-size, not half-size.
    :param attachment: How primitives are connected. If "random", the parent geom is randomly
        selected from the already placed geoms. If "last", the geom that was place last is used.
    :param object_size: If this is not None, the final object) will be re-scaled so that the longest
        side has exactly object_size half-size. This parameter is in half-size, as per Mujoco
        convention.
    :return: a MujocoXML object.
    """
    assert 0 <= mesh_size_range[0] <= mesh_size_range[1]
    assert attachment in ["random", "last"]

    def compute_pos_and_size(geoms):
        min_max_xyzs = np.array([geom.min_max_xyz() for geom in geoms])
        min_xyz = np.min(min_max_xyzs[:, 0, :], axis=0)
        max_xyz = np.max(min_max_xyzs[:, 1, :], axis=0)
        size = (max_xyz - min_xyz) / 2.0
        pos = min_xyz + size
        return pos, size

    geoms: List[MeshGeom] = []
    for i, mesh_path in enumerate(primitives):
        # Load mesh.
        mesh = trimesh.load(mesh_path)

        # Scale randomly but such that the mesh is within mesh_size_range.
        min_scale = mesh_size_range[0] / mesh.bounding_box.extents
        max_scale = mesh_size_range[1] / mesh.bounding_box.extents
        assert min_scale.shape == max_scale.shape == (3,)
        scale = random_state.uniform(min_scale, max_scale) * random_state.choice(
            [-1, 1], size=3
        )
        assert scale.shape == (3,)
        scale_matrix = np.eye(4)
        np.fill_diagonal(scale_matrix[:3, :3], scale)

        # Rotate randomly.
        quat = uniform_quat(random_state)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = quat2mat(quat)

        # Apply transformations. Apply scaling first since we computed the scale
        # in the original reference frame! In principle, we could also sheer the
        # object, but we currently do not do this.
        mesh.apply_transform(scale_matrix)
        mesh.apply_transform(rotation_matrix)

        if len(geoms) == 0:
            pos = -mesh.center_mass
        else:
            if attachment == "random":
                parent_geom = random_state.choice(geoms)
            elif attachment == "last":
                parent_geom = geoms[-1]
            else:
                raise ValueError()
            # We sample 10 points here because sample_surface sometimes returns less points
            # than we requested (unclear why).
            surface_pos = trimesh.sample.sample_surface(parent_geom.mesh, 10)[0][0]
            pos = parent_geom.pos + (surface_pos - mesh.center_mass)

        geom = MeshGeom(mesh_path=mesh_path, mesh=mesh, pos=pos, quat=quat, scale=scale)
        geoms.append(geom)

    # Shift everything so that the reference of the body is at the very center of the composed
    # object. This is very important.
    off_center_pos, _ = compute_pos_and_size(geoms)
    for geom in geoms:
        geom.pos -= off_center_pos

    # Ensure that the object origin is exactly at the center.
    assert np.allclose(compute_pos_and_size(geoms)[0], 0.0)

    # Resize object.
    if object_size is not None:
        _, size = compute_pos_and_size(geoms)

        # Apply global scale (so that ratio is not changed and longest side is exactly
        # object_size).
        ratio = object_size / np.max(size)
        for geom in geoms:
            geom.scale *= ratio
            geom.pos *= ratio

    geoms_str = "\n".join([g.to_geom_xml(name, idx) for idx, g in enumerate(geoms)])
    meshes_str = "\n".join([g.to_mesh_xml(name, idx) for idx, g in enumerate(geoms)])

    xml_source = f"""
    <mujoco>
        <asset>
          {meshes_str}
        </asset>
        <worldbody>
        <body name="{name}" pos="0 0 0">
          {geoms_str}
          <joint name="{name}:joint" type="free"/>
        </body>
        </worldbody>
    </mujoco>
    """
    return MujocoXML.from_string(xml_source)


def geom_ids_of_body(sim: MjSim, body_name: str) -> List[int]:
    object_id = sim.model.body_name2id(body_name)
    object_geomadr = sim.model.body_geomadr[object_id]
    object_geomnum = sim.model.body_geomnum[object_id]
    return list(range(object_geomadr, object_geomadr + object_geomnum))


def mesh_vert_range_of_geom(sim: MjSim, geom_id: int):
    assert sim.model.geom_type[geom_id] == const.GEOM_MESH
    data_id = sim.model.geom_dataid[geom_id]
    vert_adr = sim.model.mesh_vertadr[data_id]
    vert_num = sim.model.mesh_vertnum[data_id]
    return range(vert_adr, vert_adr + vert_num)


def mesh_face_range_of_geom(sim: MjSim, geom_id: int):
    assert sim.model.geom_type[geom_id] == const.GEOM_MESH
    data_id = sim.model.geom_dataid[geom_id]
    face_adr = sim.model.mesh_faceadr[data_id]
    face_num = sim.model.mesh_facenum[data_id]
    return range(face_adr, face_adr + face_num)


def update_object_body_quat(sim: MjSim, body_name: str, new_quat: np.ndarray):
    body_id = sim.model.body_name2id(body_name)
    sim.model.body_quat[body_id][:] = new_quat.copy()


def _is_valid_proposal(o1_x, o1_y, object1_index, bounding_boxes, placements):
    o1_x += bounding_boxes[object1_index, 0, 0]
    o1_y += bounding_boxes[object1_index, 0, 1]

    # Check if object collides with any of already placed objects. We use half-sizes,
    # but collision uses full-sizes. That's why we multiply by 2x here.
    o1_w, o1_h, _ = bounding_boxes[object1_index, 1]
    object1 = Poly.from_box(Vector(o1_x, o1_y), o1_w * 2.0, o1_h * 2.0)
    for object2_index in range(len(placements)):
        # Don't care about z placement
        o2_x, o2_y, _ = placements[object2_index]
        o2_x += bounding_boxes[object2_index, 0, 0]
        o2_y += bounding_boxes[object2_index, 0, 1]
        # Don't care about object depth.
        o2_w, o2_h, _ = bounding_boxes[object2_index, 1]
        object2 = Poly.from_box(Vector(o2_x, o2_y), o2_w * 2.0, o2_h * 2.0)

        if collide(object1, object2):
            return False

    return True


def _place_objects(
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
    placement_area: PlacementArea,
    get_proposal: Callable[[int], Tuple[NumType, NumType]],
    max_placement_trial_count: int,
    max_placement_trial_count_per_object: int,
    run_collision_check: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Wrapper for _place_objects_trial() function. Call _place_object_trial() multiple times until it
    returns a valid placements. The _place_objects_trial() function can be called for
    `max_placement_trial_count` times.
    """
    assert max_placement_trial_count >= 1
    assert max_placement_trial_count_per_object >= 1
    for _ in range(max_placement_trial_count):
        placements, is_valid = _place_objects_trial(
            object_bounding_boxes,
            table_dimensions,
            placement_area,
            get_proposal,
            max_placement_trial_count_per_object,
            run_collision_check,
        )
        if is_valid:
            return placements, is_valid
    return placements, False


def _place_objects_trial(
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
    placement_area: PlacementArea,
    get_proposal: Callable[[int], Tuple[NumType, NumType]],
    max_placement_trial_count_per_object: int,
    run_collision_check: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Place objects within rectangular boundaries with given get proposal function.

    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :param placement_area: the placement area in which to place objects.
    :param get_proposal: Function to get a proposal of target position for given object. This
        function takes in object index and return proposed (x, y) position.
    :param max_placement_trial_count_per_object: If set, will give up re-generating new proposal
        after this number is hit.
    :param run_collision_check: If true, run collision to check if proposal is valid.
    :return: np.ndarray of size (num_objects, 3) where columns are x, y, z coordinates of objects
        relative to the world frame and boolean indicating whether the placement is valid.
    """

    offset_x, offset_y, _ = placement_area.offset
    width, height, _ = placement_area.size
    table_pos, table_size, table_height = table_dimensions

    def _get_global_placement(placement: np.ndarray):
        return placement + [offset_x, offset_y, 0.0] - table_size + table_pos

    # place the objects one by one, resampling if a collision with previous objects happens
    n_objects = object_bounding_boxes.shape[0]
    placements: List[Tuple[NumType, NumType, NumType]] = []

    for i in range(n_objects):
        placement_trial_count = 0

        # Reference is to (xmin, ymin, zmin) of table.
        prop_z = object_bounding_boxes[i, 1, -1] + 2 * table_size[-1]
        prop_z -= object_bounding_boxes[i, 0, -1]
        while True:
            prop_x, prop_y = get_proposal(i)
            placement = _get_global_placement(np.array([prop_x, prop_y, prop_z]))
            b1_x, b1_y = placement[:2]
            if not run_collision_check or _is_valid_proposal(
                b1_x, b1_y, i, object_bounding_boxes, placements
            ):
                break

            placement_trial_count += 1

            if placement_trial_count > max_placement_trial_count_per_object:
                return np.zeros((n_objects, len(placement))), False

        placements.append(placement)

    return np.array(placements), True


def place_objects_in_grid(
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
    placement_area: PlacementArea,
    random_state: np.random.RandomState,
    max_num_trials: int = 5,
) -> Tuple[np.ndarray, bool]:
    """
    Place objects within rectangular boundaries by dividing the placement area into a grid of cells
    of equal size, and then randomly sampling cells for each object to be placed in.

    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :param placement_area: the placement area in which to place objects.
    :param random_state: numpy random state to use to shuffle placement positions
    :param max_num_trials: maximum number of trials to run (a trial will fail if there is overlap
        detected between any two placements; generally this shouldn't happen with this algorithm)
    :return: Tuple[np.ndarray, bool], where the array is of size (num_objects, 3) with columns set
        to the x, y, z coordinates of objects relative to the world frame, and the boolean
        indicates whether the placement is valid.
    """
    offset_x, offset_y, _ = placement_area.offset
    width, height, _ = placement_area.size
    table_pos, table_size, table_height = table_dimensions

    def _get_global_placement(placement: np.ndarray):
        return placement + [offset_x, offset_y, 0.0] - table_size + table_pos

    # 1. Determine the number of rows and columns of the grid, based on the largest object width
    # and height.
    total_object_area = 0.0
    n_objects = object_bounding_boxes.shape[0]
    max_obj_height = 0.0
    max_obj_width = 0.0
    for i in range(n_objects):
        # Bounding boxes are in half-sizes.
        obj_width = object_bounding_boxes[i, 1, 0] * 2
        obj_height = object_bounding_boxes[i, 1, 1] * 2

        max_obj_height = max(max_obj_height, obj_height)
        max_obj_width = max(max_obj_width, obj_width)

        object_area = obj_width * obj_height
        total_object_area += object_area

    n_columns = int(width // max_obj_width)
    n_rows = int(height // max_obj_height)
    n_cells = n_columns * n_rows

    cell_width = width / n_columns
    cell_height = height / n_rows

    if n_cells < n_objects:
        # Cannot find a valid placement via this method; give up.
        logging.warning(
            f"Unable to fit {n_objects} objects into placement area with {n_cells} cells"
        )
        return np.zeros(shape=(n_objects, 3)), False

    for trial_i in range(max_num_trials):
        placement_valid = True
        placements: List[Tuple[NumType, NumType, NumType]] = []

        # 2. Initialize an array with all valid cell coordinates.

        # Create an array of shape (n_rows, n_columns, 2) where each element contains the row,col
        # coord
        coords = np.dstack(np.mgrid[0:n_rows, 0:n_columns])
        # Create a shuffled list where ever entry is a valid (row, column) coordinate.
        coords = np.reshape(coords, (n_rows * n_columns, 2))
        random_state.shuffle(coords)
        coords = list(coords)

        # 3. Place each object into a randomly selected cell.
        for object_idx in range(n_objects):
            row, col = coords.pop()
            pos, size = object_bounding_boxes[object_idx]

            prop_x = cell_width * col + size[0] - pos[0]
            prop_y = cell_height * row + size[1] - pos[1]

            # Reference is to (xmin, ymin, zmin) of table.
            prop_z = object_bounding_boxes[object_idx, 1, -1] + 2 * table_size[-1]
            prop_z -= object_bounding_boxes[object_idx, 0, -1]

            placement = _get_global_placement(np.array([prop_x, prop_y, prop_z]))

            b1_x, b1_y = placement[:2]
            if not _is_valid_proposal(
                b1_x, b1_y, object_idx, object_bounding_boxes, placements
            ):
                placement_valid = False
                logging.warning(f"Trial {trial_i} failed on object {object_idx}")
                break

            placements.append(placement)

        if placement_valid:
            assert (
                len(placements) == n_objects
            ), "There should be a placement for every object"
            break

    return np.array(placements), placement_valid


def place_objects_with_no_constraint(
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
    placement_area: PlacementArea,
    max_placement_trial_count: int,
    max_placement_trial_count_per_object: int,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, bool]:
    """
    Place objects within rectangular boundaries without any extra constraint.

    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :param placement_area: the placement area in which to place objects
    :param max_placement_trial_count: To prevent infinite loop caused by target placements,
        max_placement_trial_count should set to a finite positive number.
    :param max_placement_trial_count_per_object: To prevent infinite loop caused by target
        placements, max_placement_trial_count_per_object should set to a finite positive number.
    :param random_state: numpy RandomState to use for sampling
    :return: np.ndarray of size (num_objects, 3) where columns are x, y, z coordinates of objects
        relative to the world frame and boolean indicating whether if the proposal is valid.
    """

    def _get_placement_proposal(object_idx):
        # randomly place the object within the bounds
        pos, size = object_bounding_boxes[object_idx]
        offset_x, offset_y, _ = placement_area.offset
        width, height, _ = placement_area.size
        x, y = random_state.uniform(
            low=(size[0], size[1]), high=(width - size[0], height - size[1])
        )
        x -= pos[0]
        y -= pos[1]
        return x, y

    return _place_objects(
        object_bounding_boxes,
        table_dimensions,
        placement_area,
        _get_placement_proposal,
        max_placement_trial_count,
        max_placement_trial_count_per_object,
    )


def place_targets_with_fixed_position(
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
    placement_area: PlacementArea,
    target_placements: np.ndarray,
):
    """
    Place target object according to specified placement positions.
    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :param placement_area: the placement area in which to place objects
    :param target_placements: Placement positions (x, y) relative to the placement area. Normalized
        to [0, 1]
    :return: Global placement positions (x, y, z) for all objects.
    """

    def _get_placement_proposal(object_idx):
        width, height, _ = placement_area.size
        return target_placements[object_idx] * [width, height]

    return _place_objects(
        object_bounding_boxes,
        table_dimensions,
        placement_area,
        _get_placement_proposal,
        max_placement_trial_count=1,
        max_placement_trial_count_per_object=1,
        run_collision_check=False,
    )


def place_targets_with_goal_distance_ratio(
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
    placement_area: PlacementArea,
    object_placements: np.ndarray,
    goal_distance_ratio: float,
    goal_distance_min: float,
    max_placement_trial_count: int,
    max_placement_trial_count_per_object: int,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, bool]:
    """
    Place targets around objects with goal distance.
    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :param placement_area: the placement area in which to place objects
    :param object_placements: placements of boxes - this is the result of place_objects
    :param goal_distance_ratio: goal is uniformly sampled first and then distance beween the
        object and the goal is shrinked. The shrinked distance is original distance times
        goal_distance_ratio.
    :param goal_distance_min: minimum goal distance to ensure that goal is not too close to the
        object position.
    :param max_placement_trial_count: To prevent infinite loop caused by target placements,
        max_placement_trial_count should set to a finite positive number.
    :param max_placement_trial_count_per_object: To prevent infinite loop caused by target
        placements, max_placement_trial_count_per_object should set to a finite positive number.
    :param random_state: numpy RandomState to use for sampling
    :return: np.ndarray of size (num_objects, 3) where columns are x, y coordinates of objects
        and boolean indicating whether if the proposal is valid.
    """

    def _get_placement_proposal(object_idx):
        # Sample goal position relative to table area
        pos, size = object_bounding_boxes[object_idx]
        offset_x, offset_y, _ = placement_area.offset
        width, height, _ = placement_area.size
        gx, gy = random_state.uniform(
            low=(size[0], size[1]), high=(width - size[0], height - size[1])
        )
        # Retrieve object position relative to table area
        table_pos, table_size, table_height = table_dimensions
        object_place = (
            object_placements[object_idx]
            - [offset_x, offset_y, 0.0]
            + table_size
            - table_pos
        )
        x = object_place[0] + pos[0]
        y = object_place[1] + pos[1]
        # Pull goal position close to the object position
        dist = np.linalg.norm([gx - x, gy - y])
        min_ratio = goal_distance_min / dist if dist >= goal_distance_min else 0.0
        ratio = np.clip(goal_distance_ratio, min_ratio, 1.0)
        gx = x + (gx - x) * ratio
        gy = y + (gy - y) * ratio

        return gx - pos[0], gy - pos[1]

    return _place_objects(
        object_bounding_boxes,
        table_dimensions,
        placement_area,
        _get_placement_proposal,
        max_placement_trial_count,
        max_placement_trial_count_per_object,
    )


def find_meshes_by_dirname(root_mesh_dir) -> Dict[str, list]:
    """
    Find all meshes under given mesh directory, grouped by top level
    folder name.
    :param root_mesh_dir: The root directory name for mesh files.
    :return: {dir_name -> list of mesh files}
    """
    root_path = os.path.join(ASSETS_DIR, "stls", root_mesh_dir)

    all_stls = {}
    for subdir in os.listdir(root_path):
        curr_path = os.path.join(root_path, subdir)
        if not os.path.isdir(curr_path) and not curr_path.endswith(".stl"):
            continue

        if curr_path.endswith(".stl"):
            stls = [curr_path]
        else:
            stls = glob.glob(os.path.join(curr_path, "*.stl"))
        assert len(stls) > 0

        all_stls[subdir] = stls

    assert len(all_stls) > 0
    return all_stls


def find_stls(mesh_dir) -> List[str]:
    return glob.glob(os.path.join(mesh_dir, "**", "*.stl"), recursive=True)


def load_all_materials() -> List[str]:
    """
    Return name for all material files under envs/rearrange/materials
    """
    return [
        os.path.splitext(os.path.basename(material_path))[0]
        for material_path in glob.glob(os.path.join(MATERIAL_DIR, "*.jsonnet"))
    ]


# NOTE: Use lru_cache so that we don't have to re-compile materials files over and over
@lru_cache()
def load_material_args(material_name: str) -> dict:
    """
    Load mujoco args related to given material.
    """
    material_path = os.path.join(MATERIAL_DIR, f"{material_name}.jsonnet")
    return json.loads(_jsonnet.evaluate_file(material_path))


def safe_reset_env(
    env, max_reset_retry_on_invalid_sim_error=100, only_reset_goal=False
) -> dict:
    for i in range(max_reset_retry_on_invalid_sim_error):
        try:
            if only_reset_goal:
                obs = env.reset_goal()
            else:
                obs = env.reset()
        except InvalidSimulationError:
            if i == max_reset_retry_on_invalid_sim_error - 1:
                raise RuntimeError(
                    f"Too many consecutive env reset error:"
                    f" {max_reset_retry_on_invalid_sim_error} times"
                )
        else:
            break

    return obs


def rotate_bounding_box(
    bounding_box: np.ndarray, quat: np.ndarray
) -> Tuple[float, float]:
    """ Rotates a bounding box by applying the quaternion and then re-computing the tightest
    possible fit of an *axis-aligned* bounding box.
    """
    pos, size = bounding_box

    # Compute 8 corners of bounding box.
    signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    corners = pos + signs * size
    assert corners.shape == (8, 3)

    # Rotate corners.
    mat = quat2mat(quat)
    rotated_corners = corners @ mat

    # Re-compute bounding-box.
    min_xyz = np.min(rotated_corners, axis=0)
    max_xyz = np.max(rotated_corners, axis=0)
    size = (max_xyz - min_xyz) / 2.0
    assert np.all(size >= 0.0)
    pos = min_xyz + size

    return pos, size


def update_object_z_coordinate(
    position: np.ndarray,
    object_bounding_boxes: np.ndarray,
    table_dimensions: Tuple[np.ndarray, np.ndarray, float],
) -> np.ndarray:
    """
    Update object z position based on bounding box. In case an object is rotated, the z position of
    object, which is computed to be on top of the table, can be invalidated. This method is
    useful to update z position based on up-to-date bounding box information.

    :param position: position of objects (num_objects, 3) where the second dimension of the
        tensor corresponds to (x, y, z) world coordinates of each object.
    :param object_bounding_boxes: matrix of bounding boxes (num_objects, 2, 3) where [:, 0, :]
        contains the center position of the bounding box in Cartesian space relative to the body's
        frame of reference and where [:, 1, :] contains the half-width, half-height, and half-depth
        of the object.
    :param table_dimensions: Tuple (table_pos, table_size, table_height) defining dimension of
        the table where
            table_pos: position of the table.
            table_size: half-size of the table along (x, y, z).
            table_height: height of the table.
    :return: position of objects (num_objects, 3) with updated z coordinate
    """
    table_pos, table_size, table_height = table_dimensions
    n_objects = object_bounding_boxes.shape[0]
    updated = position.copy()
    for i in range(n_objects):
        center, size = object_bounding_boxes[i]
        updated[i, -1] = size[-1] - center[-1] + table_size[-1] + table_pos[-1]
    return updated
