from typing import Tuple

import numpy as np
import trimesh


def get_vertices_bounding_box(vertices: np.ndarray) -> Tuple[float, float, float]:
    min_xyz = np.min(vertices, axis=0)
    max_xyz = np.max(vertices, axis=0)
    size = (max_xyz - min_xyz) / 2.0
    assert np.all(size >= 0.0)
    pos = min_xyz + size
    return pos, size, np.linalg.norm(size)


def subdivide_mesh(
    vertices: np.ndarray, faces: np.ndarray, subdivide_threshold: float
) -> np.ndarray:
    """
    Subdivide mesh into smaller triangles.

    :param vertices: Vertices of the mesh.
    :param faces: Faces of the mesh.
    :param subdivide_threshold: The max length for edges after the subdivision is
    defined as norm(bounding_box_size) * subdivide_threshold

    :return: Vertices after subdivision
    """

    max_edge = get_vertices_bounding_box(vertices)[-1] * subdivide_threshold

    return trimesh.remesh.subdivide_to_size(vertices, faces, max_edge)[0]
